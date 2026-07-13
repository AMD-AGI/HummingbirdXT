import math
import importlib

import torch

from .attention import (
    VSA_TILE_SIZE,
    _construct_variable_block_sizes,
    _get_non_pad_index,
    _get_reverse_tile_partition_indices,
    _get_tile_partition_indices,
    _tile,
    _untile,
)

_FASTVIDEO_VSA_OPS = None


def _get_fastvideo_vsa_ops():
    global _FASTVIDEO_VSA_OPS
    if _FASTVIDEO_VSA_OPS is None:
        try:
            _FASTVIDEO_VSA_OPS = importlib.import_module("fastvideo_kernel.ops")
        except ImportError as exc:
            raise ImportError(
                "fastvideo_kernel.ops is required for the deprecated SLA-v2 sparse path. "
                "Run train/running_scripts/train/install_vsa_kernel_variants.sh "
                "and make sure FASTVIDEO_VSA_OPT_KERNEL_DIR points to the local install."
            ) from exc
    return _FASTVIDEO_VSA_OPS


def _video_sparse_attn_v2_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Copied VSA operator path for the deprecated SLA-v2 sparse branch.

    Compared with fastvideo_kernel.ops.video_sparse_attn, this version:
    - keeps the same math
    - uses BH-flattened batched GEMMs for the compression branch
    - skips the unused map_to_index conversion on the autograd-enabled path
    """
    fastvideo_ops = _get_fastvideo_vsa_ops()

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    if v.shape[2] != kv_seq_len:
        raise ValueError(
            f"Expected k and v to have the same sequence length, got "
            f"k.shape[2]={kv_seq_len}, v.shape[2]={v.shape[2]}"
        )
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("Expected q/k/v to have the same batch and head dimensions.")

    if q_seq_len % block_elements != 0 or kv_seq_len % block_elements != 0:
        raise ValueError(
            f"q_seq_len and kv_seq_len must be divisible by block_elements={block_elements}, "
            f"got q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}"
        )

    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements

    if variable_block_sizes.numel() != kv_num_blocks:
        raise ValueError(
            f"variable_block_sizes must have length kv_num_blocks={kv_num_blocks}, "
            f"got {variable_block_sizes.numel()}"
        )
    if q_variable_block_sizes.numel() != q_num_blocks:
        raise ValueError(
            f"q_variable_block_sizes must have length q_num_blocks={q_num_blocks}, "
            f"got {q_variable_block_sizes.numel()}"
        )

    q_c = q.view(batch, heads, q_num_blocks, block_elements, dim)
    k_c = k.view(batch, heads, kv_num_blocks, block_elements, dim)
    v_c = v.view(batch, heads, kv_num_blocks, block_elements, dim)

    q_c = (q_c.float().sum(dim=3) / q_variable_block_sizes.view(1, 1, -1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(v.dtype)

    bh = batch * heads
    q_bh = q_c.reshape(bh, q_num_blocks, dim)
    k_bh = k_c.reshape(bh, kv_num_blocks, dim)
    v_bh = v_c.reshape(bh, kv_num_blocks, dim)

    scores = torch.bmm(q_bh, k_bh.transpose(1, 2)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.bmm(attn, v_bh)
    out_c = (
        out_c.reshape(batch, heads, q_num_blocks, 1, dim)
        .expand(batch, heads, q_num_blocks, block_elements, dim)
        .reshape(batch, heads, q_seq_len, dim)
    )

    topk_idx = torch.topk(scores, topk, dim=-1).indices

    if fastvideo_ops.block_sparse_fwd is not None:
        mask = torch.zeros(
            (batch, heads, q_num_blocks, kv_num_blocks),
            dtype=torch.bool,
            device=scores.device,
        )
        mask.scatter_(-1, topk_idx.reshape(batch, heads, q_num_blocks, topk), True)
        out_s = fastvideo_ops.block_sparse_attn(q, k, v, mask, variable_block_sizes)[0]
    else:
        idx = topk_idx.reshape(batch, heads, q_num_blocks, topk).to(torch.int32)
        num = torch.full(
            (batch, heads, q_num_blocks),
            topk,
            dtype=torch.int32,
            device=idx.device,
        )
        out_s, _ = fastvideo_ops.triton_block_sparse_attn_forward(
            q,
            k,
            v,
            idx,
            num,
            variable_block_sizes,
        )

    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s


def video_sparse_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_compress: torch.Tensor,
    grid_sizes: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
    VSA_sparsity: float = 0.8,
    dtype: torch.dtype = torch.bfloat16,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    b, total_len, n, d = q.shape
    out_dtype = q.dtype

    def half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    f_dim, h_dim, w_dim = (int(val) for val in grid_sizes[0].tolist())
    actual_len = f_dim * h_dim * w_dim
    dit_seq_shape = (f_dim, h_dim, w_dim)
    device = q.device

    num_tiles = tuple(math.ceil(s / t) for s, t in zip(dit_seq_shape, VSA_TILE_SIZE))
    tile_idx = _get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    rev_idx = _get_reverse_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    var_blk = _construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
    npi = _get_non_pad_index(var_blk, math.prod(VSA_TILE_SIZE))

    q_v, k_v, v_v, gc_v = (half(t[:, :actual_len]) for t in (q, k, v, gate_compress))

    q_t = _tile(q_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()
    k_t = _tile(k_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()
    v_t = _tile(v_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()
    gc_t = _tile(gc_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()

    cur_topk = math.ceil((1 - VSA_sparsity) * (actual_len / math.prod(VSA_TILE_SIZE)))

    output = _video_sparse_attn_v2_core(
        q=q_t,
        k=k_t,
        v=v_t,
        variable_block_sizes=var_blk,
        q_variable_block_sizes=var_blk,
        topk=cur_topk,
        block_size=VSA_TILE_SIZE,
        compress_attn_weight=gc_t,
    ).transpose(1, 2)

    output = _untile(output, rev_idx, npi)

    if total_len > actual_len:
        output = torch.cat(
            [output, output.new_zeros(b, total_len - actual_len, n, d)],
            dim=1,
        )

    return output.type(out_dtype)


def sla_v2_sparse_branch(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_compress_layer: torch.nn.Module,
    grid_sizes: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
    VSA_sparsity: float = 0.8,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Copied gate_compress + VSA path for the deprecated SLA-v2 sparse branch."""
    b, s, n, d = q.shape
    gate_compress = gate_compress_layer(x).view(b, s, n, d)
    return video_sparse_attention_v2(
        q=q,
        k=k,
        v=v,
        gate_compress=gate_compress,
        grid_sizes=grid_sizes,
        seq_lens=seq_lens,
        VSA_sparsity=VSA_sparsity,
        dtype=dtype,
    )
