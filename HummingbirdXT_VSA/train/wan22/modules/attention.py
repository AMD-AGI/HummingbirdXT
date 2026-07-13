# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import functools
import importlib
import math
import os
import sys
import warnings

import torch

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TRAIN_SLA_ROOT = os.path.join(_REPO_ROOT, "train", "SLA")
_ORIG_TURBO_SLA_ROOT = os.path.join(
    _REPO_ROOT, "ori_sparse_attention", "TurboDiffusion", "turbodiffusion")
_VSA_KERNEL_ROOT = os.path.join(_REPO_ROOT, "local_vsa_kernels")
_AMD_VSA_OPT_KERNEL_DIR = os.path.join(
    _REPO_ROOT, "amd_VSA", "FastVideo", "fastvideo-kernel", "python")
_VSA_OPT_KERNEL_DIR = os.environ.get(
    "FASTVIDEO_VSA_OPT_KERNEL_DIR",
    _AMD_VSA_OPT_KERNEL_DIR)
_VSA_ORIG_KERNEL_DIR = os.environ.get(
    "FASTVIDEO_VSA_ORIG_KERNEL_DIR",
    os.path.join(_VSA_KERNEL_ROOT, "original"))
_ATTN_BACKEND = os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")


def _prepend_sys_path(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _select_vsa_kernel_dir(attention_backend):
    if attention_backend in {
        "VIDEO_SPARSE_ATTN",
        "SPARSE_LINEAR_ATTN",
    }:
        return _VSA_OPT_KERNEL_DIR
    if attention_backend == "VIDEO_SPARSE_ATTN_ORIG":
        return _VSA_ORIG_KERNEL_DIR
    return None


def _validate_vsa_kernel_dir(kernel_dir):
    if kernel_dir is None:
        return
    package_dir = os.path.join(kernel_dir, "fastvideo_kernel")
    if not os.path.isdir(package_dir):
        raise ImportError(
            f"Expected a fastvideo_kernel install under {kernel_dir}, "
            "but the package directory is missing. "
            "Run train/running_scripts/train/install_vsa_kernel_variants.sh "
            "or set FASTVIDEO_VSA_*_KERNEL_DIR to a valid --target install.")


_SELECTED_VSA_KERNEL_DIR = _select_vsa_kernel_dir(_ATTN_BACKEND)
_VSA_IMPORT_ERROR = None
_VSA_ORIG_KERNEL_LOGGED = False
if _SELECTED_VSA_KERNEL_DIR is not None:
    _validate_vsa_kernel_dir(_SELECTED_VSA_KERNEL_DIR)
    _prepend_sys_path(_SELECTED_VSA_KERNEL_DIR)

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    fastvideo_kernel = importlib.import_module("fastvideo_kernel")
    video_sparse_attn = fastvideo_kernel.video_sparse_attn
    _FASTVIDEO_KERNEL_PATH = os.path.abspath(getattr(fastvideo_kernel, "__file__", ""))
    if _SELECTED_VSA_KERNEL_DIR is not None:
        selected_dir = os.path.abspath(_SELECTED_VSA_KERNEL_DIR)
        if not _FASTVIDEO_KERNEL_PATH.startswith(selected_dir):
            raise ImportError(
                f"Loaded fastvideo_kernel from {_FASTVIDEO_KERNEL_PATH}, "
                f"expected it under {selected_dir}.")
        print(
            f"Using {_ATTN_BACKEND} fastvideo_kernel from {_FASTVIDEO_KERNEL_PATH}")
    VIDEO_SPARSE_ATTN_AVAILABLE = True
except ImportError:
    video_sparse_attn = None
    VIDEO_SPARSE_ATTN_AVAILABLE = False
    _VSA_IMPORT_ERROR = sys.exc_info()[1]
    if _SELECTED_VSA_KERNEL_DIR is not None:
        warnings.warn(
            f"Unable to import fastvideo_kernel for {_ATTN_BACKEND} from "
            f"{_SELECTED_VSA_KERNEL_DIR}: {_VSA_IMPORT_ERROR}")


@functools.lru_cache(maxsize=2)
def _load_video_sparse_attn_from_dir(kernel_dir):
    _validate_vsa_kernel_dir(kernel_dir)
    kernel_dir = os.path.abspath(kernel_dir)
    package_dir = os.path.join(kernel_dir, "fastvideo_kernel")
    existing_module = sys.modules.get("fastvideo_kernel")
    existing_path = os.path.abspath(
        getattr(existing_module, "__file__", "")) if existing_module is not None else ""
    if existing_path.startswith(package_dir):
        op = getattr(existing_module, "video_sparse_attn", None)
        if op is None:
            raise ImportError(
                f"fastvideo_kernel under {kernel_dir} does not export video_sparse_attn.")
        return op, existing_path

    cached_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "fastvideo_kernel" or name.startswith("fastvideo_kernel.")
    }
    try:
        for name in list(cached_modules):
            sys.modules.pop(name, None)
        _prepend_sys_path(kernel_dir)
        importlib.invalidate_caches()
        module = importlib.import_module("fastvideo_kernel")
    except Exception:
        for name in list(sys.modules):
            if name == "fastvideo_kernel" or name.startswith("fastvideo_kernel."):
                sys.modules.pop(name, None)
        sys.modules.update(cached_modules)
        raise

    loaded_path = os.path.abspath(getattr(module, "__file__", ""))
    if not loaded_path.startswith(package_dir):
        raise ImportError(
            f"Loaded fastvideo_kernel from {loaded_path}, expected it under {package_dir}.")
    op = getattr(module, "video_sparse_attn", None)
    if op is None:
        raise ImportError(
            f"fastvideo_kernel under {kernel_dir} does not export video_sparse_attn.")
    return op, loaded_path


try:
    _prepend_sys_path(_TRAIN_SLA_ROOT)
    from sparse_linear_attention import SparseLinearAttention
    SPARSE_LINEAR_ATTN_AVAILABLE = True
    print("SparseLinearAttention is available!!!!!!!!!!!!!!!!!!!!!!!!!")
except ImportError:
    SparseLinearAttention = None
    SPARSE_LINEAR_ATTN_AVAILABLE = False

try:
    _prepend_sys_path(_ORIG_TURBO_SLA_ROOT)
    from SLA import SparseLinearAttention as OriginalSparseLinearAttention
    ORIGINAL_SPARSE_LINEAR_ATTN_AVAILABLE = True
except ImportError:
    OriginalSparseLinearAttention = None
    ORIGINAL_SPARSE_LINEAR_ATTN_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
    'video_sparse_attention',
    'video_sparse_attention_original',
    'VIDEO_SPARSE_ATTN_AVAILABLE',
    'SparseLinearAttention',
    'SPARSE_LINEAR_ATTN_AVAILABLE',
    'OriginalSparseLinearAttention',
    'ORIGINAL_SPARSE_LINEAR_ATTN_AVAILABLE',
]

VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=10)
def _get_tile_partition_indices(dit_seq_shape, tile_size, device):
    """Build a 1-D index that reorders tokens into tile-major order."""
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(
                    indices[t * ts:min(t * ts + ts, T),
                            h * hs:min(h * hs + hs, H),
                            w * ws:min(w * ws + ws, W)].flatten())
    return torch.cat(ls, dim=0)


@functools.lru_cache(maxsize=10)
def _get_reverse_tile_partition_indices(dit_seq_shape, tile_size, device):
    return torch.argsort(
        _get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def _construct_variable_block_sizes(dit_seq_shape, num_tiles, device):
    """Number of valid tokens inside every (ts_t x ts_h x ts_w) tile."""
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = VSA_TILE_SIZE
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len, tile, n):
        sizes = torch.full((n,), tile, dtype=torch.int, device=device)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)
    h_sizes = _sizes(h, ts_h, n_h)
    w_sizes = _sizes(w, ts_w, n_w)

    return (
        t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]
    ).reshape(-1)


@functools.lru_cache(maxsize=10)
def _get_non_pad_index(variable_block_sizes, max_block_size):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def _tile(x, num_tiles, tile_partition_indices, non_pad_index):
    """Rearrange tokens into tile order with zero-padding for boundary tiles."""
    padded_len = math.prod(n * t for n, t in zip(num_tiles, VSA_TILE_SIZE))
    x_padded = torch.zeros(
        (x.shape[0], padded_len, x.shape[-2], x.shape[-1]),
        device=x.device, dtype=x.dtype)
    x_padded[:, non_pad_index] = x[:, tile_partition_indices]
    return x_padded


def _untile(x, reverse_tile_partition_indices, non_pad_index):
    """Reverse tile ordering back to original token order."""
    return x[:, non_pad_index][:, reverse_tile_partition_indices]


def video_sparse_attention(
    q, k, v, gate_compress,
    grid_sizes, seq_lens=None, VSA_sparsity=0.8,
    dtype=torch.bfloat16,
    video_sparse_attn_op=None,
    import_error=None,
):
    """
    Video Sparse Attention (VSA) — drop-in replacement for flash_attention
    in self-attention layers.

    @param q: Query tensor [B, L, N, D] (after RoPE).
    @param k: Key tensor [B, L, N, D] (after RoPE).
    @param v: Value tensor [B, L, N, D].
    @param gate_compress: Compression gate [B, L, N, D].
    @param grid_sizes: Per-sample (F, H, W) grid dimensions [B, 3].
    @param seq_lens: Actual sequence lengths per sample [B] (unused, for API compat).
    @param VSA_sparsity: Sparsity ratio (0.0 = dense, 1.0 = fully sparse).
    @param dtype: Compute dtype (float16 or bfloat16).
    @return: Attention output [B, L, N, D].
    """
    op = video_sparse_attn if video_sparse_attn_op is None else video_sparse_attn_op
    error_source = _VSA_IMPORT_ERROR if import_error is None else import_error
    if op is None:
        error_detail = ""
        if error_source is not None:
            error_detail = f" Import error: {error_source}"
        raise ImportError(
            "fastvideo_kernel.video_sparse_attn is required for VSA. "
            "Install fastvideo-kernel or set FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN."
            f"{error_detail}")

    half_dtypes = (torch.float16, torch.bfloat16)
    b, total_len, n, d = q.shape
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    F_dim, H_dim, W_dim = (int(v) for v in grid_sizes[0].tolist())
    actual_len = F_dim * H_dim * W_dim
    dit_seq_shape = (F_dim, H_dim, W_dim)
    device = q.device

    num_tiles = tuple(
        math.ceil(s / t) for s, t in zip(dit_seq_shape, VSA_TILE_SIZE))

    tile_idx = _get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    rev_idx = _get_reverse_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    var_blk = _construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
    npi = _get_non_pad_index(var_blk, math.prod(VSA_TILE_SIZE))

    # Extract valid region and cast to half precision
    q_v, k_v, v_v, gc_v = (half(t[:, :actual_len]) for t in (q, k, v, gate_compress))

    # Tile → transpose to [B, N, S, D]
    q_t = _tile(q_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()
    k_t = _tile(k_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()
    v_t = _tile(v_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()
    gc_t = _tile(gc_v, num_tiles, tile_idx, npi).transpose(1, 2).contiguous()

    cur_topk = math.ceil(
        (1 - VSA_sparsity) * (actual_len / math.prod(VSA_TILE_SIZE)))

    # VSA CUDA kernel: expects [B, N, S, D], returns [B, N, S, D]
    output = op(
        q_t, k_t, v_t,
        var_blk, var_blk, cur_topk,
        block_size=VSA_TILE_SIZE,
        compress_attn_weight=gc_t,
    ).transpose(1, 2)  # → [B, S_padded, N, D]

    output = _untile(output, rev_idx, npi)  # → [B, actual_len, N, D]

    # Pad back to original sequence length if needed
    if total_len > actual_len:
        output = torch.cat([
            output,
            output.new_zeros(b, total_len - actual_len, n, d),
        ], dim=1)

    return output.type(out_dtype)


def video_sparse_attention_original(
    q, k, v, gate_compress,
    grid_sizes, seq_lens=None, VSA_sparsity=0.8,
    dtype=torch.bfloat16,
):
    """Original FastVideo VSA backend loaded from the original kernel dir."""
    global _VSA_ORIG_KERNEL_LOGGED
    original_import_error = None
    original_video_sparse_attn = None
    try:
        original_video_sparse_attn, kernel_path = _load_video_sparse_attn_from_dir(
            _VSA_ORIG_KERNEL_DIR)
        if not _VSA_ORIG_KERNEL_LOGGED:
            print(
                "Using VIDEO_SPARSE_ATTN_ORIG fastvideo_kernel from "
                f"{kernel_path}")
            _VSA_ORIG_KERNEL_LOGGED = True
    except ImportError as exc:
        original_import_error = exc
    return video_sparse_attention(
        q=q,
        k=k,
        v=v,
        gate_compress=gate_compress,
        grid_sizes=grid_sizes,
        seq_lens=seq_lens,
        VSA_sparsity=VSA_sparsity,
        dtype=dtype,
        video_sparse_attn_op=original_video_sparse_attn,
        import_error=original_import_error,
    )


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
