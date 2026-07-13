from contextlib import nullcontext

import torch
import triton
import triton.language as tl


_SUPPORTED_HEAD_DIMS = {64, 128}


def _launch_config(head_dim: int) -> dict[str, int]:
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"Linear attention kernel only supports head_dim in "
            f"{sorted(_SUPPORTED_HEAD_DIMS)}, got {head_dim}.")
    return {
        "BLOCK_M": 64,
        "num_warps": 4 if head_dim == 64 else 8,
        "num_stages": 2,
    }


@triton.jit
def _linear_attn_fwd_kernel(
    Q,
    KV,
    KSUM,
    O,
    DENOM,
    EPS,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    o_ptrs = O + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    denom_ptrs = DENOM + pid_bh * L + offs_m
    kv_ptrs = KV + pid_bh * D * D + offs_d[:, None] * D + offs_d[None, :]
    ksum_ptrs = KSUM + pid_bh * D + offs_d

    mask_m = offs_m < L
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    kv = tl.load(kv_ptrs).to(tl.float32)
    ksum = tl.load(ksum_ptrs).to(tl.float32)

    numerator = tl.dot(q, kv)
    denominator = tl.sum(q * ksum[None, :], axis=1) + EPS
    out = numerator / denominator[:, None]

    tl.store(o_ptrs, out.to(O.type.element_ty), mask=mask_m[:, None])
    tl.store(denom_ptrs, denominator, mask=mask_m)


@triton.jit
def _linear_attn_bwd_prep_kernel(
    DO,
    O,
    DENOM,
    DNUM,
    DDEN,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    do_ptrs = DO + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    o_ptrs = O + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    dnum_ptrs = DNUM + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    dden_ptrs = DDEN + pid_bh * L + offs_m
    denom_ptrs = DENOM + pid_bh * L + offs_m

    mask_m = offs_m < L
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    denom = tl.load(denom_ptrs, mask=mask_m, other=1.0).to(tl.float32)

    dnum = do / denom[:, None]
    dden = -tl.sum(do * o, axis=1) / denom

    tl.store(dnum_ptrs, dnum.to(DNUM.type.element_ty), mask=mask_m[:, None])
    tl.store(dden_ptrs, dden, mask=mask_m)


@triton.jit
def _linear_attn_bwd_dq_kernel(
    DNUM,
    DDEN,
    KV,
    KSUM,
    DQ,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    dnum_ptrs = DNUM + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    dden_ptrs = DDEN + pid_bh * L + offs_m
    dq_ptrs = DQ + pid_bh * L * D + offs_m[:, None] * D + offs_d[None, :]
    kv_ptrs = KV + pid_bh * D * D + offs_d[:, None] * D + offs_d[None, :]
    ksum_ptrs = KSUM + pid_bh * D + offs_d

    mask_m = offs_m < L
    dnum = tl.load(dnum_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    dden = tl.load(dden_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    kv = tl.load(kv_ptrs).to(tl.float32)
    ksum = tl.load(ksum_ptrs).to(tl.float32)

    dq = tl.dot(dnum, tl.trans(kv)) + dden[:, None] * ksum[None, :]
    tl.store(dq_ptrs, dq.to(DQ.type.element_ty), mask=mask_m[:, None])


class _TritonLinearAttentionFn(torch.autograd.Function):
    """Exact linear-attention core with Triton kernels for tokenwise work."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise RuntimeError("Linear attention Triton kernels require CUDA/ROCm.")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise RuntimeError("Linear attention expects contiguous q/k/v tensors.")

        bh, seq_len, head_dim = q.shape
        cfg = _launch_config(head_dim)

        kv = torch.bmm(k.transpose(1, 2), v)
        ksum = k.sum(dim=1)

        out = torch.empty_like(q)
        denom = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(seq_len, cfg["BLOCK_M"]), bh)
        _linear_attn_fwd_kernel[grid](
            q,
            kv,
            ksum,
            out,
            denom,
            eps,
            seq_len,
            D=head_dim,
            BLOCK_M=cfg["BLOCK_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

        ctx.save_for_backward(q, k, v, kv, ksum, out, denom)
        ctx.cfg = cfg
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, kv, ksum, out, denom = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        bh, seq_len, head_dim = q.shape
        cfg = ctx.cfg
        grid = (triton.cdiv(seq_len, cfg["BLOCK_M"]), bh)

        grad_num = torch.empty_like(grad_out)
        grad_denom = torch.empty((bh, seq_len), device=grad_out.device, dtype=torch.float32)
        grad_q = torch.empty_like(q)

        _linear_attn_bwd_prep_kernel[grid](
            grad_out,
            out,
            denom,
            grad_num,
            grad_denom,
            seq_len,
            D=head_dim,
            BLOCK_M=cfg["BLOCK_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
        _linear_attn_bwd_dq_kernel[grid](
            grad_num,
            grad_denom,
            kv,
            ksum,
            grad_q,
            seq_len,
            D=head_dim,
            BLOCK_M=cfg["BLOCK_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

        grad_kv = torch.bmm(q.transpose(1, 2), grad_num)
        grad_ksum = torch.bmm(
            grad_denom.to(dtype=q.dtype).unsqueeze(1),
            q,
        ).squeeze(1)
        grad_k = torch.bmm(v, grad_kv.transpose(1, 2)) + grad_ksum[:, None, :]
        grad_v = torch.bmm(k, grad_kv)

        return grad_q, grad_k, grad_v, None


def sla_v2_linear_branch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_l: torch.nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Kernel-level pure linear-attention branch.

    This keeps the same linear-attention math as SPARSE_LINEAR_ATTN's linear
    branch, but moves the tokenwise forward and dQ backward hot path into
    Triton kernels.
    """
    b, h, seq_len, head_dim = q.shape

    q_bh = q.reshape(b * h, seq_len, head_dim).to(compute_dtype).contiguous()
    k_bh = k.reshape(b * h, seq_len, head_dim).to(compute_dtype).contiguous()
    v_bh = v.reshape(b * h, seq_len, head_dim).to(compute_dtype).contiguous()

    o_l = _TritonLinearAttentionFn.apply(q_bh, k_bh, v_bh, eps)
    o_l = o_l.reshape(b, h, seq_len, head_dim)

    amp_context = (
        torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        if o_l.device.type == "cuda"
        else nullcontext()
    )
    with amp_context:
        o_l = proj_l(o_l)
    return o_l
