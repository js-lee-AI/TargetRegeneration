import argparse
from dataclasses import dataclass
from typing import Any, Dict

try:
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES
except Exception:
    ATTENTION_BACKEND_CHOICES = [
        "flashinfer",
        "triton",
        "torch_native",
        "fa3",
        "flex_attention",
        "cutlass_mla",
        "flashmla",
        "trtllm_mha",
        "trtllm_mla",
        "ascend",
    ]


@dataclass
class TrackerArgs:
    report_to: str = "none"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--report-to", default="none", choices=["none"])


@dataclass
class SGLangBackendArgs:
    sglang_attention_backend: str = "fa3"
    sglang_mem_fraction_static: float = 0.4
    sglang_context_length: int = None
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_ep_size: int = 1
    sglang_max_running_requests: int = None
    sglang_max_total_tokens: int = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sglang-attention-backend",
            type=str,
            default="flashinfer",
            choices=ATTENTION_BACKEND_CHOICES,
        )
        parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
        parser.add_argument("--sglang-context-length", type=int, default=None)
        parser.add_argument("--sglang-enable-nccl-nvls", action="store_true")
        parser.add_argument("--sglang-enable-symm-mem", action="store_true")
        parser.add_argument("--sglang-enable-torch-compile", action="store_true")
        parser.add_argument("--sglang-enable-dp-attention", action="store_true")
        parser.add_argument("--sglang-enable-dp-lm-head", action="store_true")
        parser.add_argument("--sglang-ep-size", type=int, default=1)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SGLangBackendArgs":
        return SGLangBackendArgs(
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_context_length=args.sglang_context_length,
            sglang_enable_nccl_nvls=args.sglang_enable_nccl_nvls,
            sglang_enable_symm_mem=args.sglang_enable_symm_mem,
            sglang_enable_torch_compile=args.sglang_enable_torch_compile,
            sglang_enable_dp_attention=args.sglang_enable_dp_attention,
            sglang_enable_dp_lm_head=args.sglang_enable_dp_lm_head,
            sglang_ep_size=args.sglang_ep_size,
            sglang_max_running_requests=args.target_batch_size
            if hasattr(args, "target_batch_size")
            else None,
            sglang_max_total_tokens=args.target_batch_size * args.max_length
            if hasattr(args, "target_batch_size") and hasattr(args, "max_length")
            else None,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(
            attention_backend=self.sglang_attention_backend,
            mem_fraction_static=self.sglang_mem_fraction_static,
            context_length=self.sglang_context_length,
            enable_nccl_nvls=self.sglang_enable_nccl_nvls,
            enable_symm_mem=self.sglang_enable_symm_mem,
            enable_torch_compile=self.sglang_enable_torch_compile,
            enable_dp_attention=self.sglang_enable_dp_attention,
            enable_dp_lm_head=self.sglang_enable_dp_lm_head,
            ep_size=self.sglang_ep_size,
            max_running_requests=self.sglang_max_running_requests,
            max_total_tokens=self.sglang_max_total_tokens,
        )
