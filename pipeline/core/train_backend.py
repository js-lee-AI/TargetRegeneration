import argparse
import functools
import logging
import math
import contextlib
import os
import shutil
import time
import warnings
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig
from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import FINAL_HIDDEN_LOSS_TYPES, OnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import (
    get_last_checkpoint,
    get_local_device,
    load_tokenizer,
    print_on_rank0,
    print_with_rank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend", type=str, default="hf", choices=["sglang", "hf"]
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--draft-init-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument("--mask-token-id", type=int, default=None)
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
    )
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument("--loss-decay-gamma", type=float, default=None)
    model_group.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=["dflash", "dflash_targetkd"],
    )
    model_group.add_argument("--dflash-kd-topk", type=int, default=8)
    model_group.add_argument("--dflash-kd-hard-label-mix", type=float, default=0.0)
    model_group.add_argument("--dflash-init-l2", type=float, default=0.0)
    model_group.add_argument(
        "--dflash-kd-tail-bucket",
        dest="dflash_kd_tail_bucket",
        action="store_true",
        default=True,
    )
    model_group.add_argument(
        "--no-dflash-kd-tail-bucket", dest="dflash_kd_tail_bucket", action="store_false"
    )
    model_group.add_argument("--dflash-kd-lambda-scale", type=float, default=1.0)
    model_group.add_argument("--dflash-kd-chunk-size", type=int, default=4096)
    model_group.add_argument(
        "--dflash-kd-survival-mask", action="store_true", default=False
    )
    model_group.add_argument(
        "--dflash-kd-survival-soft", action="store_true", default=False
    )
    model_group.add_argument(
        "--dflash-kd-anchor-rollout", action="store_true", default=False
    )
    model_group.add_argument("--dflash-kd-rollout-steps", type=int, default=0)
    model_group.add_argument(
        "--dflash-kd-inrollout-anchors", action="store_true", default=False
    )
    model_group.add_argument(
        "--dflash-kd-slotmatch-control", action="store_true", default=False
    )
    model_group.add_argument(
        "--dflash-kd-rollout-survival-soft", action="store_true", default=False
    )
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)
    model_group.add_argument("--is-vlm", action="store_true")
    model_group.add_argument("--min-pixels", type=int, default=50176)
    model_group.add_argument("--max-pixels", type=int, default=200704)
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=0.0006)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--data-order-seed", type=int, default=0)
    training_group.add_argument("--max-train-steps", type=int, default=0)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--log-grad-norm", action="store_true")
    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)
    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)
    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)
    return parser.parse_args()


def build_models(args) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )
    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    device = get_local_device()
    device_type = device.type
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device_type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        is_vlm=args.is_vlm,
        **target_model_kwargs,
    )
    _tp = next(target_model.model.parameters())
    print_on_rank0(
        f"Target model: dtype={_tp.dtype} device={_tp.device} params={sum((p.numel() for p in target_model.model.parameters())) / 1000000000.0:.2f}B"
    )
    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: checkpoint block_size ({draft_config.block_size}) differs from command-line arg ({args.block_size}). Using checkpoint value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")
    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}
    args.loss_type = (
        args.loss_type
        or draft_config.dflash_config.get("training_mode")
        or draft_config.dflash_config.get("loss_type")
        or "dflash"
    )
    if args.loss_type in FINAL_HIDDEN_LOSS_TYPES and args.target_model_backend != "hf":
        raise ValueError(
            f"loss_type={args.loss_type!r} requires --target-model-backend hf (the sglang backend does not return final_hidden_states); got --target-model-backend {args.target_model_backend!r}."
        )
    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")
    print_on_rank0(f"Using DFlash training loss_type: {args.loss_type}")
    if args.dflash_init_l2 > 0.0 and (args.resume or not args.draft_init_path):
        raise ValueError("--dflash-init-l2 anchors to the released drafter; it needs --draft-init-path and no --resume")
    draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)
    if args.draft_init_path and (not args.resume):
        init = DFlashDraftModel.from_pretrained(
            args.draft_init_path, dtype=torch.bfloat16
        )
        draft_model.load_state_dict(init.state_dict(), strict=True)
        del init
        draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
        print_on_rank0(f"Warm-started the drafter from {args.draft_init_path}")
    elif args.draft_init_path:
        print_on_rank0(
            "--draft-init-path ignored: --resume takes precedence, this run is continuing its own checkpoints"
        )
    target_model.set_capture_layers(draft_model.target_layer_ids)
    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, num_hidden_layers={draft_config.num_hidden_layers}, num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum((p.numel() for p in draft_model.parameters())):,}"
    )
    return (target_model, draft_model)


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    import hashlib

    cache_params_string = f"{args.train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}"
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    processor = None
    if args.is_vlm:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
        is_vlm=args.is_vlm,
        processor=processor,
    )
    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
        sampler_seed=args.data_order_seed,
    )
    print_on_rank0(
        f"Data order seed {args.data_order_seed} (--seed {args.seed} sets the anchors and torch randomness only)"
    )
    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )
    return (train_dataloader, eval_dataloader)


def _schedule_total_steps(
    num_epochs, steps_per_epoch, max_train_steps, accumulation_steps=1
):
    total = num_epochs * steps_per_epoch
    if max_train_steps > 0:
        total = min(total, math.ceil(max_train_steps / accumulation_steps))
    return total


def _reached_max_steps(global_step, max_train_steps):
    return 0 < max_train_steps <= global_step


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    _sd_ctx = (
        FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT)
        if isinstance(dflash_model, FSDP)
        else contextlib.nullcontext()
    )
    with _sd_ctx:
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }
        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )
            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)
            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)
            print_on_rank0(f"Saved checkpoint to {save_dir}")
    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
    grad_norm: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    logdict = {}
    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if grad_norm is not None:
        logdict[f"{mode}/grad_norm"] = grad_norm
    for k, v in (extra or {}).items():
        logdict[f"{mode}/{k}"] = v
    suffix = "".join((f", {k}: {v:.4f}" for k, v in (extra or {}).items()))
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}{suffix}"
    )
    tracker.log(logdict, step=global_step)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")
    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")
    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path
    target_model, draft_model = build_models(args)
    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print("Loaded draft model weights from checkpoint")
        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print(
                f"Will resume from epoch {resume_state['epoch']}, step {resume_state['global_step']}"
            )
    tokenizer = load_tokenizer(args.target_model_path)
    rollout_stop_ids = None
    if args.dflash_kd_anchor_rollout:
        stop = set()
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and (tid != tokenizer.unk_token_id):
                stop.add(tid)
        if tokenizer.eos_token_id is not None:
            stop.add(int(tokenizer.eos_token_id))
        rollout_stop_ids = tuple(sorted(stop))
        print_on_rank0(f"[ALR] anchor rollout on; end-of-turn ids {rollout_stop_ids}")
    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif (
        dflash_config := getattr(draft_model.config, "dflash_config", {})
    ) and dflash_config.get("mask_token_id") is not None:
        mask_token_id = dflash_config["mask_token_id"]
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")
    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = _schedule_total_steps(
        args.num_epochs, steps_per_epoch, args.max_train_steps, args.accumulation_steps
    )
    print_on_rank0(
        f"Total training steps: {total_steps}"
        + (
            f" (--max-train-steps {args.max_train_steps})"
            if args.max_train_steps > 0
            else ""
        )
    )
    print_on_rank0("Loading target embeddings and head...")
    device = get_local_device()
    device_type = device.type
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device_type,
        trust_remote_code=args.trust_remote_code,
    )
    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        kd_topk=args.dflash_kd_topk,
        kd_hard_label_mix=args.dflash_kd_hard_label_mix,
        kd_tail_bucket=args.dflash_kd_tail_bucket,
        kd_lambda_scale=args.dflash_kd_lambda_scale,
        kd_chunk_size=args.dflash_kd_chunk_size,
        kd_survival_mask=args.dflash_kd_survival_mask,
        kd_survival_soft=args.dflash_kd_survival_soft,
        kd_anchor_rollout=args.dflash_kd_anchor_rollout,
        kd_rollout_stop_ids=rollout_stop_ids,
        kd_rollout_steps=args.dflash_kd_rollout_steps,
        kd_inrollout_anchors=args.dflash_kd_inrollout_anchors,
        kd_inrollout_seed=args.seed,
        kd_slotmatch_control=args.dflash_kd_slotmatch_control,
        kd_rollout_survival_soft=args.dflash_kd_rollout_survival_soft,
    )
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    if os.environ.get("DF_REPLICATED_DP") == "1":
        fsdp_kwargs["sharding_strategy"] = ShardingStrategy.NO_SHARD
        print_with_rank("DF_REPLICATED_DP=1: replicated data parallel training")
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dflash_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=block_classes
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit FSDP wrap (no compute-comm overlap)."
        )
    if dist.get_world_size() == 1 and os.environ.get("DF_NO_FSDP"):
        print_with_rank("DF_NO_FSDP=1: single GPU, using plain module (no FSDP wrap)")
    else:
        dflash_model = FSDP(dflash_model, **fsdp_kwargs)
        print_with_rank("Initialized FSDP")
    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]
    _optimized = draft_model
    optimizer = BF16Optimizer(
        _optimized,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
        init_l2=args.dflash_init_l2,
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: epoch={start_epoch}, step={global_step}, lr={optimizer.get_learning_rate():.6f}"
        )
    skip_steps = global_step - start_epoch * len(train_dataloader)
    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")
    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()
        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader
        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1
            input_ids = data["input_ids"].to(device, non_blocking=True)
            attention_mask = data["attention_mask"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"].to(device, non_blocking=True)
            target_gen_kwargs = {}
            if args.is_vlm:
                target_gen_kwargs["pixel_values"] = data["pixel_values"].to(
                    device, non_blocking=True
                )
                target_gen_kwargs["image_grid_thw"] = data["image_grid_thw"].to(
                    device, non_blocking=True
                )
            if args.loss_type in FINAL_HIDDEN_LOSS_TYPES:
                target_gen_kwargs["return_final_hidden"] = True
            rollout_kwargs = {}
            if args.dflash_kd_anchor_rollout:
                _inner = getattr(dflash_model, "module", dflash_model)
                if args.dflash_kd_inrollout_anchors:
                    _ira = _inner._sample_inrollout_anchors(
                        input_ids.shape[1], loss_mask, device
                    )
                    _anchors, _keep = (_ira["primary"], _ira["keep1"])
                    target_gen_kwargs["rollout_context_steps"] = _inner.block_size - 3
                elif args.dflash_kd_slotmatch_control:
                    _anchors, _keep, _cap = _inner._sample_slotmatched_anchors(
                        input_ids.shape[1], loss_mask, device
                    )
                else:
                    _anchors, _keep = _inner._sample_anchor_positions(
                        input_ids.shape[1], loss_mask, device
                    )
                target_gen_kwargs["anchor_rollout"] = (_anchors, _inner.block_size)
                if args.dflash_kd_rollout_steps:
                    target_gen_kwargs["rollout_steps"] = args.dflash_kd_rollout_steps
                rollout_kwargs = dict(anchor_positions=_anchors, block_keep_mask=_keep)
                if args.dflash_kd_inrollout_anchors:
                    rollout_kwargs["inrollout"] = _ira
                if args.dflash_kd_slotmatch_control:
                    rollout_kwargs["slot_cap"] = _cap
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask, **target_gen_kwargs
            )
            if args.dflash_kd_anchor_rollout:
                rollout_kwargs["rollout_hidden"] = target_output.rollout_hidden
                rollout_kwargs["rollout_tokens"] = target_output.rollout_tokens
                if target_output.rollout_context is not None:
                    rollout_kwargs["rollout_context"] = target_output.rollout_context
            hidden_states = target_output.hidden_states.to(device, non_blocking=True)
            final_hidden_states = None
            if target_output.final_hidden_states is not None:
                final_hidden_states = target_output.final_hidden_states.to(
                    device, non_blocking=True
                )
            loss, accuracy = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                final_hidden_states=final_hidden_states,
                **rollout_kwargs,
            )
            (loss / args.accumulation_steps).backward()
            grad_norm_log = None
            if args.log_grad_norm and global_step % args.log_interval == 0:
                _gsq = 0.0
                for p in optimizer.model.parameters():
                    if p.grad is not None:
                        _gsq += float(p.grad.detach().float().pow(2).sum())
                grad_norm_log = _gsq**0.5
            if global_step % args.accumulation_steps == 0:
                optimizer.step()
            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()
                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                    grad_norm=grad_norm_log,
                )
            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dflash_model, draft_model, optimizer
                )
            if _reached_max_steps(global_step, args.max_train_steps):
                break
        if _reached_max_steps(global_step, args.max_train_steps):
            break
    save_checkpoint(
        args, args.num_epochs, global_step, dflash_model, draft_model, optimizer
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
