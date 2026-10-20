import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

# Objective flags of the paper. CE trains each slot on its corpus token; every KD objective uses the
# target's eight most probable tokens plus a tail outcome, and all of them share the slot weight
# exp(-(k - 1) / gamma) of the config.
KD = ["--loss-type", "dflash_targetkd"]
OBJECTIVES = {
    "ce": ["--loss-type", "dflash"],
    "kd": KD,
    "gate": KD + ["--dflash-kd-survival-mask"],
    "soft": KD + ["--dflash-kd-survival-soft"],
    "greedy_labels": KD + ["--dflash-kd-hard-label-mix", "1.0"],
}


def build_command(args, config, draft_init):
    if args.gpus < 1 or args.microbatch < 1:
        raise ValueError("GPU count and microbatch must be positive")
    denom = args.gpus * args.microbatch
    if config["global_batch_size"] % denom:
        raise ValueError("GPU count times microbatch must divide the global batch")
    if args.random_init and args.l2_anchor:
        raise ValueError("The L2 anchor pulls toward the released drafter, so it needs the released initialization")
    gamma = config["gamma"] if args.gamma is None else args.gamma
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.gpus}",
        "-m",
        "core.train_backend",
        "--target-model-path",
        args.target or config["target"],
        "--draft-config-path",
        str(Path(draft_init) / "config.json"),
        "--train-data-path",
        str(Path(args.data).resolve()),
        "--output-dir",
        str(Path(args.output).resolve()),
        "--cache-dir",
        str(Path(args.cache).resolve()),
        "--num-epochs",
        str(args.epochs or config["epochs"]),
        "--batch-size",
        str(args.microbatch),
        "--accumulation-steps",
        str(config["global_batch_size"] // denom),
        "--learning-rate",
        str(args.learning_rate or config["learning_rate"]),
        "--max-length",
        str(config["max_length"]),
        "--chat-template",
        config["chat_template"],
        "--attention-backend",
        "sdpa",
        "--target-model-backend",
        "hf",
        "--block-size",
        str(config["block_size"]),
        "--num-anchors",
        str(config["num_anchors"]),
        "--embedding-key",
        config["embedding_key"],
        "--lm-head-key",
        config["lm_head_key"],
        "--loss-decay-gamma",
        str(gamma),
        *OBJECTIVES[args.objective],
        "--dflash-kd-topk",
        str(args.kd_topk or config["topk"]),
        "--dflash-kd-lambda-scale",
        str(config["kd_scale"]),
        "--seed",
        str(args.seed),
        "--data-order-seed",
        str(config["data_order_seed"] if args.data_order_seed is None else args.data_order_seed),
        "--dataloader-num-workers",
        "2",
        "--build-dataset-num-proc",
        "2",
        "--save-interval",
        "500",
        "--log-interval",
        "20",
        "--report-to",
        "none",
    ]
    if not args.random_init:
        # The embeddings and the output head always come from the target; this loads the drafter's layers.
        command += ["--draft-init-path", str(draft_init)]
    if args.l2_anchor:
        command += ["--dflash-init-l2", str(args.l2_anchor)]
    if config["is_vlm"]:
        command += [
            "--is-vlm",
            "--min-pixels",
            str(config["min_pixels"]),
            "--max-pixels",
            str(config["max_pixels"]),
        ]
    else:
        command += ["--is-preformatted"]
    if args.resume:
        command += ["--resume"]
    if args.max_steps:
        command += ["--max-train-steps", str(args.max_steps)]
    return command


def main():
    parser = argparse.ArgumentParser(description="Train a DFlash drafter with one of the paper's objectives")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="one response version, as written by build_versions.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), required=True)
    parser.add_argument("--seed", type=int, default=0, help="anchor seed; the paper trains 0 and 1")
    parser.add_argument("--epochs", type=int, help="overrides the config's epoch count")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stops and anneals after this many updates (the 0.25 and 0.5 epoch schedules)")
    parser.add_argument("--gamma", type=float, help="overrides the slot weight gamma (the gamma = 2 control)")
    parser.add_argument("--learning-rate", type=float, help="overrides the learning rate")
    parser.add_argument("--kd-topk", type=int, help="overrides the number of distilled tokens")
    parser.add_argument("--data-order-seed", type=int, help="overrides the config's data order")
    parser.add_argument("--random-init", action="store_true", help="start the drafter's layers from random weights")
    parser.add_argument("--l2-anchor", type=float, default=0.0, help="pull toward the released drafter (lambda)")
    parser.add_argument("--target")
    parser.add_argument("--draft-init")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=6)
    parser.add_argument("--cache", default=".cache/training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    draft_init = args.draft_init or config["draft_init"]
    if not args.dry_run:
        if not Path(args.data).is_file():
            parser.error("Training JSONL does not exist")
        if not Path(draft_init).is_dir():
            from huggingface_hub import snapshot_download

            draft_init = snapshot_download(draft_init)
    command = build_command(args, config, draft_init)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


if __name__ == "__main__":
    main()
