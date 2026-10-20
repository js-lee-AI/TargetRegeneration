import argparse
import json
from pathlib import Path

from core.metrics import summarize

# The chain verifier is the tree routine restricted to a single path of seven tokens.
VERIFIERS = {"tree": {"budget": 63, "branch_k": 8}, "chain": {"budget": 7, "branch_k": 1}}


def load_prompts(path, count, image_size):
    from PIL import Image

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()][
        :count
    ]
    if len(rows) != count:
        raise ValueError(f"Expected {count} prompts in {path.name}, found {len(rows)}")
    result = []
    for i, row in enumerate(rows):
        image = Image.open(path.parent / row["image"]).convert("RGB")
        if max(image.size) > image_size:
            w, h = image.size
            scale = image_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)))
        result.append((str(row.get("id", i)), row["prompt"], image))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language tree drafting"
    )
    parser.add_argument("--target", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", default="caption,textvqa,docvqa")
    parser.add_argument("--temperatures", default="0,1")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--verifier",
        choices=sorted(VERIFIERS),
        default="tree",
        help="tree: 63 draft tokens, 8 branches; chain: one path of 7 tokens",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or args.n_prompts < 2:
        parser.error("Use a new output path and at least two prompts")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from dflash.model import DFlashDraftModel
    from core.features import _LayerTap
    from core.vision_decode import Noise, ar_generate, prompt_seed, tree_generate
    from specforge.modeling.target.qwen3vl_vision_fastpath import (
        apply_qwen3vl_vision_fastpath,
    )

    target = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.target, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    apply_qwen3vl_vision_fastpath(target)
    draft = (
        DFlashDraftModel.from_pretrained(
            args.draft, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.target)
    tap = _LayerTap(target, draft.target_layer_ids)
    result = {"config": vars(args), "cells": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    for temperature in map(float, args.temperatures.split(",")):
        for domain in args.domains.split(","):
            pool = load_prompts(
                Path(args.data_dir) / (domain + ".jsonl"),
                args.n_prompts,
                args.image_size,
            )
            rows = []
            for i, (identifier, prompt, image) in enumerate(pool):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
                ids = inputs["input_ids"]
                noise = Noise(
                    temperature,
                    prompt_seed(args.sample_seed, domain, i),
                    args.max_new + 2 * draft.block_size + 4,
                    target.lm_head.out_features,
                    ids.device,
                )
                shared = (
                    ids,
                    inputs.get("pixel_values"),
                    inputs.get("image_grid_thw"),
                    args.max_new,
                    [processor.tokenizer.eos_token_id],
                    noise,
                )
                _, ar = ar_generate(target, *shared, return_stats=True)
                _, st = tree_generate(
                    draft,
                    target,
                    tap,
                    *shared,
                    **VERIFIERS[args.verifier],
                    return_stats=True,
                )
                if i == 0:  # The first vision prompt is the warm-up.
                    continue
                rows.append(
                    {
                        "id": identifier,
                        "ar": {"decode_ms_token": ar["ms_per_token"]},
                        "draft": {
                            "tau": st["tau_mean"],
                            "rounds": st["n_rounds"],
                            "decode_ms_token": st["ms_per_token"],
                            "tokens": st["n_out"],
                        },
                    }
                )
            result["cells"][f"{domain}/T{temperature:g}"] = {
                "summary": summarize(rows),
                "rows": rows,
            }
            output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
