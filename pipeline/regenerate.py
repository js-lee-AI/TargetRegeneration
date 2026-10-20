"""Regenerate the assistant turn of every training example with the target model (vLLM).

The regenerated corpus is the target's own continuation of each prompt, so the prompt has to be the
one the trainer conditions on: the chat template of training (a system turn "You are a helpful
assistant.", then one user turn with the image first and the text second) and the image bounds of
training. Decoding is greedy unless --temperature is set; a sampled corpus uses top-p 1, no top-k and
one fixed seed for every request.

Each process handles one shard (row i goes to shard i mod --num-shards). The paper ran four shards
and concatenated them in shard order 0, 1, 2, 3, so the rows of its regenerated corpora are grouped by
shard, and the training data order applies to that row order:

    python regenerate.py --model Qwen/Qwen3-VL-8B-Instruct --src train/allava_external.jsonl \
        --out train/allava_full.shard0 --shard 0 --num-shards 4 --media-root /path/to/allava/images
    cat train/allava_full.shard0 train/allava_full.shard1 train/allava_full.shard2 \
        train/allava_full.shard3 > train/allava_full.jsonl

A response that reaches --max-new tokens is kept as it is, and the trainer closes the turn there.
Empty responses are kept, so that the row count of every version stays the same.
"""
import argparse
import json
import os


def read_shard(path, shard, num_shards, limit):
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % num_shards == shard and line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def write_rows(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sampling_params(args):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new,
        seed=args.gen_seed if args.temperature > 0 else None,
    )


def regenerate(args):
    from vllm import LLM

    rows = read_shard(args.src, args.shard, args.num_shards, args.limit)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        trust_remote_code=True,
        disable_log_stats=True,
        allowed_local_media_path=args.media_root,
    )
    # The trainer skips rows whose first turn is not the user's, so they are skipped here too.
    convs, keep = [], []
    for row in rows:
        source = row.get("conversations") or []
        if source and source[0].get("role") != "user":
            source = source[1:]
        if not source or source[0].get("role") != "user":
            continue
        convs.append(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "file://" + row["image"]}},
                        {"type": "text", "text": source[0]["content"]},
                    ],
                },
            ]
        )
        keep.append(row)
    outputs = llm.chat(convs, sampling_params(args))
    result, empty = [], 0
    for row, out in zip(keep, outputs):
        text = out.outputs[0].text
        empty += not text.strip()
        rec = dict(row)
        rec["conversations"] = [
            {"role": "user", "content": row["conversations"][0]["content"]},
            {"role": "assistant", "content": text},
        ]
        result.append(rec)
    write_rows(args.out, result)
    print(f"shard {args.shard}: {len(result)} rows ({empty} empty) -> {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the training responses with the target")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--media-root", required=True, help="directory that holds the training images")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gen-seed", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0, help="first N rows of the shard, for a quick check")
    regenerate(parser.parse_args())


if __name__ == "__main__":
    main()
