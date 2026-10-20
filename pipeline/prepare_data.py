import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare_vision(args):
    selected = read_rows(ROOT / "data" / (args.corpus + "_selection.json"))
    raw = read_rows(args.annotations)
    by_image = {}
    for row in raw:
        by_image.setdefault(Path(row.get("image", "")).name, []).append(row)
    image_root = Path(args.images).resolve()
    images = {}
    for path in image_root.rglob("*"):
        if path.is_file():
            images.setdefault(path.name, []).append(path)
    output = []
    for item in selected:
        if args.corpus == "sharegpt4v":
            # IDs repeat in this release, so retain the original row index.
            source = raw[item["source_index"]]
            digest = hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest()
            if digest != item["source_sha256"]:
                raise ValueError(
                    "ShareGPT4V annotation release differs from the fixed selection"
                )
            relative = source["image"].removeprefix("llava/llava_pretrain/images/")
            matches = [image_root / relative]
            if not matches[0].is_file():
                raise FileNotFoundError(matches[0])
        else:
            candidates = by_image[item["image"]]
            if len(candidates) != 1:
                raise ValueError("Ambiguous annotation image: " + item["image"])
            source = candidates[0]
            matches = images.get(item["image"], [])
        conversation = source["conversations"]
        if len(conversation) != 2:
            raise ValueError("Expected a single-turn image conversation")
        prompt = conversation[0].get("value", conversation[0].get("content", ""))
        response = conversation[1].get("value", conversation[1].get("content", ""))
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous image: " + item["image"])
        output.append(
            {
                "id": item["id"],
                "image": str(matches[0]),
                "conversations": [
                    {"role": "user", "content": prompt.replace("<image>", "").strip()},
                    {"role": "assistant", "content": response.strip()},
                ],
            }
        )
    write_rows(args.output, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the fixed training subsets")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    vision = subparsers.add_parser("vision")
    vision.add_argument("--corpus", choices=["allava", "sharegpt4v"], required=True)
    vision.add_argument("--annotations", required=True)
    vision.add_argument("--images", required=True)
    vision.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_vision(args)
