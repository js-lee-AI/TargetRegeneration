import argparse
import hashlib
import io
import json
from pathlib import Path

SOURCES = {
    "caption": ("lmms-lab/COCO-Caption2017", None, "val", 1024, 90),
    # TextVQA loads the public train split at the revision in data/vision_revisions.json,
    # and every prompt is checked against data/vision_manifest.json.
    "textvqa": ("lmms-lab/textvqa", None, "train", 1024, 90),
    "docvqa": ("lmms-lab/DocVQA", "DocVQA", "validation", 1280, 90),
}
DOC_PREFIX = (
    "Read the document image and answer the question. Quote the exact "
    "text from the document, then briefly explain.\nQuestion: "
)


def encode_image(image, size, quality):
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > size:
        image = image.resize((int(w * size / max(w, h)), int(h * size / max(w, h))))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def prepare(output, domains):
    from datasets import load_dataset

    manifest = json.loads(
        (Path(__file__).parent / "data/vision_manifest.json").read_text()
    )
    output = Path(output)
    revisions = json.loads(
        (Path(__file__).parent / "data/vision_revisions.json").read_text()
    )
    for domain in domains:
        path = output / (domain + ".jsonl")
        if path.exists():
            raise FileExistsError(path)
        repo, subset, split, size, quality = SOURCES[domain]
        expected = {r["source_index"]: r for r in manifest[domain]}
        dataset = load_dataset(
            repo, subset, split=split, streaming=True, revision=revisions[repo]
        )
        rows = []
        for index, source in enumerate(dataset):
            if index > max(expected):
                break
            if index not in expected:
                continue
            record = expected[index]
            prompt = (
                "Describe this image in detail."
                if domain == "caption"
                else source["question"]
                if domain == "textvqa"
                else DOC_PREFIX + source["question"].strip()
            )
            if prompt != record["prompt"]:
                raise ValueError(
                    f"{domain} prompt {index} differs from the fixed evaluation set"
                )
            content = encode_image(source["image"], size, quality)
            digest = hashlib.sha256(content).hexdigest()
            image_path = output / record["image"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(content)
            rows.append(
                {
                    "id": f"{domain}:{index}",
                    "image": record["image"],
                    "prompt": prompt,
                    "image_sha256": digest,
                }
            )
        if len(rows) != len(expected):
            raise ValueError(f"Incomplete {domain} split")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(domain, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the fixed vision evaluation prompts"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", default="caption,textvqa,docvqa")
    args = parser.parse_args()
    prepare(args.output, args.domains.split(","))
