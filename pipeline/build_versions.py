"""Build the response versions of the paper from the external and the regenerated corpus.

    python build_versions.py truncate --external EXT --regenerated FULL --out TRUNC
    python build_versions.py window   --regenerated FULL --truncated TRUNC --out WINDOW
    python build_versions.py closure  --regenerated FULL --truncated TRUNC --window WINDOW --out CLOSURE
    python build_versions.py mix      --external EXT --regenerated FULL --share 0.25 --out MIX

truncate  cuts each regenerated response to the token count of the external response of the same
          example; the trainer then closes the assistant turn after the cut.
window    keeps the full regenerated rows and adds "loss_window": for each assistant span, the number
          of supervised tokens the truncated row has there. The trainer masks the loss after that count.
closure   keeps the truncated rows and masks the loss on the tokens that close a cut turn, unless the
          cut left the response whole or masking would leave fewer than 32 supervised tokens.
mix       replaces the external response of a fixed share of the examples by the regenerated one. The
          assignment is nested in the share, and examples the trainer drops keep their external response.

Token counts follow the trainer's loss-mask rule (specforge/data/preprocessing.py) on the chat template
of training, rendered without the image: the image placeholder sits before the assistant span and
does not change its token count.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_LOSS = 32           # the trainer drops examples with fewer supervised tokens
MIX_SALT = "supp6-e3:"  # fixes the nested mixture assignment of the paper


def read_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"wrote {len(rows)} rows -> {path}")


def by_id(rows):
    out = {r["id"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate example ids")
    return out


def assistant_turns(row):
    return [t for t in row["conversations"] if t["role"] == "assistant"]


class Encoder:
    """Token ids and the trainer's loss mask for a vision row, without the image."""

    def __init__(self, model, template="qwen2-vl"):
        import sys

        sys.path.insert(0, str(ROOT))
        from transformers import AutoTokenizer

        from specforge.data.preprocessing import _apply_loss_mask_from_chat_template
        from specforge.data.template import TEMPLATE_REGISTRY

        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.tpl = TEMPLATE_REGISTRY.get(template)
        self.rule = _apply_loss_mask_from_chat_template

    def count(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def render(self, row):
        source = row["conversations"]
        if source and source[0]["role"] != "user":
            source = source[1:]
        messages = [{"role": "system", "content": self.tpl.system_prompt}]
        for turn in source:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": [
                    {"type": "image", "image": row.get("image", "")},
                    {"type": "text", "text": turn["content"]}]})
            else:
                messages.append({"role": "assistant", "content": turn["content"]})
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def encode(self, row):
        """(ids, spans): token ids and the [start, end) spans of supervised tokens."""
        import torch

        enc = self.tok(self.render(row), return_offsets_mapping=True, add_special_tokens=False)
        decoded = self.tok.decode(enc["input_ids"], skip_special_tokens=False)
        mask = self.rule(decoded, torch.tensor(enc["offset_mapping"]), self.tpl).tolist()
        spans = []
        for i, m in enumerate(mask):
            if not m:
                continue
            if spans and spans[-1][1] == i:
                spans[-1][1] = i + 1
            else:
                spans.append([i, i + 1])
        return enc["input_ids"], spans


def truncate(args):
    enc = Encoder(args.model)
    external = read_rows(args.external)
    regenerated = by_id(read_rows(args.regenerated))
    out, cut = [], 0
    for ext in external:
        row = json.loads(json.dumps(regenerated[ext["id"]]))
        want = [enc.count(t["content"]) for t in assistant_turns(ext)]
        turns = assistant_turns(row)
        if len(turns) != len(want):
            raise ValueError(f"turn count differs on {ext['id']}")
        for turn, n in zip(turns, want):
            ids = enc.tok(turn["content"], add_special_tokens=False)["input_ids"]
            if len(ids) > n:
                turn["content"] = enc.tok.decode(ids[:n], skip_special_tokens=True)
                cut += 1
        out.append(row)
    print(f"cut {cut} responses to the external length")
    write_rows(args.out, out)


def window(args):
    enc = Encoder(args.model)
    regenerated = by_id(read_rows(args.regenerated))
    out = []
    for trunc in read_rows(args.truncated):
        _, spans = enc.encode(trunc)
        row = dict(regenerated[trunc["id"]])
        row["loss_window"] = [end - start for start, end in spans]
        out.append(row)
    write_rows(args.out, out)


def closure(args):
    enc = Encoder(args.model)
    im_end = enc.tok.convert_tokens_to_ids("<|im_end|>")
    newline = enc.tok.encode("\n", add_special_tokens=False)
    regenerated = by_id(read_rows(args.regenerated))
    windows = {r["id"]: r["loss_window"] for r in read_rows(args.window)}
    out, opened = [], 0
    for trunc in read_rows(args.truncated):
        rid = trunc["id"]
        if len(windows[rid]) != 1:
            raise ValueError(f"closure expects one assistant turn, {rid} has {len(windows[rid])}")
        w = windows[rid][0]
        ids, spans = enc.encode(trunc)
        if ids[-2:] != [im_end] + newline:
            raise ValueError(f"{rid} does not end with a closed turn")
        start, end = spans[0]
        closing = max(0, end - (len(ids) - 2))  # closing tokens inside the supervised span
        cut = assistant_turns(trunc)[0]["content"] != assistant_turns(regenerated[rid])[0]["content"]
        keep = w - closing if (cut and closing and w - closing >= MIN_LOSS) else w
        opened += keep < w
        row = dict(trunc)
        row["loss_window"] = [keep]
        out.append(row)
    print(f"masked the closing tokens of {opened} cut responses")
    write_rows(args.out, out)


def share_key(rid):
    return int(hashlib.sha256((MIX_SALT + rid).encode()).hexdigest()[:16], 16) / 2**64


def mix(args):
    if not 0.0 <= args.share <= 1.0:
        raise ValueError("--share must lie in [0, 1]")
    enc = Encoder(args.model)
    external = read_rows(args.external)
    regenerated = by_id(read_rows(args.regenerated))

    def kept(row):
        return sum(end - start for start, end in enc.encode(row)[1]) >= MIN_LOSS

    out, n_reg = [], 0
    for ext in external:
        reg = regenerated[ext["id"]]
        use = share_key(ext["id"]) < args.share and kept(ext) and kept(reg)
        n_reg += use
        out.append(reg if use else ext)
    print(f"regenerated responses in {n_reg} of {len(out)} rows")
    write_rows(args.out, out)


def main():
    parser = argparse.ArgumentParser(description="Build the response versions of the paper")
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, fn, need in (
        ("truncate", truncate, ("external", "regenerated")),
        ("window", window, ("regenerated", "truncated")),
        ("closure", closure, ("regenerated", "truncated", "window")),
        ("mix", mix, ("external", "regenerated")),
    ):
        p = sub.add_parser(name)
        for flag in need:
            p.add_argument("--" + flag, required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="tokenizer of the target")
        if name == "mix":
            p.add_argument("--share", type=float, required=True, help="share of regenerated responses")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
