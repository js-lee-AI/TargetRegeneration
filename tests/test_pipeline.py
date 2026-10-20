"""Pipeline logic that runs without a GPU: training commands, shards and mixture assignments."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.append(str(PIPELINE))

import build_versions  # noqa: E402
import regenerate  # noqa: E402
import train  # noqa: E402

CONFIG = json.loads((PIPELINE / "configs" / "vision_8b.json").read_text())


def command(**overrides):
    args = dict(gpus=4, microbatch=6, random_init=False, l2_anchor=0.0, gamma=None, target=None, data="d.jsonl",
                output="out", cache="cache", epochs=None, learning_rate=None, objective="ce", kd_topk=None, seed=0,
                data_order_seed=None, resume=False, max_steps=0)
    args.update(overrides)
    return train.build_command(Namespace(**args), CONFIG, "init")


def value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_objectives_map_to_the_trainer_flags():
    assert value(command(objective="ce"), "--loss-type") == "dflash"
    kd = command(objective="kd")
    assert value(kd, "--loss-type") == "dflash_targetkd" and value(kd, "--dflash-kd-topk") == "8"
    assert "--dflash-kd-survival-mask" in command(objective="gate")
    assert "--dflash-kd-survival-soft" in command(objective="soft")
    assert value(command(objective="greedy_labels"), "--dflash-kd-hard-label-mix") == "1.0"


def test_the_global_batch_of_24_is_kept():
    cmd = command(gpus=4, microbatch=6)
    assert value(cmd, "--accumulation-steps") == "1"
    assert value(command(gpus=2, microbatch=3), "--accumulation-steps") == "4"
    with pytest.raises(ValueError):
        command(gpus=5, microbatch=1)


def test_controls():
    assert value(command(gamma=2.0), "--loss-decay-gamma") == "2.0"
    assert value(command(), "--loss-decay-gamma") == "7.0"
    assert "--draft-init-path" not in command(random_init=True)
    assert value(command(l2_anchor=1.0), "--dflash-init-l2") == "1.0"
    with pytest.raises(ValueError):
        command(random_init=True, l2_anchor=1.0)
    assert value(command(max_steps=546), "--max-train-steps") == "546"
    assert value(command(data_order_seed=1000), "--data-order-seed") == "1000"


def test_shards_take_every_nth_row(tmp_path):
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(10)))
    shards = [[r["id"] for r in regenerate.read_shard(src, s, 4, 0)] for s in range(4)]
    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]


def test_mixture_assignment_is_nested_in_the_share():
    ids = [f"ex{i}" for i in range(2000)]
    chosen = {share: {i for i in ids if build_versions.share_key(i) < share} for share in (0.25, 0.5, 0.75)}
    assert chosen[0.25] <= chosen[0.5] <= chosen[0.75]
    assert abs(len(chosen[0.5]) / len(ids) - 0.5) < 0.05
