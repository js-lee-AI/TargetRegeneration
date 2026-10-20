import json
from pathlib import Path
import torch.distributed as dist


class LocalTracker:
    def __init__(self, args, output_dir):
        self.stream = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.stream = (Path(output_dir) / "metrics.jsonl").open("a")

    def log(self, values, step=None):
        if self.stream:
            self.stream.write(json.dumps({"step": step, **values}) + "\n")
            self.stream.flush()

    def close(self):
        if self.stream:
            self.stream.close()


def create_tracker(args, output_dir):
    return LocalTracker(args, output_dir)


def get_tracker_class(name):
    return LocalTracker
