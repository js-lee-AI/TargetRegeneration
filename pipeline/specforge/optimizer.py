import math

import torch
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800000,
        warmup_ratio=0.015,
        init_l2=0.0,
    ):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.last_grad_norm = None
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )
        # L2 anchor: after each AdamW step, theta <- theta - lr * init_l2 * (theta - theta0), where theta0 are the
        # weights the model holds when the optimizer is built (the released drafter). theta0 is not saved, so a run
        # with init_l2 > 0 cannot be resumed.
        if not (math.isfinite(init_l2) and init_l2 >= 0.0):
            raise ValueError(f"init_l2 must be a finite number >= 0, got {init_l2!r}")
        self.init_l2 = float(init_l2)
        self.init_params = (
            [mp.detach().clone() for mp in self.fp32_params] if self.init_l2 > 0.0 else None
        )

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        if self.init_params is not None:
            coef = self.optimizer.param_groups[0]["lr"] * self.init_l2
            for mp, p0 in zip(self.fp32_params, self.init_params):
                mp.sub_(mp - p0, alpha=coef)
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        if self.init_l2 > 0.0:
            raise RuntimeError("A run with the L2 anchor cannot be resumed: its anchor weights are not checkpointed")
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
