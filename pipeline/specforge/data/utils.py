import os
from typing import Any, Dict, List, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
from specforge.distributed import get_draft_sp_group, get_sp_ulysses_group


class DataCollatorWithPadding:
    def __init__(self):
        self.sp_degree = torch.distributed.get_world_size(get_draft_sp_group())
        self.ulysses_degree = torch.distributed.get_world_size(get_sp_ulysses_group())

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, S, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max((item["input_ids"].shape[1] for item in features))
        max_length = (
            (max_length + self.sp_degree - 1) // self.sp_degree * self.sp_degree
        )
        position_max_len = max_length * self.ulysses_degree
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        if "position_ids" in features[0]:
            batch_position_ids = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], position_max_len)
                    for item in features
                ]
            )
        else:
            batch_position_ids = None
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_state": None,
            "target": None,
        }
        if batch_position_ids is not None:
            batch["position_ids"] = batch_position_ids
        if all(("hidden_state" in item for item in features)):
            assert all(("target" in item for item in features)), (
                "target is required when hidden_state is provided"
            )
            if self.sp_degree > 1:
                batch["hidden_state"] = torch.cat(
                    [item["hidden_state"] for item in features]
                )
            else:
                batch["hidden_state"] = torch.cat(
                    [
                        self.paddingtensor(item["hidden_state"], max_length)
                        for item in features
                    ]
                )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


class VlmDataCollatorWithPadding:
    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max((item["input_ids"].shape[1] for item in features))
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch_pixel_values = torch.cat(
            [item["pixel_values"] for item in features], dim=0
        )
        batch_image_grid_thw = torch.cat(
            [item["image_grid_thw"] for item in features], dim=0
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
            "hidden_state": None,
            "target": None,
        }
        if all(("hidden_state" in item for item in features)):
            assert all(("target" in item for item in features)), (
                "target is required when hidden_state is provided"
            )
            batch["hidden_state"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_state"], max_length)
                    for item in features
                ]
            )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


def prepare_dp_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
    is_vlm: Optional[bool] = False,
    prefetch_factor: Optional[int] = 2,
    sampler_seed: int = 0,
    **dataloader_kwargs,
) -> DataLoader:
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    _full_coverage = os.environ.get("SF_FULL_COVERAGE") == "1"
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False if _full_coverage else shuffle,
        seed=sampler_seed,
    )
    if is_vlm:
        datacollator_cls = VlmDataCollatorWithPadding
    else:
        datacollator_cls = DataCollatorWithPadding
    if num_workers == 0:
        prefetch_factor = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=datacollator_cls(),
        drop_last=not _full_coverage,
        **dataloader_kwargs,
    )
    return dataloader
