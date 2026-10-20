import json
import logging
import os
import re
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)
from transformers.utils import cached_file

logger = logging.getLogger(__name__)


def _pre_tokenizer_types(pre_tokenizer):
    types = set()
    if not isinstance(pre_tokenizer, dict):
        return types
    if pre_tokenizer.get("type"):
        types.add(pre_tokenizer["type"])
    for sub in pre_tokenizer.get("pretokenizers") or []:
        types |= _pre_tokenizer_types(sub)
    return types


def load_tokenizer(pretrained_model_name_or_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if not getattr(tokenizer, "is_fast", False):
        return tokenizer
    passthrough = {
        k: kwargs[k]
        for k in ("revision", "token", "cache_dir", "local_files_only")
        if k in kwargs
    }
    try:
        tokenizer_json_path = cached_file(
            pretrained_model_name_or_path,
            "tokenizer.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **passthrough,
        )
    except Exception:
        tokenizer_json_path = None
    if not tokenizer_json_path or not os.path.isfile(tokenizer_json_path):
        return tokenizer
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        saved_pre_tokenizer = json.load(f).get("pre_tokenizer")
    loaded_pre_tokenizer = json.loads(tokenizer.backend_tokenizer.to_str()).get(
        "pre_tokenizer"
    )
    saved_types = _pre_tokenizer_types(saved_pre_tokenizer)
    loaded_types = _pre_tokenizer_types(loaded_pre_tokenizer)
    if "ByteLevel" not in saved_types or "ByteLevel" in loaded_types:
        return tokenizer
    logger.warning(
        "Tokenizer class %s dropped the ByteLevel pre-tokenizer saved in tokenizer.json (loaded %s); reloading with PreTrainedTokenizerFast to preserve the saved tokenization.",
        type(tokenizer).__name__,
        sorted(loaded_types) or "none",
    )
    faithful_kwargs = {k: v for k, v in kwargs.items() if k != "trust_remote_code"}
    return PreTrainedTokenizerFast.from_pretrained(
        pretrained_model_name_or_path, **faithful_kwargs
    )


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()
    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    return PretrainedConfig.from_dict(config)


def get_device_type() -> str:
    dt = os.environ.get("SPECFORGE_DEVICE", None)
    if dt:
        return dt
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def get_local_device() -> torch.device:
    device_type = get_device_type()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        return torch.device("cuda", local_rank)
    if device_type == "npu":
        return torch.device("npu", local_rank)
    return torch.device("cpu")


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_args_with_dots(args):
    if dist.get_rank() == 0:
        args_dict = vars(args)
        max_key_length = max((len(key) for key in args_dict.keys()))
        total_width = 50
        print("\n -----------【args】-----------")
        for key, value in args_dict.items():
            key_str = f"{key:<{max_key_length}}"
            value_str = str(value)
            dot_count = total_width - len(key_str) - len(value_str)
            dot_fill = "·" * dot_count
            print(f"{key_str} {dot_fill} {value_str}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def get_last_checkpoint(folder, prefix="epoch"):
    content = os.listdir(folder)
    _re_checkpoint = re.compile(f"^{re.escape(prefix)}_(\\d+)(?:_step_(\\d+))?$")
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return (None, (0, 0))

    def sort_key(x):
        match = _re_checkpoint.search(x)
        epoch = int(match.group(1))
        step = int(match.group(2)) if match.group(2) else 0
        return (epoch, step)

    last_checkpoint = max(checkpoints, key=sort_key)
    match = _re_checkpoint.search(last_checkpoint)
    epoch = int(match.group(1))
    step = int(match.group(2)) if match.group(2) else 0
    return (os.path.join(folder, last_checkpoint), (epoch, step))


def generate_draft_model_config(
    target_model_path: str, template_config_path: str = None, cache_dir: str = None
):
    target_config = AutoConfig.from_pretrained(target_model_path, cache_dir=cache_dir)
    if template_config_path is None:
        import sys

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)
        template_config_path = os.path.join(
            project_root, "configs", "llama3-8B-eagle3.json"
        )
    with open(template_config_path, "r") as f:
        draft_config = json.load(f)
    if hasattr(target_config, "model_type"):
        draft_config["model_type"] = "llama"
    param_mappings = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rms_norm_eps": "rms_norm_eps",
        "hidden_act": "hidden_act",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "torch_dtype": "torch_dtype",
    }
    for target_param, draft_param in param_mappings.items():
        if hasattr(target_config, target_param):
            value = getattr(target_config, target_param)
            if target_param == "torch_dtype" and isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            draft_config[draft_param] = value
    draft_config["num_hidden_layers"] = 1
    draft_config["tie_word_embeddings"] = False
    draft_config["use_cache"] = True
    if "draft_vocab_size" not in draft_config:
        draft_config["draft_vocab_size"] = 32000
    return draft_config


def save_draft_model_config(config_dict: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Draft model config saved to: {output_path}")


def create_draft_config_from_target(
    target_model_path: str,
    output_dir: str = None,
    template_config_path: str = None,
    cache_dir: str = None,
):
    rank = dist.get_rank()
    if rank == 0:
        print_with_rank(
            "No draft model config provided, auto-generating from target model..."
        )
        config_dict = generate_draft_model_config(
            target_model_path, template_config_path, cache_dir
        )
    dist.barrier()
    if output_dir is None:
        import sys

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "configs")
    model_name = target_model_path.split("/")[-1].lower()
    output_filename = f"{model_name}-eagle3-auto.json"
    output_path = os.path.join(output_dir, output_filename)
    if rank == 0:
        save_draft_model_config(config_dict, output_path)
        print_with_rank(f"Auto-generated draft model config saved to: {output_path}")
    dist.barrier()
    return output_path


def get_full_optimizer_state(optimizer_state_dict: dict):
    full_optimizer_state_dict = {
        k: v for k, v in optimizer_state_dict.items() if k != "state"
    }
    if "state" in optimizer_state_dict:
        full_optimizer_state_dict["state"] = {
            param_id: {
                state_key: state_tensor.full_tensor()
                if isinstance(state_tensor, torch.distributed.tensor.DTensor)
                else state_tensor
                for state_key, state_tensor in param_state.items()
            }
            for param_id, param_state in optimizer_state_dict["state"].items()
        }
    return full_optimizer_state_dict


def shard_optimizer_state_with_dtensor(bf16_optimizer, device_mesh):
    optim = bf16_optimizer.optimizer
    for group in optim.param_groups:
        for p in group["params"]:
            if not isinstance(p, DTensor):
                continue
            state = optim.state.get(p, None)
            if state is None:
                continue
            mesh = device_mesh
            placements = (Shard(dim=0),)
            for k, v in list(state.items()):
                if k == "step":
                    continue
                if isinstance(v, DTensor):
                    continue
                if not isinstance(v, torch.Tensor):
                    continue
                state[k] = distribute_tensor(
                    v.to(p.device), device_mesh=mesh, placements=placements
                )


def safe_conversations_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                raw_convs = row.get("conversations", [])
                if not isinstance(raw_convs, list):
                    if raw_convs is None:
                        raw_convs = []
                    else:
                        logger.warning(
                            f"Line {i + 1}: 'conversations' is not a list. Please check!"
                        )
                        continue
                cleaned_convs = []
                for msg in raw_convs:
                    if not isinstance(msg, dict):
                        continue
                    new_msg = {}
                    for k, v in msg.items():
                        if isinstance(v, (list, dict)):
                            new_msg[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            new_msg[k] = v
                    cleaned_convs.append(new_msg)
                result = {"conversations": cleaned_convs}
                if "image" in row and row["image"] is not None:
                    result["image"] = row["image"]
                if "tools" in row:
                    tools = row["tools"]
                    if tools is not None:
                        if isinstance(tools, str):
                            try:
                                tools = json.loads(tools)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Line {i + 1}: 'tools' is a string but not valid JSON, keeping as-is"
                                )
                                result["tools"] = tools
                                yield result
                                continue
                        if isinstance(tools, (list, dict)):
                            result["tools"] = json.dumps(tools, ensure_ascii=False)
                        else:
                            result["tools"] = tools
                    else:
                        result["tools"] = []
                yield result
            except Exception as e:
                logger.warning(f"Skipping line {i + 1}: {e}")
                continue
