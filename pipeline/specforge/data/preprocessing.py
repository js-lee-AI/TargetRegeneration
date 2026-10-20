import gzip
import io
import json
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ImageProcessingMixin, PreTrainedTokenizer
from datasets import Dataset as HFDataset
from ..distributed import get_draft_sp_group, get_sp_ring_group

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None
from .parse import GeneralParser, ThinkingParser
from .template import TEMPLATE_REGISTRY, ChatTemplate

Conversation = List[Dict[str, str]]


def _apply_loss_mask_from_chat_template(
    text: str, offsets: torch.Tensor, chat_template: ChatTemplate
) -> torch.Tensor:
    loss_mask = torch.zeros(len(offsets), dtype=torch.long)
    user_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.user_header}"
    )
    assistant_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
    )
    assistant_pattern = (
        re.escape(assistant_message_separator)
        + "(.*?)(?="
        + re.escape(user_message_separator)
        + "|$)"
    )
    matches_found = 0
    for match in re.finditer(assistant_pattern, text, re.DOTALL):
        matches_found += 1
        assistant_start_char = match.start(1)
        assistant_end_char = match.end(1)
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_end <= assistant_start_char:
                continue
            if token_start > assistant_end_char:
                continue
            loss_mask[idx] = 1
    if matches_found == 0:
        print("WARNING: No assistant response spans found in the conversation text.")
    return loss_mask


def _apply_loss_window(
    loss_mask: torch.Tensor, window: Optional[List[int]]
) -> torch.Tensor:
    if window is None:
        return loss_mask
    trainable = torch.nonzero(loss_mask, as_tuple=False).flatten().tolist()
    spans: List[List[int]] = []
    for pos in trainable:
        if spans and pos == spans[-1][1]:
            spans[-1][1] = pos + 1
        else:
            spans.append([pos, pos + 1])
    if len(spans) != len(window):
        raise ValueError(
            f"loss_window has {len(window)} entries but the sample has {len(spans)} assistant spans"
        )
    out = loss_mask.clone()
    for (start, end), keep in zip(spans, window):
        if keep < 0:
            raise ValueError(f"loss_window entries must be >= 0, got {keep}")
        out[start + min(int(keep), end - start) : end] = 0
    return out


def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Conversation], List[str]],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    is_preformatted: bool = False,
    train_only_last_turn: bool = False,
    tools: Optional[List[List[Dict]]] = [[]],
    **kwargs,
) -> Dict[str, List[torch.Tensor]]:
    results = {"input_ids": [], "loss_mask": [], "attention_mask": []}
    if chat_template.parser_type == "general":
        parser = GeneralParser(tokenizer, chat_template)
    elif chat_template.parser_type == "thinking":
        parser = ThinkingParser(tokenizer, chat_template)
    else:
        raise ValueError(f"Invalid parser type: {chat_template.parser_type}")
    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for source, tool, kwargs_item in zip(conversations, tools, kwargs_list):
        if not source:
            continue
        input_ids, loss_mask = parser.parse(
            source,
            max_length,
            preformatted=is_preformatted,
            train_only_last_turn=train_only_last_turn,
            tool=tool,
            **kwargs_item,
        )
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def preprocess_vlm_conversations(
    processor: ImageProcessingMixin,
    examples: List[Conversation],
    chat_template: ChatTemplate,
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    system_prompt = chat_template.system_prompt
    results = {
        "input_ids": [],
        "loss_mask": [],
        "attention_mask": [],
        "pixel_values": [],
        "image_grid_thw": [],
    }
    for i, image in enumerate(examples["image"]):
        source = examples["conversations"][i]
        messages = [{"role": "system", "content": system_prompt}]
        if not source:
            continue
        if source[0]["role"] != "user":
            source = source[1:]
        convroles = ["user", "assistant"]
        for j, sentence in enumerate(source):
            role = sentence["role"]
            assert role == convroles[j % 2], f"unexpected role {role}"
            if role == "user":
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": sentence["content"]},
                        ],
                    }
                )
            else:
                messages.append({"role": role, "content": sentence["content"]})
        conversation = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if not HAS_QWEN_VL_UTILS:
            raise ImportError(
                "qwen_vl_utils is required for VLM preprocessing but is not installed. Please install it to use VLM features."
            )
        image_inputs, video_inputs = process_vision_info(messages)
        assert image_inputs is not None, "image_inputs must not be None"
        encoding = processor(
            text=[conversation],
            images=image_inputs,
            videos=video_inputs,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        pixel_values = encoding.pixel_values
        image_grid_thw = encoding.image_grid_thw[0]
        decoded_conversation = processor.tokenizer.decode(
            encoding.input_ids[0], skip_special_tokens=False
        )
        loss_mask = _apply_loss_mask_from_chat_template(
            decoded_conversation, offsets, chat_template
        )
        if "loss_window" in examples:
            loss_mask = _apply_loss_window(loss_mask, examples["loss_window"][i])
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        results["pixel_values"].append(pixel_values)
        results["image_grid_thw"].append(image_grid_thw[None, :])
    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: Optional[str] = None,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    is_vlm: Optional[bool] = False,
    processor: Optional[ImageProcessingMixin] = None,
    is_preformatted: Optional[bool] = False,
    train_only_last_turn: Optional[bool] = False,
    minimum_valid_tokens: Optional[int] = None,
) -> HFDataset:
    if minimum_valid_tokens is not None and minimum_valid_tokens < 0:
        raise ValueError("minimum_valid_tokens must be >= 0")
    if is_vlm:
        assert processor is not None, "processor must be provided when is_vlm is True"
    if chat_template is None:
        raise ValueError("chat_template must be provided for all dataset types")
    assert chat_template in TEMPLATE_REGISTRY.get_all_template_names(), (
        f"Chat template {chat_template} not found in TEMPLATE_REGISTRY, you may need to register it first"
    )
    template: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template)
    dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    def preprocess_function(examples):
        if is_vlm:
            processed = preprocess_vlm_conversations(
                processor, examples, template, max_length
            )
        elif is_preformatted:
            if "text" not in examples:
                raise ValueError(
                    f"Expected 'text' column for is_preformatted=True, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer,
                examples["text"],
                template,
                max_length,
                is_preformatted=True,
                train_only_last_turn=train_only_last_turn,
                tools=[[] for _ in examples["text"]],
            )
        else:
            if "conversations" not in examples:
                raise ValueError(
                    f"Expected 'conversations' column for is_preformatted=False, but found columns: {list(examples.keys())}"
                )
            conversations = examples.pop("conversations")
            if "id" in examples:
                examples.pop("id")
            if "tools" in examples:
                tools_raw = examples.pop("tools")
                tools = []
                for tool_item in tools_raw:
                    if isinstance(tool_item, (str, list)):
                        try:
                            tools.append(json.loads(tool_item))
                        except json.JSONDecodeError:
                            warnings.warn(
                                f"Failed to parse tools JSON string: {tool_item[:100]}..."
                            )
                            tools.append([])
                    elif isinstance(tool_item, list):
                        tools.append(tool_item)
                    elif tool_item is None:
                        tools.append([])
                    else:
                        warnings.warn(
                            f"Unexpected tools type: {type(tool_item)}, using empty list"
                        )
                        tools.append([])
            else:
                tools = [[] for _ in range(len(conversations))]
            processed = preprocess_conversations(
                tokenizer,
                conversations,
                template,
                max_length,
                is_preformatted=False,
                train_only_last_turn=train_only_last_turn,
                tools=tools,
                **examples,
            )
        return processed

    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}.pkl")
        print(f"dataset is cached at {cache_file_name}")
    elif cache_dir is None and cache_key is None:
        load_from_cache_file = False
        cache_file_name = None
        print(f"dataset is not cached")
    else:
        warnings.warn(
            f"cache_dir and cache_key must be provided together to make caching work"
        )
    if num_proc is not None and num_proc > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if is_vlm:
        batch_size = 200
    else:
        batch_size = 1000
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=original_cols,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )
    if minimum_valid_tokens is not None:
        before_filter = len(dataset)

        def has_minimum_valid_tokens(example):
            loss_mask = example["loss_mask"]
            if isinstance(loss_mask, torch.Tensor):
                valid_tokens = int(loss_mask.sum().item())
            else:
                valid_tokens = sum(
                    (
                        int(token)
                        for row in loss_mask
                        for token in (row if isinstance(row, list) else [row])
                    )
                )
            return valid_tokens >= minimum_valid_tokens

        dataset = dataset.filter(
            has_minimum_valid_tokens,
            num_proc=num_proc,
            desc=f"Filtering samples with >= {minimum_valid_tokens} trainable tokens",
        )
        print(
            f"Filtered dataset by trainable tokens: {before_filter} -> {len(dataset)}"
        )
    dataset.set_format(type="torch")
    return dataset


def list_local_files(path, suffixes=None):
    if suffixes is None:
        suffixes = [".ckpt", ".ckpt.gz"]
    datapaths = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapaths.append(file_path)
    if suffixes:
        datapaths = [
            f_name
            for f_name in datapaths
            if any((f_name.endswith(suffix) for suffix in suffixes))
        ]
    datapaths.sort()
    return datapaths


class OfflineEagle3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapath,
        transform=None,
        max_len=2048,
        ttt_length=1,
        use_usp_preprocess=False,
    ):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len
        self.ttt_length = ttt_length
        self.use_usp_preprocess = use_usp_preprocess
        if use_usp_preprocess:
            sp_group = get_draft_sp_group()
            self.sp_rank = torch.distributed.get_rank(sp_group)
            self.sp_size = torch.distributed.get_world_size(sp_group)
            ring_group = get_sp_ring_group()
            self.ring_rank = torch.distributed.get_rank(ring_group)
            self.sp_ring_size = torch.distributed.get_world_size(ring_group)

    @staticmethod
    def process_data(data, max_len, transform=None):
        new_data = {}
        hidden_state = data["aux_hidden_state"].squeeze(0)[:max_len][None, :]
        target = data["hidden_state"].squeeze(0)[:max_len][None, :]
        input_ids = data["input_ids"][:max_len][None, :]
        loss_mask = data["loss_mask"][:max_len][None, :]
        loss_mask[0, -1] = 0
        new_data["attention_mask"] = torch.ones_like(loss_mask, dtype=torch.long)
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state"] = hidden_state
        new_data["input_ids"] = input_ids
        if transform:
            new_data = transform(new_data)
        return new_data

    @staticmethod
    def process_data_usp(
        data,
        max_len,
        ttt_length=1,
        transform=None,
        sp_rank=0,
        sp_size=1,
        ring_rank=0,
        sp_ring_size=1,
    ):
        new_data = {}
        input_ids = data["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        global_len = min(max_len, input_ids.shape[1])
        chunk_size = (global_len + sp_size - 1) // sp_size
        start = sp_rank * chunk_size
        local_len = chunk_size + ttt_length
        end = min(start + local_len, global_len)

        def _slice_and_pad(tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            tensor = tensor[:, :global_len]
            sliced = tensor[:, start : min(end, tensor.shape[1])]
            valid_len = sliced.shape[1]
            if valid_len < local_len:
                pad_len = local_len - valid_len
                if tensor.ndim == 2:
                    sliced = F.pad(sliced, (0, pad_len))
                else:
                    sliced = F.pad(sliced, (0, 0, 0, pad_len))
            return (sliced.contiguous(), valid_len)

        if "aux_hidden_state" not in data or data["aux_hidden_state"] is None:
            raise KeyError("aux_hidden_state is required for OfflineEagle3Dataset")
        new_data["hidden_state"], _ = _slice_and_pad(data["aux_hidden_state"])
        new_data["target"], _ = _slice_and_pad(data["hidden_state"])
        new_data["input_ids"], valid_len = _slice_and_pad(input_ids)
        full_loss_mask = data["loss_mask"]
        if full_loss_mask.ndim == 1:
            full_loss_mask = full_loss_mask.unsqueeze(0)
        full_loss_mask = full_loss_mask[:, :global_len].clone()
        if full_loss_mask.numel() > 0:
            full_loss_mask[0, -1] = 0
        new_data["loss_mask"], _ = _slice_and_pad(full_loss_mask)
        local_len = new_data["input_ids"].shape[1]
        attention_mask = torch.zeros((1, local_len), dtype=torch.long)
        attention_mask[:, :valid_len] = 1
        new_data["attention_mask"] = attention_mask
        sp_ulysses_size = max(1, sp_size // sp_ring_size)
        usp_chunk_size = max(local_len - ttt_length, 0)
        ring_chunk = usp_chunk_size * sp_ulysses_size
        ulysses_rank = sp_rank % sp_ulysses_size
        ring_start = ring_rank * ring_chunk + ulysses_rank * usp_chunk_size
        new_data["position_ids"] = torch.arange(
            ring_start, ring_start + usp_chunk_size, dtype=torch.long
        ).unsqueeze(0)
        if transform:
            new_data = transform(new_data)
        return new_data

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        data_path = self.datapaths[index]
        if data_path.endswith(".gz"):
            with gzip.open(data_path, "rb") as f:
                return torch.load(io.BytesIO(f.read()), weights_only=False)
        return torch.load(data_path, weights_only=False, mmap=True)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"ERROR Failed to load {self.datapaths[index]} with error {e}")
            data = self._open_file(0)
        if self.use_usp_preprocess:
            return self.process_data_usp(
                data,
                self.max_len,
                ttt_length=self.ttt_length,
                transform=self.transform,
                sp_rank=self.sp_rank,
                sp_size=self.sp_size,
                ring_rank=self.ring_rank,
                sp_ring_size=self.sp_ring_size,
            )
        return self.process_data(data, self.max_len, self.transform)

    def set_epoch(self, epoch):
        self._epoch = epoch


def build_offline_eagle3_dataset(
    hidden_states_path: str,
    max_len: int = 2048,
    ttt_length: int = 1,
    use_usp_preprocess: bool = False,
) -> torch.utils.data.Dataset:
    return OfflineEagle3Dataset(
        list_local_files(hidden_states_path),
        max_len=max_len,
        ttt_length=ttt_length,
        use_usp_preprocess=use_usp_preprocess,
    )


def generate_vocab_mapping_file(
    dataset: HFDataset,
    target_vocab_size: int,
    draft_vocab_size: int,
    cache_dir: str = "./cache/vocab_mapping",
    cache_key: str = "vocab_mapping",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pt")
    if os.path.exists(vocab_mapping_path):
        print(f"Loading vocab mapping from the cached file at: {vocab_mapping_path}")
        return vocab_mapping_path
    token_dict = Counter()
    for input_ids, loss_mask in tqdm(
        zip(dataset["input_ids"], dataset["loss_mask"]),
        total=len(dataset),
        desc="Counting tokens for vocab mapping",
    ):
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
        token_dict.update(batch_token_dict)
    d2t, t2d = process_token_dict_to_mappings(
        token_dict, draft_vocab_size, target_vocab_size
    )
    vocab_mapping = {"d2t": d2t, "t2d": t2d}
    torch.save(vocab_mapping, vocab_mapping_path)
    print(f"Saved vocab mapping to: {vocab_mapping_path}")
    return vocab_mapping_path


def process_token_dict_to_mappings(
    token_dict: Counter, draft_vocab_size: int, target_vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
    print(f"Added missing tokens to reach draft vocab size: {draft_vocab_size}")
    print(f"Total tokens after addition: {len(token_dict)}")
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum((freq for key, freq in top_N))
    if total_frequency == 0:
        print(
            "Warning: Total token frequency is zero. All tokens will have zero ratio."
        )
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency
    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)
    return (d2t, t2d)
