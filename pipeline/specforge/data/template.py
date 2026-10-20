from typing import List, Optional
from pydantic import BaseModel


class ChatTemplate(BaseModel):
    assistant_header: Optional[str] = None
    user_header: Optional[str] = None
    system_prompt: Optional[str] = None
    end_of_turn_token: Optional[str] = None
    parser_type: str = "general"
    assistant_pattern_type: str = "general"
    enable_thinking: bool = False
    ignore_token: Optional[List[str]] = None


class TemplateRegistry:
    def __init__(self):
        self.templates = {}

    def register(self, name: str, template: ChatTemplate, override: bool = False):
        assert override or name not in self.templates, (
            f"Chat template for the model type {name} has already been registered"
        )
        self.templates[name] = template

    def get(self, name: str) -> ChatTemplate:
        return self.templates[name]

    def get_all_template_names(self) -> List[str]:
        return list(self.templates.keys())


TEMPLATE_REGISTRY = TemplateRegistry()
TEMPLATE_REGISTRY.register(
    name="llama3",
    template=ChatTemplate(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_header="<|start_header_id|>user<|end_header_id|>",
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        end_of_turn_token="<|eot_id|>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="llama4",
    template=ChatTemplate(
        assistant_header="<|header_start|>assistant<|header_end|>\n\n",
        user_header="<|header_start|>user<|header_end|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|eot|>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="qwen",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)
TEMPLATE_REGISTRY.register(
    name="lfm",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="",
        end_of_turn_token="<|im_end|>\n",
    ),
)
TEMPLATE_REGISTRY.register(
    name="qwen2-vl",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)
TEMPLATE_REGISTRY.register(
    name="phi3",
    template=ChatTemplate(
        assistant_header="<|assistant|>\n",
        user_header="<|user|>\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>\n",
    ),
)
TEMPLATE_REGISTRY.register(
    name="phi4",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant<|im_sep|>",
        user_header="<|im_start|>user<|im_sep|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="phi4-mini",
    template=ChatTemplate(
        assistant_header="<|assistant|>",
        user_header="<|user|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="deepseek-r1-distill",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        end_of_turn_token=None,
        system_prompt=None,
    ),
)
TEMPLATE_REGISTRY.register(
    name="qwen3-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
TEMPLATE_REGISTRY.register(
    name="qwen3-instruct",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        ignore_token=["<think>\n\n</think>\n\n"],
    ),
)
TEMPLATE_REGISTRY.register(
    name="qwen3-next-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
TEMPLATE_REGISTRY.register(
    name="kimi-k2-thinking",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
TEMPLATE_REGISTRY.register(
    name="kimi-k2-instruct",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="deepseek-v3",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<｜end▁of▁sentence｜>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="ling-flash-2.0",
    template=ChatTemplate(
        assistant_header="<role>ASSISTANT</role>",
        user_header="<role>HUMAN</role>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|role_end|>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="deepseek-v32",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="",
        end_of_turn_token="<｜end▁of▁sentence｜>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
TEMPLATE_REGISTRY.register(
    name="gemma",
    template=ChatTemplate(
        assistant_header="<start_of_turn>model\n",
        user_header="<start_of_turn>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<end_of_turn>\n",
    ),
)
TEMPLATE_REGISTRY.register(
    name="longcat",
    template=ChatTemplate(
        assistant_header=" ASSISTANT:",
        user_header=" USER:",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="</longcat_s>",
        assistant_pattern_type="longcat",
    ),
)
TEMPLATE_REGISTRY.register(
    name="longcat_xml",
    template=ChatTemplate(
        assistant_header="<longcat_assistant>",
        user_header="<longcat_user>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="</longcat_s>",
    ),
)
TEMPLATE_REGISTRY.register(
    name="qwen3.5",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>\n",
        user_header="<|im_start|>user\n",
        system_prompt="",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
