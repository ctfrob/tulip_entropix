import os
from logging import getLogger
from pathlib import Path
from transformers import AutoTokenizer
import torch
from typing import (
  AbstractSet,
  cast,
  Collection,
  Dict,
  Iterator,
  List,
  Literal,
  Optional,
  Sequence,
  Union,
)

logger = getLogger(__name__)

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000

class Tokenizer:
    def __init__(self, model_path: str):
        """Initialize tokenizer from saved model path"""
        self.model = AutoTokenizer.from_pretrained(model_path)
        
        # LLaMA special tokens mapping
        self.special_tokens = {
            'bos_token': '<s>',
            'eos_token': '</s>',
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'im_start': '<|im_start|>',
            'im_end': '<|im_end|>',
            'assistant': '<|assistant|>',
            'user': '<|user|>',
            'python': '<|python_tag|>'
        }
        
        # Add special tokens if not present
        self.model.add_special_tokens({'additional_special_tokens': list(self.special_tokens.values())})
        
        # Set token IDs
        self.bos_id = self.model.bos_token_id
        self.eos_id = self.model.eos_token_id
        self.pad_id = self.model.pad_token_id
        self.eot_id = self.eos_id  # Using EOS as EOT
        self.eom_id = self.model.convert_tokens_to_ids(self.special_tokens['im_end'])
        self.python_tag_id = self.model.convert_tokens_to_ids(self.special_tokens['python'])
        
        # Stop tokens
        self.stop_tokens = [
            self.eom_id,
            self.eot_id,
            self.model.convert_tokens_to_ids(self.special_tokens['im_end']),
            self.model.convert_tokens_to_ids(self.special_tokens['assistant']),
            self.model.convert_tokens_to_ids(self.special_tokens['user'])
        ]
        
        self.n_words = len(self.model.get_vocab())

    def encode(
        self,
        text: str,
        bos: bool = False,
        eos: bool = False,
        **kwargs
    ) -> List[int]:
        """Encode text to token ids"""
        tokens = self.model.encode(
            text,
            add_special_tokens=False,
            **kwargs
        )
        
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
            
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids to text"""
        text = self.model.decode(token_ids, skip_special_tokens=False)
        return text

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Batch encode texts to token ids"""
        batch = self.model(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt"
        )
        return batch.input_ids