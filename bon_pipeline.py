import json
import os
import time
from functools import partial
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, DataCollatorWithPadding
import torch
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BaseRolloutStore
from trlx.pipeline.offline_pipeline import DialogMessage, tokenize_dialogue
from typing import Any, Dict, Iterable, List, Tuple, Union

def bon_collate_fn(max_length,tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], elems: Iterable[Iterable[str]]):

    dialogs =[tokenize_dialogue(elem,tokenizer,max_length) for elem in elems]
    attention_masks = [torch.ones(sum(len(m.tokens) for m in d), dtype=torch.bool) for d in dialogs]
    input_ids = [torch.tensor([t for m in d for t in m.tokens], dtype=torch.long) for d in dialogs]
    # -100 is the ignore index for CrossEntropyLoss
    labels = [
        torch.tensor([t if m.is_output else -100 for m in d for t in m.tokens], dtype=torch.long) for d in dialogs
    ]
    history= [
        dict(input_ids=i, attention_mask=a, labels=l) for i, a, l in zip(input_ids, attention_masks, labels)
    ]
    hf_collate_fn = DataCollatorWithPadding(tokenizer)
    batch = hf_collate_fn(
        {"input_ids": [e["input_ids"] for e in history], "attention_mask": [e["attention_mask"] for e in history]}
    )
    labels = hf_collate_fn([{"input_ids": e["labels"]} for e in history])["input_ids"]
    batch["labels"] = labels
    return batch

class BONRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self,max_length,tokenizer):
        super().__init__()
        self.history: Iterable[Iterable[str]]= [None]
        self.max_length=max_length
        self.tokenizer=tokenizer
    def push(self, exps: Iterable[Iterable[str]]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def __getitem__(self, index: int) -> Iterable[str]:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:

        return DataLoader(
            self, batch_size, shuffle=shuffle, collate_fn=partial(bon_collate_fn, self.max_length, self.tokenizer)
        )
