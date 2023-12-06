from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch

from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@dataclass
class DpoMessage:
    """
    Single message in dpo

    :param type: 0:prompt 1:chosen_output 2:rejected_output
    :type type: int

    :param tokens: Tokenized message
    :type tokens: Tuple[int]
    """

    type: int
    tokens: Tuple[int]


def tokenize_dpo_dialogue(  # noqa: C901
    dialogue: Union[str, Iterable[str]], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_length=2048
) -> List[DpoMessage]:
    """
    Tokenize sample with the interleaved form of (prompt_1, chosen_1, reject_1, prompt_2, chosen_2, reject_3...)
    """
    if isinstance(dialogue, str):
        bos_token = tokenizer.bos_token or tokenizer.eos_token
        dialogue = [bos_token, dialogue]
    elif isinstance(dialogue, Iterable):
        if len(dialogue) % 3 != 0:
            raise ValueError("Dpo must have an 3n number of phrases, alternating prompt and output")
        dialogue = list(dialogue)
    # add eos on the end of each sequences
    if not dialogue[-1].endswith(tokenizer.eos_token):
        dialogue[-1] = dialogue[-1] + tokenizer.eos_token

    tokenized = [
        DpoMessage(type=i % 3, tokens=tuple(tokenizer(dialogue[i], add_special_tokens=False).input_ids))
        for i in range(len(dialogue))
    ]

    # flip to truncate from the left
    if tokenizer.truncation_side == "left":
        tokenized = [DpoMessage(type=m.type, tokens=m.tokens[::-1]) for m in tokenized[::-1]]

    # truncate if necessary
    lengths = [len(t.tokens) for t in tokenized]
    cumsum_lengths = [sum(lengths[:i]) for i in range(len(lengths))]
    truncated = [
        DpoMessage(type=t.type, tokens=t.tokens[: max(max_length - cl, 0)])
        for t, cl in zip(tokenized, cumsum_lengths)
    ]

    # flip back if was fliped to left truncate
    if tokenizer.truncation_side == "left":
        truncated = [DpoMessage(type=m.type, tokens=m.tokens[::-1]) for m in truncated[::-1]]

    # remove empty messages
    out = [t for t in truncated if len(t.tokens) > 0]

    if out[0].type>0:
        if sum(map(lambda msg: len(msg.tokens), out)) == max_length:
            if tokenizer.truncation_side == "left":
                out[0].tokens = out[0].tokens[1:]
            else:
                out[-1].tokens = out[-1].tokens[:-1]

        out.insert(0, DpoMessage(0, (tokenizer.bos_token_id,)))
    return out
class DpoStore(BaseRolloutStore):
    def __init__(self, dialogs: List[List[DpoMessage]], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # -100 is the ignore index for CrossEntropyLoss

        chosen_attention_masks=[]
        chosen_input_ids=[]
        chosen_labels=[]
        rejected_attention_masks=[]
        rejected_input_ids=[]
        rejected_labels=[]

        for d in dialogs:
            prompt_chosen_attention_mask=torch.ones(sum(len(m.tokens) for m in [d[0],d[1]]), dtype=torch.bool)
            prompt_chosen_input_id = torch.tensor([t for m in [d[0],d[1]] for t in m.tokens], dtype=torch.long)
            prompt_chosen_label =torch.tensor([t if m.type >0 else -100 for m in [d[0],d[1]] for t in m.tokens], dtype=torch.long)

            prompt_rejected_attention_mask=torch.ones(sum(len(m.tokens) for m in [d[0],d[1]]), dtype=torch.bool)
            prompt_rejected_input_id = torch.tensor([t for m in [d[0],d[1]] for t in m.tokens], dtype=torch.long)
            prompt_rejected_label =torch.tensor([t if m.type >0 else -100 for m in [d[0],d[1]] for t in m.tokens], dtype=torch.long)
            chosen_attention_masks.append(prompt_chosen_attention_mask)
            chosen_input_ids.append(prompt_chosen_input_id)
            chosen_labels.append(prompt_chosen_label)
            rejected_attention_masks.append(prompt_rejected_attention_mask)
            rejected_input_ids.append(prompt_rejected_input_id)
            rejected_labels.append(prompt_rejected_label)

        self.history = [
            dict(chosen_input_ids=a, chosen_attention_masks=b, chosen_labels=c,rejected_input_ids=d, rejected_attention_masks=e, rejected_labels=f) for a, b, c,d,e,f in zip(chosen_input_ids, chosen_attention_masks, chosen_labels,rejected_input_ids,rejected_attention_masks,rejected_labels)
        ]

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

        def collate_fn(elems: Iterable[dict]):
            chosen = hf_collate_fn(
                {"input_ids": [e["chosen_input_ids"] for e in elems], "attention_mask": [e["chosen_attention_masks"] for e in elems]}
            )
            chosen_labels = hf_collate_fn([{"input_ids": e["chosen_labels"]} for e in elems])["input_ids"]
            rejected = hf_collate_fn(
                {"input_ids": [e["rejected_input_ids"] for e in elems], "attention_mask": [e["rejected_attention_masks"] for e in elems]}
            )
            rejected_labels = hf_collate_fn([{"input_ids": e["rejected_labels"]} for e in elems])["input_ids"]

            batch=dict()
            batch["chosen_input_ids"]=chosen["input_ids"]
            batch["chosen_attention_masks"] = chosen["attention_mask"]
            batch["chosen_labels"]=chosen_labels
            batch["rejected_input_ids"]=rejected["input_ids"]
            batch["rejected_attention_masks"] = rejected["attention_mask"]
            batch["rejected_labels"]=rejected_labels
            return batch

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
