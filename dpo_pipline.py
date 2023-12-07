from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
import torch

from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast, BatchEncoding,
)

from trlx.data.dpo_types import DPORLBatch
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline
import torch.nn.functional as F

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


def tokenize_dpo_dialogue(
    max_length,
    max_prompt_length,
    dialogue: Union[str, Iterable[str]],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
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
    return tokenize_batch_element(max_length,max_prompt_length,tokenizer,dialogue[0],dialogue[1],dialogue[2])
def tokenize_batch_element(
        max_length,
        max_prompt_length,
        tokenizer,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not False:
            chosen_tokens = tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)

            eos_token_id = tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:

                    prompt_tokens = {k: v[-max_prompt_length :] for k, v in prompt_tokens.items()}


            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
                chosen_tokens = {k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens



        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch
class DpoStore(BaseRolloutStore):
    def __init__(self, dialogs: List[dict], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # -100 is the ignore index for CrossEntropyLoss



        self.history = dialogs



    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

        def collate_fn(batch: Iterable[dict]):
            padded_batch = {}
            for k in batch[0].keys():
                if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):

                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = -100
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
                else:
                    padded_batch[k] = [ex[k] for ex in batch]


            return padded_batch

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    def push(self, exps: Iterable[Any]):
        self.history += exps