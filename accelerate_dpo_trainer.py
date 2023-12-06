from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig
import torch.nn.functional as F
from trlx.data.configs import TRLConfig
from trlx.data.method_configs import MethodConfig, register_method
from trlx.pipeline.dpo_pipline import tokenize_dpo_dialogue, DpoStore
from trlx.pipeline.offline_pipeline import PromptPipeline

from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

@dataclass
@register_method
class DPOConfig(MethodConfig):
    """
    Config for SFT training

    :param gen_kwargs: kwargs for generation
    :type gen_kwargs: Dict[str, Any]
    """

    gen_kwargs: dict


@register_trainer
class AccelerateDPOTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )


    def get_arch(self, config):
        from_fn = AutoModelForCausalLM.from_pretrained
        if issubclass(type(config.model.model_path), PretrainedConfig):
            from_fn = AutoModelForCausalLM.from_config

        model = from_fn(config.model.model_path, **config.model.model_extra_configs)

        if config.model.peft_config is not None:
            # Initialize the peft adapter
            import peft

            peft_config = config.model.peft_config
            if not isinstance(peft_config, peft.PeftConfig):
                if isinstance(peft_config, dict):
                    peft_config = peft.get_peft_config(peft_config)
                else:
                    raise ValueError("`peft_config` should be an instance of `peft.PeftConfig` or a dict.")
            model = peft.get_peft_model(model, peft_config)
            if self.accelerator.is_main_process:
                model.print_trainable_parameters()

        return model

    def loss(self, batch):

        loss, metrics = self.get_batch_metrics(self.model, batch)

        stats = {"loss": loss.item()}

        return loss, stats

    def create_train_dataloader(self):
        return self.accelerator.prepare(self.store.create_loader(self.config.train.batch_size))

    def prepare_learning(self):

        self.train_dataloader = self.create_train_dataloader()
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, eval_dataloader)
        if not hasattr(self.model, "frozen_head") and not self.model.peft_type:
            self.ref_model = self.get_arch(self.config)
            self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()
        self.n_inner_epochs = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std
    def make_experience(self, samples, seq_length):

        dialogs = [tokenize_dpo_dialogue(d, self.tokenizer, seq_length) for d in samples]
        self.store = DpoStore(dialogs, self.tokenizer)
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.dual_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.dual_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.dual_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()


        metrics["rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics["rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics["rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics["logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics["logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics["logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics["logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics
    def dual_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:


        chosen_logits = model(input_ids=batch['chosen_input_ids'], attention_mask=batch['chosen_attention_masks'], labels=batch['chosen_labels']).logits.to(torch.float32)
        chosen_logps = self._get_batch_logps(
            chosen_logits,
            batch['chosen_labels'],
            average_log_prob=False,
        )

        rejected_logits = model(input_ids=batch['rejected_input_ids'], attention_mask=batch['rejected_attention_masks'], labels=batch['rejected_labels']).logits.to(torch.float32)
        rejected_logps = self._get_batch_logps(
            rejected_logits,
            batch['chosen_labels'],
            average_log_prob=False,
        )



        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)



    def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [
                    tensor,
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")


        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.tokenizer.pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.tokenizer.pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
