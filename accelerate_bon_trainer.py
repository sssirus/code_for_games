import json
import os
import uuid
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.method_configs import register_method, MethodConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.bon_pipeline import BONRolloutStorage
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)

@dataclass
@register_method
class BONConfig(MethodConfig):
    """
    Config for RFT training

    :param gen_kwargs: kwargs for generation
    :type gen_kwargs: Dict[str, Any]

    :param start_percentile: percentile for the starting score threshold for each prompt used for the first improvement step
    :type start_percentile: float

    :param end_percentile: percentile for the final score threshold for each prompt
    :type end_percentile: float

    :param n_improve_steps: the number of improvement steps for each growth step with linearly increasing score threshold
    :type n_improve_steps: int

    :param n_generations_per_prompt: number of generations to sample per each prompt per each growth step
    :type n_generations_per_prompt: int
    """
    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict


    gen_experience_kwargs: Optional[dict] = None
    num_value_layers_unfrozen: int = 0

@register_trainer
class AccelerateBONTrainer(AccelerateRLTrainer):
    """BON Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """BON Accelerate Trainer initialization

        Args:
            config: `TRLConfig`
            kwargs: Additional keyword arguments passed to `AccelerateRLTrainer`
        """
        super().__init__(config, **kwargs)

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = BONRolloutStorage(self.max_length, self.tokenizer)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Set up a reference model when hydra heads are not used
        # if not hasattr(self.model, "frozen_head") and not self.model.peft_type:
        self.ref_model = self.get_arch(self.config)
        self.ref_model.to(self.accelerator.device)
        self.ref_model.eval()

        # Set up the KL controller
        # This helps prevent large divergences in the controller (policy)
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(config.method.init_kl_coef, config.method.target, config.method.horizon)
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        generate_kwargs = dict(
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            synced_gpus=os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3",
        )
        self.generate_kwargs = {**generate_kwargs, **config.method.gen_kwargs}

        if config.method.gen_experience_kwargs is not None:
            self.generate_experience_kwargs = {**generate_kwargs, **config.method.gen_experience_kwargs}

        else:
            self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper given a model's architecture"""
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

    def loss(self, batch) -> Tuple[float, Dict[str, Any]]:
        batch =batch.to(self.accelerator.device)
        loss = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch["labels"]).loss
        # print('loss:',loss)
        # print('loss.shape:', loss.shape)
        loss=loss+self.kl_penalty.mean()
        stats = {"loss": loss.item()}

        return loss, stats
    #
    def setup_rollout_logging(self, config):
        """Make rollout logging directory to log rollouts to"""
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(config.train.rollout_logging_dir, self.run_id)
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Clears the rollout store and creates `num_rollouts` new samples"""

        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl, n_steps=self.config.train.batch_size)

    def create_train_dataloader(self):
        return self.store.create_loader(self.config.train.batch_size, shuffle=True)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.method.chunk_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)
        self.best_of_n = dict()
        self.make_experience(self.config.method.num_rollouts)

        self.train_dataloader = self.create_train_dataloader()

        self.n_inner_epochs = 1
        self.total_steps = self.config.train.epochs * self.n_inner_epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=True)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """
        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates for all batches & epochs
        """

        #try:
        #logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        bon_rl_elements = []
        accumulated_stats = []

        while len(bon_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            #"""Generate samples for the experience buffer using method's specific `self.generate_experience_kwargs`"""
            # num_return_sequences=n
            samples = self.generate(batch["input_ids"], batch["attention_mask"])
            stats["time/rollout_generate"] = time() - rollout_generate_time
            #print('samples',samples)

            prompt_tensors = batch.input_ids
            final_prompt_tensors=[]

            device = samples.device
            times = self.generate_experience_kwargs['num_return_sequences']
            for prompt_tensor in prompt_tensors:
                for _ in range(times):
                    final_prompt_tensors.append(prompt_tensor.tolist())
            final_prompt_tensors=torch.tensor(final_prompt_tensors)
            prompt_sizes = torch.tensor([final_prompt_tensors.shape[1]] * len(final_prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                final_prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})
            # print('gathered_prompts:',gathered_prompts.shape)
            # print('gathered_samples:', gathered_samples.shape)
            # print('gathered_prompt_sizes:', gathered_prompt_sizes.shape)
            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )
                # print(len(all_str_samples))
                rollout_score_time = time()
                # reward_fn should return list of rewards at each token per sample
                # NOTE: all_scores[0][i] is the reward due to token (action) i in prompt + response (b/c of how kl is computed)
                all_scores = self.reward_fn(
                    samples=all_str_samples,
                    prompts=all_str_prompts,
                    outputs=all_str_outputs,
                    tokenizer=self.tokenizer,
                    **metadata,
                )
                selected_samples =[]
                selected_prompts = []
                selected_outputs=[]
                selected_prompts_tensor=[]
                for i, element in enumerate(all_scores):
                    if all_str_prompts[i] in self.best_of_n.keys():
                        if element > self.best_of_n[all_str_prompts[i]][0]:
                            self.best_of_n[all_str_prompts[i]]=(element,all_str_outputs[i])
                    else:
                        self.best_of_n[all_str_prompts[i]] = (element, all_str_outputs[i])
                pre_prompt = ''
                for i in range(len(all_scores)):
                    current_prompt=all_str_prompts[i]
                    # print("current_prompt:",current_prompt)
                    # print("answer:",self.best_of_n[current_prompt][1])
                    if current_prompt == pre_prompt:
                        continue
                    else:
                        selected_prompts.append(current_prompt)
                        selected_outputs.append(self.best_of_n[current_prompt][1])
                        selected_samples.append(current_prompt+self.best_of_n[current_prompt][1])
                        selected_prompts_tensor.append(final_prompt_tensors[i].tolist())
                        pre_prompt=current_prompt
                        #print(i)


                all_scores = [
                    torch.tensor(score, dtype=torch.float, device=device).view(
                        -1,
                    )
                    for score in all_scores
                ]
                # Pad 0 reward on the ends
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                max_len = torch.tensor(all_scores.shape[1], dtype=torch.long, device=device)

                stats["time/rollout_score"] = time() - rollout_score_time
            #
                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1, max_len).unbind())
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(max_len, 0)
                scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()
            scores_mask = scores != -np.inf
            #
            # str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)
            #
            #
            # Pad the sample outputs
            outputs = self.tokenizer(selected_outputs).input_ids
            for i in range(len(outputs)):
                if self.tokenizer.eos_token_id not in set(outputs[i]):
                    outputs[i] =  outputs[i] +[self.tokenizer.eos_token_id]
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)
            #
            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(
                    dim=1
                ).std()
            all_scores_mean, all_scores_std = self.running_moments.update(torch.sum(scores * scores_mask, dim=1))
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()
            #
            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std
            #
            # Precompute logprobs, values
            selected_prompts_tensor = torch.tensor(selected_prompts_tensor)
            # print('selected_prompts_tensor.shape:',selected_prompts_tensor.shape)
            # print('sample_outputs.shape:', sample_outputs.shape)

            all_tokens = torch.cat((selected_prompts_tensor.to(device), sample_outputs), dim=1)
            selected_prompts_tensor_list = selected_prompts_tensor.tolist()
            labels = all_tokens.tolist()
            for idx,_ in enumerate(selected_prompts_tensor_list):
                labels[idx][:len(selected_prompts_tensor_list[idx])] =[-100] * selected_prompts_tensor.shape[1]

            labels = torch.tensor(labels)
            attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            with torch.no_grad():
                # logits, *_, values = self.model(
                #     all_tokens, attention_mask=attention_mask, position_ids=position_ids
                # )
                logits = self.model(
                    all_tokens,
                    attention_mask=attention_mask,
                    labels=labels
                ).logits
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.model, "frozen_head"): #or self.model.peft_type:
                    ref_logits = self.model.forward_hydra(
                        all_tokens,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    ).logits
                else:
                    # ref_logits = self.ref_model(
                    #     all_tokens,
                    #     attention_mask=attention_mask,
                    #     position_ids=position_ids,
                    #     return_dict=True,
                    # ).logits
                    ref_logits = self.ref_model(
                        all_tokens,
                        attention_mask=attention_mask,
                        labels=labels
                    ).logits
                    ref_logits = ref_logits.to(device)
            #

            # NOTE: logprob[i] is (log)prob at which all_token[i+1] was sampled
            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])
            #
            n_samples: int = len(selected_samples)
            #
            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = selected_prompts_tensor.shape[1] - 1

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl_per_token = kl.mean()
            mean_kl = kl.sum(1).mean()

            # logprobs = logprobs.cpu()
            # ref_logprobs = ref_logprobs.cpu()
            # prompt_tensors = prompt_tensors.cpu()
            # sample_outputs = sample_outputs.cpu()
            # values = values.cpu()[:, :-1]
            #
            # # Get the logprobs and values, for tokens that are not padding,
            # # from the end of the prompt up to the <eos> token, while also including the latter
            # # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1)
            # ends = start + attention_mask[:, start:].sum(1) - 1
            # all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            # all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]
            #
            self.kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            #self.kl_penalty = torch.tensor([xs[start : ends[ix]].tolist() for ix, xs in enumerate(kl_penalty)])

            rollout_count = 0

            for sample_idx in range(n_samples):
                #rewards = kl_penalty[sample_idx]
                # # Then add in rewards
                # if scores.shape[1] == 1:
                #     # NOTE: Final reward given at EOS token following HHH practice
                #     rewards[-1] += scores[sample_idx][0].cpu()
                # else:
                #     score = scores[sample_idx]
                #     score_right_padding = torch.sum(scores_mask[sample_idx])
                #     score = score[:score_right_padding].cpu()
                #     p_score = torch.zeros_like(rewards)
                #     p_score[: score.shape[0]] += score
                #     rewards += p_score

                bon_rl_elements.append(
                    [selected_prompts[sample_idx],selected_outputs[sample_idx]]
                )

                rollout_count += 1

            # if torch.distributed.is_initialized():
            #     torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)
        #
            tbar.set_description(f"[rollout {len(bon_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()
        #
        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(bon_rl_elements)

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        pass
        """
        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")

        self.accelerator.wait_for_everyone()

        # Save only the base model, so that is could be loaded directly
        # with Hugging Face's `from_pretrained` method
        state_dict = self.accelerator.get_state_dict(self.model, unwrap=True)

        self.accelerator.unwrap_model(self.model).save_pretrained(
            directory,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
            **kwargs,
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)
