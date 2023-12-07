import json
import sys

from datasets import load_dataset


import trlx
from examples.wordle.generate_dataset import generate_dataset, generate_dpo_dataset
from examples.wordle.wordle import generate_random_words
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig, default_ppo_config, default_sft_config,
)
from trlx.trainer.accelerate_dpo_trainer import DPOConfig

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=128,
        epochs=200,
        total_steps=100000,
        batch_size=1,
        checkpoint_interval=10000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateDPOTrainer",
        checkpoint_dir="/data/user_model/wordle/checkpoints/dpo_wordle/hf_model/",
    ),
    model=ModelConfig(model_path="/data/user_model/wordle/checkpoints/sft_wordle/hf_model/checkpoint_02000/hf_model/", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="/data/pretrain_model/llama-huggingface-7b/", truncation_side="right",tokenizer_extra_configs=dict( use_fast=False)),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=100000000, eta_min=1e-6)),
    method=DPOConfig(
        name="dpoconfig",
        gen_kwargs=dict(max_new_tokens=10, top_k=3, top_p=1.0, do_sample=True),
    ),
)




def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    NUMBER_OF_SAMPLE=10000

    metric_fn,prompts,goal = generate_random_words(n_walks=NUMBER_OF_SAMPLE)  # 此函数是为了生成一系列prompts
    dataset = generate_dpo_dataset(prompts,goal)
    #number_of_train=len(prompts)-NUMBER_OF_SAMPLE//100
    #rewards = [20.0 for _ in range(len(dataset["train"]["chosen_sample"]))]
    #print(rewards)
    trlx.train(
        config=config,
        samples=dataset["train"]["prompt_chosen_rejected"],#只用正确答案来训练
        eval_prompts=dataset["train"]["prompt"][:3],
        metric_fn=lambda samples, prompts,outputs, **kwargs: metric_fn(samples,prompts,outputs),
        #stop_sequences=["</s>", "</s", "</", "<"],
        #rewards=[20.0 for _ in range(len(dataset["train"]["chosen_sample"]))]
    )


if __name__ == "__main__":
    import json
    import sys
    config = default_sft_config()
    # micro batch size per gpu
    config.train.batch_size = 1
    # freeze all transformer layers
    config.model.num_layers_unfrozen = 0
    # maximum sample length, prompts or samples longer than that will be truncated
    config.train.seq_length = 128

    # micro batch size for sampling (specific for PPO)
    config.method.chunk_size = 1
    # use an additional Q-head (specific for ILQL)
    config.method.two_qs = False
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
