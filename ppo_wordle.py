import sys

from examples.wordle.wordle import generate_random_words
from trlx.data.default_configs import default_ppo_config
sys.path.append("/home/inspur/Documents/trlx-main/")
import trlx
from examples.wordle import wordle
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=128,
        epochs=200,
        total_steps=200000,
        batch_size=64,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(model_path="/data/user_model/wordle/checkpoints/sft_wordle/hf_model/checkpoint_100000/hf_model", num_layers_unfrozen=1),
    tokenizer=TokenizerConfig(tokenizer_path='/data/pretrain_model/llama-huggingface-7b/', truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=3.0e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3.0e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=64,
        ppo_epochs=4,
        init_kl_coef=0.005,
        target=6,
        horizon=10000,
        gamma=0.1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.2,
        scale_reward="ref",
        ref_mean=0,
        ref_std=2,
        cliprange_reward=10,

        gen_kwargs=dict(
            max_new_tokens=2,
            top_k=3,
            top_p=1.0,
            temperature=1.0,
            do_sample=True,
        ),
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    metric_fn, prompts,goal= generate_random_words()#此函数是为了生成一系列prompts
    print(prompts[:4])
    trainer=trlx.train(
        # An "optimality" reward function is used, with scores in [0,1]
        # depending on how close the path is to the shortest possible path.
        reward_fn=lambda samples,prompts, outputs, **kwargs: metric_fn(samples,prompts,outputs)['optimality'],
        # The prompts are simply the first nodes (represented as letters) to
        # start from.
        prompts=prompts,#prompts是生成答案需要的信息，可能是一个词，两个词，三个词，四个词，目标是生成下一个词
        eval_prompts=prompts,
        metric_fn=lambda samples, prompts,outputs, **kwargs: metric_fn(samples,prompts,outputs),
        config=config,
        stop_sequences=["</s>", "</s", "</", "<"],
    )
    #trainer.save_pretrained('/path/to/output/folder/')
#如果当前生成的词长度不对或者不是词汇，需要给一个很差的分数惩罚

if __name__ == "__main__":
    import json
    import sys

    config = default_ppo_config()
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
