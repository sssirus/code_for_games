from examples.wordle.wordle import generate_random_words, generate_rand_choice_excluding

import random
import string
def generate_random_string(length):
    letters = string.ascii_letters.lower()  # 包含小写字母
    return ''.join([random.choice(letters) for _ in range(length)])
def generate_dataset(prompts,goal):
    #reward_fuction, prompts,samples_goal= generate_random_words(n_walks=length_of_train_data)#此函数是为了生成一系列prompts
    #_, test_prompts, test_samples_goal = generate_random_words(n_walks=len(prompts)/)  # 此函数是为了生成一系列prompt
    evel_num=int(len(prompts)//100)
    train_num=int(len(prompts)-evel_num)
    return  {
        'train':{
            'prompt':prompts[:train_num],
            'chosen':goal[:train_num],
            'chosen_sample':[[prompts[i],goal[i]] for i in range(len(prompts[:train_num]))]
        },
        'test': {
            'prompt': prompts[train_num:],
            'chosen': goal[train_num:]
        }
    }
def generate_dpo_dataset(prompts,goal):
    file = open('/home/inspur/Documents/trlx-main/examples/wordle/data/wordle_words.txt', 'r')
    candidates=[]
    try:
        while True:
            text_line = file.readline()
            if text_line:
                candidates.append(text_line.strip('\n'))
            else:
                break
    finally:
        file.close()

    res=  {
        'train':{
            'prompt':prompts,
            'chosen':goal,
            'prompt_chosen_rejected':[[prompts[i],goal[i],generate_rand_choice_excluding(goal[i], candidates)+generate_random_string(random.randint(0, 4))] for i in range(len(prompts))]
        }
    }
    for i in range(len(prompts)):
        res['train']['prompt'].append(prompts[i])
        res['train']['chosen'].append(goal[i])
        res['train']['prompt_chosen_rejected'].append([prompts[i],goal[i],generate_rand_choice_excluding(goal[i], candidates)[:random.randint(0, 4)]])
    return res
    # file = open('data/wordle_words.txt', 'r')
    # candidates=[]
    # try:
    #     while True:
    #         text_line = file.readline()
    #         if text_line:
    #             candidates.append(text_line)
    #         else:
    #             break
    # finally:
    #     file.close()
    # sample=candidates[:5000]
    # print(sample)
    # score = metric_fn(prompts,sample)
    # for i,item in enumerate(prompts):
    #     print(sample[i])
    #     print(score[i])