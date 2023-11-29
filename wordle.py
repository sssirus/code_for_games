import random
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import re
def generate_rand_choice_excluding( exclude,candidates) -> int:
    """Random  generator, excluding a specific number
    """
    while True:
        # Create the random integer
        x = random.choices(candidates)[0]

        #x=x.strip('\n')
        #print(x)
        if  x!= exclude:
            return x

def eval_a_word(goal,guess):

    #print("goal",goal)
    #print("guess",guess)
    contain_but_not_correct_location_idx=[]
    contain_and_correct_location_idx = []
    contained=set()
    for idx,chara in enumerate(guess):
        if chara == goal[idx]:

            contain_and_correct_location_idx.append(idx)
            contained.add(chara)
        elif chara in set(goal) and chara not in set(contain_and_correct_location_idx):
            contain_but_not_correct_location_idx.append(idx)
            contained.add(chara)
    return contain_and_correct_location_idx,contain_but_not_correct_location_idx,contained


def generate_random_words(  # noqa: max-complexity
    n_walks: int = 1000,
    gpt2_tokenizer: bool = False,
) :
    """Generate random walks
        Args:
            n_walks: Number of random walks (samples) to create
            gpt2_tokenizer: True if GPT2's tokenizer is being used
        Returns:
            Tuple of metric function,
        """

    file = open('data/wordle_words.txt', 'r')
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
    candidates_set=set(candidates)
    #生成一系列目标词汇
    samples_goal=random.choices(candidates,k=n_walks)
    #print(samples_goal)
    prompts=[]
    goal=[]
    # prompt = 'Let us play the wordle game! '
    for sample in samples_goal:
        sample=sample.strip(' ').strip('\n').strip('\'').strip('\"')
        not_present=set()
        prompt_i='Please guess a word with five letters.'
        contain=set()
        for i in range(5):
            #prompt_abstract = 'I have guessed '+str(i)+' times before. '
            prompt_not_exists = ''
            prompt_exists=''
            if i>0:
                #已经猜了1个词
                guessi=generate_rand_choice_excluding(sample,candidates)
                guessi=guessi.strip(' ').strip('\n').strip('\'').strip('\"')

                _, _,contained = eval_a_word(sample, guessi)
                # if len(contain_and_correct_location_idx)>0:
                #     for x in contain_and_correct_location_idx:
                #         contain_and_correct_letters.add(guessi[x])
                #         prompt_i += '\''+guessi[x]+ '\' appears in the '+str(x+1)+'th position. '
                #
                # if len(contain_but_not_correct_location_idx)>0:
                #     for x in contain_but_not_correct_location_idx :
                #         if guessi[x] not in contain_and_correct_letters:
                #             prompt_i += '\'' + guessi[x] + '\' appeared but not in the '+str(x+1)+'th position. '
                contain = list(contain.union(contained))
                if len(contain) > 0:
                    prompt_exists = "Exists letters: " + (",".join(contain)) + ". "
                contain = set(contain)

                not_present=not_present.union(set(guessi) - set(contain) )
                not_present = list(not_present)
                if len(not_present)>0:
                    prompt_not_exists="Absents letters: "+(",".join(not_present))+"."
                not_present = set(not_present)
            goal.append(sample)
            prompts.append(prompt_i+prompt_exists+prompt_not_exists+' ')

    def reward_fn(

            samples: List[str],
            prompts: List[str],
            outputs: List[str]
    ) -> Dict[str, List[float]]:
        """Metric Function

        Args:
            samples: Batch of samples

        Returns:

            list of metric values for each batch item.
        """
        #先进行解析：
        rewards=[]
        for i,sample in enumerate(samples):

            res = outputs[i].replace('</s>','')
            if len(res)>0:
                guess_word=res.lower()
            else:
                guess_word=''

            #guess_times = re.findall(r'I have guessed ([012345]) times before', prompts[i])
            guess_word = guess_word.strip(' ')
            guess_word=guess_word.strip('\\')

            reward=0
            for chara in guess_word:
                if chara not in set('abcdefghijklmnopqrstuvwxyz'):
                    reward=-10

            if len(guess_word)!=5 :
                reward-=  abs(len(guess_word) - 5) * 1.5
                #print('1')
                #reward=-10
            #else:reward+=20


            # elif guess_times[0]=='0':
            #     reward += 0
            else:
                reward+=0
                # res0=re.findall(r'次猜测的单词是：([a-zA-Z]{5}),猜测错误！', prompts[i])
                # res1= re.findall(r'该单词中第([01234,]+)位字符出现在目标单词中，且位置正确。', prompts[i])
                # res2 = re.findall(r'该单词中第([01234,]+)位字符出现在目标单词中，但位置错误。', prompts[i])
                # res3 = re.findall(r'字母([a-zA-Z,]+)不出现在目标单词中。', prompts[i])
                #res1= re.findall(r'\'([a-zA-Z])\' appears in the ([012345])th position.', prompts[i])
                #res2 = re.findall(r'\'([a-zA-Z])\' appeared but not in the ([012345])th position.', prompts[i])
                #res3 = re.findall(r'\'([a-zA-Z,]+)\' do not exist in the target word. ', prompts[i])
                res1= re.findall(r'exists: (a-zA-Z,]+)', prompts[i])
                res2 = re.findall(r'absents: (a-zA-Z,]+)', prompts[i])
                #print('res3',res3)
                not_contained=[]
                contained=[]
                # contain_and_correct_location_letters=[]
                # contain_and_correct_location_idx=[]
                # contain_but_not_correct_location_idx=[]
                # contain_but_not_correct_location_letters=[]
                # for item in res1:
                #     contain_and_correct_location_letters.append(item[0])
                #     contain_and_correct_location_idx.append(int(item[1])-1)
                # for item in res2:
                #     contain_but_not_correct_location_letters.append(item[0])
                #     contain_but_not_correct_location_idx.append(int(item[1])-1)
                for item in res1:
                    contained.extend(item.split(','))
                    contained=set(contained)
                for item in res2:
                    not_contained.extend(item.split(','))
                    not_contained=set(not_contained)

                for idx,chara in enumerate(guess_word):
                    if chara in not_contained:
                        reward-=2
                    if chara in contained:
                        reward+=2
                for chara in not_contained:
                    if chara not in set(guess_word):
                        reward+=2


                    # for j,item in enumerate(contain_and_correct_location_letters):
                    #
                    #     if chara ==item and idx == contain_and_correct_location_idx[j]:
                    #         reward += 2
                    # for j,item in enumerate(contain_but_not_correct_location_letters):
                    #     if chara ==item and idx == contain_but_not_correct_location_idx[j]:
                    #         reward -= 1
                    #     elif chara ==item and idx != contain_but_not_correct_location_idx[j]:
                    #         reward+=1
            print('guess_word:', guess_word, 'score:', reward)
            rewards.append(float(reward))
        return {

            "optimality": rewards,
        }

    return (reward_fn, prompts,goal)


# metric_fn, prompts= generate_random_words()#此函数是为了生成一系列prompts
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
# output=['fse f f f f f f f']
# prompt=['Let us play the wordle game! I have guessed 0 times before. Please guess the target word with five letters:']
# sample=['1']
# reward_fn, prompts,goal = generate_random_words(n_walks=1)
# score= reward_fn(sample,prompt,output)["optimality"]
# print(score)
