import random
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
  
def get_neg_samp(window, index, bigram, uni, vocab_dim):
    '''Get negative sample for each given sliding window

    Args:
        window: a sample of normal log sequence
        index: list of indexes which need to be replaced
        bigram: bigram dictionary for reference
        uni: list of start log keys for reference
        vocab_dim: int of vocabulary size

    Return:
        window: a negative sample
    '''
    for i in index:
        if i == 0:
            in_bag = set(uni)
            out_bag = set(range(0,vocab_dim)).difference(in_bag)
            window[i] = str(random.sample(out_bag, 1)[0])
        else:
            if str(window[i]) in bigram:
                in_bag = set(bigram.get(str(window[i])))
                out_bag = set(range(0,vocab_dim)).difference(in_bag)
                window[i+1] = str(random.sample(out_bag, 1)[0])
            else:
                out_bag = set(range(0, vocab_dim))
                window[i+1] = str(random.sample(out_bag, 1)[0])
    return window

def negative_sampling(dataset, bigram, uni, number_times, vocab_dim):
    '''Negative sampling method

    Args:
        dataset: DataFrame of training normal dataset
        bigram: bigram dictionary for reference
        uni: list of start log keys for reference
        number_times: int number of times to decide the size of negative sampling size
        vocab_dim: int of vocabulary size

    Return:
        list of negative samples 
    '''
    length = len(dataset)
    re_len = int(number_times * length)
    re_list = []
    lst_keys = list(dataset['EventId'])
    samples = list(np.random.random_integers(length-1, size=(re_len,)))
    for i in tqdm(map(lst_keys.__getitem__, samples)):
        replace_n = random.randint(1,len(i))
        rep_index = random.choices(range(len(i)-1), k = replace_n)
        temp = i[:]
        while temp in lst_keys:
            temp = get_neg_samp(temp, rep_index, bigram, uni, vocab_dim)
        re_list.append(temp)  
    return re_list

def negative_sampling_hdfs(dataset, bigram, uni, number_times, vocab_dim):
    '''Negative sampling method

    Args:
        dataset: DataFrame of training normal dataset
        bigram: bigram dictionary for reference
        uni: list of start log keys for reference
        number_times: int number of times to decide the size of negative sampling size
        vocab_dim: int of vocabulary size

    Return:
        list of negative samples 
    '''
    length = len(dataset)
    re_len = int(number_times * length)
    re_list = []
    lst_keys = list(dataset['EventSequence'])
    samples = list(np.random.random_integers(length-1, size=(re_len,)))
    for i in tqdm(map(lst_keys.__getitem__, samples)):
        replace_n = random.randint(1,len(i))
        rep_index = random.choices(range(len(i)-1), k = replace_n)
        temp = i[:]
        while temp in lst_keys:
            temp = get_neg_samp(temp, rep_index, bigram, uni, vocab_dim)
        re_list.append(temp)  
    return re_list