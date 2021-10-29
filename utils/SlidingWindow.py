import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.util import bigrams
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
from collections import defaultdict
import regex as re

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def preprocess(df, window_size = 100, step_size = 20):
    '''Preprocessing structured log dataset

    Args:
        df: dataframe of structured log dataset
        window_size: length of sliding window
        step_size: step length

    Return:
        DataFrame of preprocessed sliding windows
    '''
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    df = df[["Label", "EventId"]]
    df["Key_label"] = df["Label"]
    log_size = df.shape[0]  
    label_data = df.iloc[:, 0]
    logkey_data = df.iloc[:, 1]  
    new_data = []
    index = 0
    while index <= log_size-window_size:
        new_data.append([
            max(label_data[index:index+window_size]),
            logkey_data[index:index+window_size].values,
            label_data[index:index+window_size].values
        ])
        index += step_size
    return pd.DataFrame(new_data, columns=df.columns)

def get_training_dictionary(df):
    '''Get training dictionary

    Arg:
        df: dataframe of preprocessed sliding windows

    Return:
        dictionary of training data
    '''
    dic = {}
    count = 0
    for i in range(len(df)):
        lst = list(df['EventId'].iloc[i])
        for j in lst:
            if j in dic:
                pass
            else:
                dic[j] = str(count)
                count += 1
    return dic

def str_to_str_keys(df, dic):
    '''Convert original parser log keys into number version of log keys

    Args:
        df: dataframe which needs to be converted
        dic: reference dictionary

    Return:
        df: dataframe which EventId column has been converted 
    '''
    for i in range(len(df)):
        lst = list(df['EventId'].iloc[i])
        temp = []
        for j in lst:
            if j in dic:
                temp.append(dic[j])
            else:
                temp.append(str(len(dic)))
        df['EventId'].iloc[i] = temp
    return df

def get_bigram(df):
    '''Get the bigram according to the input dataframe

    Arg:
        df: dataframe which is used to compute bigrams

    Returns:
        bigram: dictionary of bigrams
        uni: sliding window first log key
    '''
    bigram = {}
    uni = []
    for i in range(len(df)):
        temp_lst = list(df['EventId'].iloc[i])
        if temp_lst[0] not in uni:
            uni.append(temp_lst[0])
        for a,b in bigrams(temp_lst):
            a = str(a)
            b = str(b)
            if a in bigram:
                if b not in bigram.get(a):
                    bigram[a].append(b)
            else:
                bigram[a] = [b]
    return bigram, uni

def get_w2v_dic(w2v):
    '''Get Word2Vec dictionary

    Arg:
        w2v: Word2Vec model

    Return:
        dic: dictionary of Word2Vec
    '''
    dic = {}
    for i in list(w2v.wv.vocab):
        dic[i] = w2v.wv.vocab.get(i).index
    return dic

def str_key_to_w2v_index(df, dic):
    '''Chenge string number keys into Word2Vec int keys

    Args:
        df: DataFrame of data which needs to be changed for InterpretableSAD model
        dic: reference dictionary

    Return:
        df: DataFrame of modified data
    '''
    lst_w2v = []
    for i in range(len(df)):
        lst = list(df['EventId'].iloc[i])
        temp = []
        for j in lst:
            if j in dic:
                temp.append(dic[j])
            else:
                print('Error: key is not in the dict')
        lst_w2v.append(temp)
    df['W2V_EventId'] = lst_w2v
    return df    

def sliding(dataset_name, window_size = 100, step_size = 20, train_size = 100000):
    '''Cut log data into sliding windows and train a Word2Vec model

    Args:
        dataset_name: name of log dataset
        window_size: length of sliding window
        step_size: length of step
        train_size: number of training samples

    Returns:
        train_normal: DataFrame of training normal samples
        test_normal: DataFrame of testing normal samples
        test_abnormal: DataFrame of testing abnormal samples
        bigram: dictionary of bigrams from training data
        unique: list of start log keys of training data
        weights: weight matrix of Word2Vec
        train_dict: dictionary of training data
        w2v_dic: dictionary of Word2Vec
    '''
    # slide log data into sliding windows
    print('Reading: ' + dataset_name)
    df = pd.read_csv('Dataset/' + dataset_name + '.log_structured.csv')
    print('Total logs in the dataset: ', len(df))
    if dataset_name == 'HDFS':
        blk_df = pd.read_csv('Dataset/anomaly_label.csv')
        blk_label_dict = {}
        for _ , row in tqdm(blk_df.iterrows()):
            blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

        def hdfs_blk_process(df, dict_blk_label):
            data_dict = defaultdict(list)
            for idx, row in tqdm(df.iterrows()):
                blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
                blkId_set = set(blkId_list)
                for blk_Id in blkId_set:
                    if blk_Id not in data_dict:
                        data_dict[blk_Id] = [row['EventId']]
                    else:
                        data_dict[blk_Id].append(row["EventId"])

            data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        
            data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

            return data_df
            
        hdfs_df = hdfs_blk_process(df, blk_label_dict)

        window_df = hdfs_df
        #########
        # Train #
        #########
        df_normal = window_df[window_df["Label"] == 0]
        # shuffle normal data
        df_normal = df_normal.sample(frac=1, random_state = 42).reset_index(drop=True)
        normal_len = len(df_normal)
        train_len = train_size

        train_normal = df_normal[:train_len]
        print("training size {}".format(train_len))


        ###############
        # Test Normal #
        ###############
        test_normal = df_normal[train_len:]
        print("test normal size {}".format(normal_len - train_len))

        #################
        # Test Abnormal #
        #################
        test_abnormal = window_df[window_df["Label"] == 1]
        print('test abnormal size {}'.format(len(test_abnormal)))

        def get_training_dictionary_hdfs(df):
            dic = {'pad':'0'}
            count = 1
            for i in range(len(df)):
                lst = [x for x in df['EventSequence'].iloc[i]]
                for j in lst:
                    if j in dic:
                        pass
                    else:
                        dic[j] = str(count)
                        count += 1
            return dic

        all_dict = get_training_dictionary_hdfs(window_df)
        train_dict = get_training_dictionary_hdfs(train_normal)
        print('Number of all keys:', len(all_dict))
        print('Number of training keys:',len(train_dict))

        def str_to_str_keys_hdfs(df, dic):
            for i in range(len(df)):
                temp_lst_seqs = []
                lst = df['EventSequence'].iloc[i]
                for j in lst:
                    if j in dic:
                        temp_lst_seqs.append(dic[j])
                    else:
                        temp_lst_seqs.append(str(len(dic)))
                df['EventSequence'].iloc[i] = temp_lst_seqs
            return df
        # change the original log keys into number log keys based on the training dictionary
        train_normal = str_to_str_keys_hdfs(train_normal, train_dict)
        test_normal = str_to_str_keys_hdfs(test_normal, train_dict)
        test_abnormal = str_to_str_keys_hdfs(test_abnormal, train_dict)

        def get_bigram_hdfs(df):
            total = {}
            uni = []
            for i in range(len(df)):
                temp_lst = list(df['EventSequence'].iloc[i])
                if temp_lst[0] not in uni:
                    uni.append(temp_lst[0])
                for a,b in bigrams(temp_lst):
                    a = str(a)
                    b = str(b)
                    if a in total:
                        if b not in total.get(a):
                            total[a].append(b)
                    else:
                        total[a] = [b]
            return total, uni
        # get the bigram dictionary and unique list from the training data
        bigram, unique = get_bigram_hdfs(train_normal)

        # define training data
        sentences = list(train_normal.EventSequence.values)
        sentences.append([str(len(train_dict))])
        sentences.append(['0'])

        # train model
        w2v = Word2Vec(sentences, size=4, min_count=1, seed=1)
        # summarize the loaded model
        print('Word2Vec model:', w2v)
        # get the Word2Vec model weights for lstm embedding layer
        weights = torch.FloatTensor(w2v.wv.vectors)
        # get the Word2Vec dictionary
        w2v_dic = get_w2v_dic(w2v)

        def str_key_to_w2v_index_hdfs(df, dic):
            lst_w2v = []
            for i in tqdm(range(len(df))):
                lst = list(df['EventSequence'].iloc[i])
                temp = []
                for j in lst:
                    if j in dic:
                        temp.append(dic[j])
                    else:
                        print('Error: key is not in the dict')
                lst_w2v.append(temp)
            df['W2V_EventId'] = lst_w2v
            return df

        # change the data with Word2Vec dictionary
        train_normal = str_key_to_w2v_index_hdfs(train_normal, w2v_dic)
        test_normal = str_key_to_w2v_index_hdfs(test_normal, w2v_dic)
        test_abnormal = str_key_to_w2v_index_hdfs(test_abnormal, w2v_dic)

    else:
        window_df = preprocess(df, window_size, step_size)

        #########
        # Train #
        #########
        df_normal = window_df[window_df["Label"] == 0]
        # shuffle normal data
        df_normal = df_normal.sample(frac=1, random_state = 42).reset_index(drop=True)
        normal_len = len(df_normal)
        train_len = train_size

        train_normal = df_normal[:train_len]
        print("training size {}".format(train_len))

        ###############
        # Test Normal #
        ###############
        test_normal = df_normal[train_len:]
        print("test normal size {}".format(normal_len - train_len))

        #################
        # Test Abnormal #
        #################
        test_abnormal = window_df[window_df["Label"] == 1]
        print('test abnormal size {}'.format(len(test_abnormal)))

        # get dictionary of training data and total data
        all_dict = get_training_dictionary(window_df)
        train_dict = get_training_dictionary(train_normal)
        print('Number of all keys:', len(all_dict))
        print('Number of training keys:',len(train_dict))

        # change the original log keys into number log keys based on the training dictionary
        train_normal = str_to_str_keys(train_normal, train_dict)
        test_normal = str_to_str_keys(test_normal, train_dict)
        test_abnormal = str_to_str_keys(test_abnormal, train_dict)

        # get the bigram dictionary and unique list from the training data
        bigram, unique = get_bigram(train_normal)

        # define training data
        sentences = list(train_normal.EventId.values)
        sentences.append([str(len(train_dict))])

        # train model
        w2v = Word2Vec(sentences, size=8, min_count=1, seed=1)
        # summarize the loaded model
        print('Word2Vec model:', w2v)
        # get the Word2Vec model weights for lstm embedding layer
        weights = torch.FloatTensor(w2v.wv.vectors)
        # get the Word2Vec dictionary
        w2v_dic = get_w2v_dic(w2v)

        # change the data with Word2Vec dictionary
        train_normal = str_key_to_w2v_index(train_normal, w2v_dic)
        test_normal = str_key_to_w2v_index(test_normal, w2v_dic)
        test_abnormal = str_key_to_w2v_index(test_abnormal, w2v_dic)

    return train_normal, test_normal, test_abnormal, bigram, unique, weights, train_dict, w2v_dic

