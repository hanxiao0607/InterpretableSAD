from captum.attr import LayerIntegratedGradients, IntegratedGradients
import torch
import collections
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics

def interpret_sequecne(model, device, lig, sequence, baseline, target=1):
    model.train()
    model.zero_grad()
    input = torch.LongTensor(sequence).to(device)
    input = input.unsqueeze(0)
    baseline = torch.LongTensor(baseline).to(device).unsqueeze(0)
    attr, _ = lig.attribute(input, baselines = baseline, n_steps=2000, target = target, return_convergence_delta=True)
    attr = list(attr.sum(dim=2).squeeze(0).detach().cpu().numpy())
    baseline = baseline.cpu()
    input = input.cpu()
    return attr

def generate_baseline(model, device, sequence, dic_app, start, ratio = 0.1, window_size = 100):
    baseline = sequence[:]
    dic_seq = collections.Counter(baseline)
    lst_app = []
    count = 0
    for (key, val) in dic_seq.most_common()[::-1]:
        if val <= ratio*window_size:
            lst_app.append(key)
    temp_dic = {}
    for i in lst_app:
        temp_dic[i] = dic_app.get(i)
    for i in sorted(temp_dic, reverse=True):
        pred_value = model(torch.LongTensor(baseline).to(device).unsqueeze(0)).detach().cpu().numpy()[0]
        abnormal_value = model(torch.LongTensor(sequence).to(device).unsqueeze(0)).detach().cpu().numpy()[0]
        pred = int(torch.argmax(model(torch.LongTensor(baseline).to(device).unsqueeze(0))).detach().cpu().numpy())
        if pred == 0 and pred_value[1] < abnormal_value[1] and pred_value[0] > abnormal_value[0]:
            break
        if count > ratio*window_size:
            break
        else:
            for j in range(len(baseline)):
                if baseline[j] == i:
                    if j == 0:
                        for k in range(1, len(baseline)):
                            if (baseline[k] != baseline[j]) and (baseline[k] in start):
                                baseline[j] = baseline[k]
                                break
                    else:
                        baseline[j] = baseline[j-1]
        count += dic_seq.get(i)
    return baseline

def to_single_array(lst):
    temp = []
    for i in lst:
        temp.extend(i)
    return np.array(temp)

def get_baseline(sequence, model, device, dic_app, start, ratio, window_size):
    min_dis = 999999
    min_index = 0
    emb = get_embedding(model, device, torch.LongTensor(sequence).to(device))
    emb = to_single_array(emb)
    baseline = generate_baseline(model, device, sequence, dic_app, start, ratio, window_size)
    emb_baseline = get_embedding(model, device, torch.LongTensor(baseline).to(device))
    emb_baseline = to_single_array(emb_baseline)
    dist = np.linalg.norm(emb-emb_baseline)
    min_dis = dist

    return baseline, min_dis

def get_embedding(model, device, lst):
    emb = []
    for i in lst:
        emb.append(list(model.embedding(torch.tensor(i).cuda()).detach().cpu().numpy()))
    return emb

def get_dataset(model, device, lig, sequences, key_label_seq, dic_app, start, ratio, window_size):
    lst_attr = []
    lst_y = []
    lst_dist = []
    lst_keys = []
    lst_baseline = []
    count = 0
    for i in tqdm(range(len(sequences))):
        input = torch.LongTensor(sequences.iloc[i]).to(device)
        input = input.unsqueeze(0)
        pred = int(torch.argmax(model(input)).detach().cpu().numpy())
        if pred != 1:
            count += 1
            continue
        # if model prediction not 1 pass this sequence
        baseline, dist = get_baseline(sequences.iloc[i], model, device, dic_app, start, ratio, window_size)
        attr = interpret_sequecne(model, device, lig, sequences.iloc[i], baseline, 1)
        lst_attr.extend(attr)
        lst_y.extend(key_label_seq.iloc[i])
        lst_dist.extend([dist for _ in range(len(attr))])
        lst_keys.extend(sequences.iloc[i])
        lst_baseline.extend(baseline)
    print('Total wrong predicted sequence number:', count)
    return lst_attr, lst_y, lst_dist, lst_keys, lst_baseline

def get_mean_inter(df, window_size = 100, ratio = 0.1):
    length = len(df)
    inter = int(1/ratio)
    best_interception = -1
    lst_pred = []
    lst_y = []
    for i in range(0, int(length/window_size), inter):
        df_temp = df.iloc[i*window_size:(i+1)*window_size]
        lst_temp = []
        for j in range(len(df_temp)):
            if df_temp['attr'].iloc[j] == 0:
                lst_temp.append(0)
            else:
                key = df_temp['key'].iloc[j]
                if np.mean(df_temp['attr'].loc[df_temp['key'] == key].values) > -1:
                    lst_temp.append(1)
                else:
                    lst_temp.append(0)
        lst_pred.extend(lst_temp)
        lst_y.extend(df_temp['y'].values.tolist())
    best_f1 = metrics.f1_score(lst_y, lst_pred, pos_label=1)

    for i in range(1, 21):
        interception = i/10 -1
        lst_pred = []
        lst_y = []
        for i in range(0, int(length/window_size), inter):
            df_temp = df.iloc[i*window_size:(i+1)*window_size]
            lst_temp = []
            for j in range(len(df_temp)):
                if df_temp['attr'].iloc[j] == 0:
                    lst_temp.append(0)
                else:
                    key = df_temp['key'].iloc[j]
                    if np.mean(df_temp['attr'].loc[df_temp['key'] == key].values) > interception:
                        lst_temp.append(1)
                    else:
                        lst_temp.append(0)
            lst_pred.extend(lst_temp)
            lst_y.extend(df_temp['y'].values.tolist())
        f1 = metrics.f1_score(lst_y, lst_pred, pos_label=1)
        pred_count_1 = collections.Counter(lst_pred).get(1)
        if f1 > best_f1:
            best_f1 = f1
            best_interception = interception
    return best_interception

def mean_inter(df, window_size = 100, interception = 0):
    iter_times = int(len(df)/window_size)
    lst_pred = []
    for i in range(iter_times):
        temp_df = df.iloc[i*window_size:(i+1)*window_size]
        lst_temp = []
        for j in range(len(temp_df)):
            if temp_df['attr'].iloc[j] == 0:
                lst_temp.append(0)
            else:
                key = temp_df['key'].iloc[j]
                if np.mean(temp_df['attr'].loc[temp_df['key'] == key].values) > interception:
                    lst_temp.append(1)
                else:
                    lst_temp.append(0)
        lst_pred.extend(lst_temp)
    return lst_pred
