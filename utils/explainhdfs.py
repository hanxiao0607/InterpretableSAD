from IPython.display import display, HTML
import torch
import collections
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients


def interpret_sequecne(model, device, lig, sequence, baseline, target=1):
    model.train()
    model.zero_grad()
    input = torch.LongTensor(sequence).to(device)
    input = input.unsqueeze(0)
    baseline = torch.LongTensor(baseline).to(device).unsqueeze(0)
    attr, _ = lig.attribute(input, n_steps=2000, target = target, return_convergence_delta=True)
    attr = list(attr.sum(dim=2).squeeze(0).detach().cpu().numpy())
    # attr = normalize(attr.reshape(1, -1))
    baseline = baseline.cpu()
    input = input.cpu()
    return attr

def generate_baseline(model, device, sequence, dic_app, start):
    baseline = sequence[:]
    dic_seq = collections.Counter(baseline)
    lst_app = []
    count = 0
    for (key, val) in dic_seq.most_common()[::-1]:
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

def get_baseline(sequence, model, device, dic_app, start):
    baseline = generate_baseline(model, device, sequence, dic_app, start)

    return baseline

def mean_attr(lst_attr, seq):
    df = pd.DataFrame()
    df['Attr'] = lst_attr
    df['Seq'] = list(seq)
    attr_return = []
    for i in range(len(df)):
        if df['Attr'].iloc[i] == 0:
            attr_return.append(0)
        else:
            key = df['Seq'].iloc[i]
            attr_return.append(np.mean(df['Attr'].loc[df['Seq'] == key].values))
    return attr_return

def get_dataset(model, device, lig, df, dic_app, start):
    lst_attr = []
    lst_blk = []
    lst_EventSeq = []
    lst_keys = []
    lst_baseline = []
    count = 0
    for i in tqdm(range(0,len(df))):
        if len(df['W2V_EventId'].iloc[i]) < 10:
            continue
        if len(df['W2V_EventId'].iloc[i]) > 30:
            continue
        input = torch.LongTensor(df['W2V_EventId'].iloc[i]).to(device)
        input = input.unsqueeze(0)
        pred = int(torch.argmax(model(input)).detach().cpu().numpy())
        if pred != 1:
            count += 1
            continue
        baseline = get_baseline(df['W2V_EventId'].iloc[i], model, device, dic_app, start)
        attr = interpret_sequecne(model, device, lig, df['W2V_EventId'].iloc[i], baseline, 1)

        attr = mean_attr(attr, df['W2V_EventId'].iloc[i])
            
        lst_attr.append(attr)
        lst_keys.append(df['W2V_EventId'].iloc[i])
        lst_baseline.append(baseline)
        lst_blk.append(df['BlockId'].iloc[i])
        lst_EventSeq.append(df['EventSequence'].iloc[i])
    df = pd.DataFrame()
    df['Attr'] = lst_attr
    df['Seq'] = lst_keys
    df['Baseline'] = lst_baseline
    df['Blk'] = lst_blk
    df['Event'] = lst_EventSeq
    print('Total wrong predicted sequence number:', count)
    return df

def visualize_token_attrs(tokens, attrs, blk):
  """
  Visualize attributions for given set of tokens.
  Args:
  - tokens: An array of tokens
  - attrs: An array of attributions, of same size as 'tokens',
    with attrs[i] being the attribution to tokens[i]
  
  Returns:
  - visualization: An IPython.core.display.HTML object showing
    tokens color-coded based on strength of their attribution.
  """
  def get_color(attr):
    if attr > 0:
      r = int(128*attr) + 127
      g = 128 - int(64*attr)
      b = 128 - int(64*attr) 
    else:
      r = 128 + int(64*attr)
      g = int(-128*attr) + 127 
      b = 128 + int(64*attr)
    return r,g,b

  # normalize attributions for visualization.
  bound = max(abs(attrs.max()), abs(attrs.min()))
  attrs = attrs/bound
  html_text = str(blk)
  for i, tok in enumerate(tokens):
    r,g,b = get_color(attrs[i])
    html_text += " <span style='color:rgb(%d,%d,%d)'>%s</span>" % (r, g, b, tok)
  return HTML(html_text)