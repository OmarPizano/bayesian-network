import pandas as pd
import numpy as np

def label_prob(df:pd.DataFrame, feature:str, label:str, deps:dict) -> float:
    subdf = df.copy()
    for dep in deps:
        subdf = subdf[subdf[dep] == deps[dep]]
    ocurrences = subdf.groupby(feature).size().get(label)
    if ocurrences == None:
        return 0.0
    return ocurrences / float(len(subdf)) # p(x|deps) = p(deps|x) / p(deps)

def joint_prob(df:pd.DataFrame, values:dict, bn:dict) -> float:
    probs = []
    for node in bn:
        deps = {}
        for dep in bn[node]:
            deps[dep] = values[dep]
        probs.append(label_prob(df, node, values[node], deps)) 
    #print(probs)
    return np.prod(probs) # p(x1,x2,..xn) = p(x1|deps) * p(x2|deps) * ... * p(xn|deps)

def get_class_probs(df:pd.DataFrame, class_ft:str, sample:dict, bn:dict, labels:dict) -> dict:
    jprobs = {} # joint probs
    lprobs = {} # final probs
    for label in labels[class_ft]:
        sample[class_ft] = label # we need current label in the sample to compute joint prob
        jprobs[label] = joint_prob(df, sample, bn)
    for label in labels[class_ft]:
        if jprobs[label] == 0:
            lprobs[label] = 0.0
        else:
            # p(class1|sample) = p(class1,sample) / ( p(class1,sample) + p(class2,sample) )
            lprobs[label] = jprobs[label] / np.sum([jprobs[label] for label in jprobs])
    return lprobs

def categorize_binary(col:pd.Series, labels:list, bins:list) -> pd.Series:
    if len(bins) != 3:
        return pd.cut(col, [col.min()-1, col.median(), col.max()], labels=labels)
    else:
        return pd.cut(col, bins, labels=labels)

