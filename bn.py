import pandas as pd
from numpy import prod

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
    return prod(probs) # p(x1,x2,..xn) = p(x1|deps) * p(x2|deps) * ... * p(xn|deps)

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
            # p(class1|sample) = ( p(class1,sample) + p(class2,sample) ) / p(class1,sample)
            lprobs[label] = sum([jprobs[label] for label in jprobs]) / jprobs[label]
    return lprobs


df = pd.read_csv('dataset.csv')

df = df.drop(['edad', 'brazo', 'espalda', 'craneo'], axis=1)

# TODO: put this in a function
df['estatura'] = pd.cut(df['estatura'], [0, 160, 180], labels=['baja', 'alta'])
df['pie'] = pd.cut(df['pie'], [20, 24, 30], labels=['chico', 'grande'])
df['peso'] = pd.cut(df['peso'], [25, 60, 110], labels=['ligero', 'pesado'])

bn = {
        'sexo': ['peso', 'estatura', 'pie'],
        'peso': ['estatura'],
        'estatura': [],
        'pie': []
        }

labels = {
        'sexo': ['hombre', 'mujer'],
        'peso': ['ligero', 'pesado'],
        'estatura': ['baja', 'alta'],
        'pie': ['chico', 'grande']
        }

# Test
print(get_class_probs(df, 'sexo', {'estatura':'alta', 'peso':'pesado', 'pie':'grande'}, bn, labels))
