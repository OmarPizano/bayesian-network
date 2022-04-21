import pandas as pd
from numpy import prod

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

def prob_cond_dep(df:pd.DataFrame, feature:str, label:str, deps:dict) -> float:
    subdf = df.copy() # TODO: test if needed
    for dep in deps:
        subdf = subdf[subdf[dep] == deps[dep]]
    ocurrences = subdf.groupby(feature).size().get(label)
    if ocurrences == None:
        return 0.0
    return ocurrences / float(len(subdf))

def prob_conj(df:pd.DataFrame, values:dict, bn:dict) -> float:
    probs = []
    for node in bn:
        deps = {}
        for dep in bn[node]:
            deps[dep] = values[dep]
        probs.append(prob_cond_dep(df, node, values[node], deps)) 
    print(probs)
    return prod(probs)

r = prob_conj(df, {'sexo':'hombre', 'estatura':'alta', 'peso':'pesado', 'pie':'grande'}, bn)
print(r)

# def get_class(df:pd.DataFrame, class_ft:str, ) -> str:
    # get_prob_cond()

