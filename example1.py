from bn import *

bn = {
        'sexo': ['peso', 'pie'],
        'estatura': [],
#        'edad': [],
        'pie': ['estatura'],
#        'brazo': ['edad'],
        'espalda': ['sexo'],
        'craneo' : ['sexo'],
        'peso': ['estatura'],
        }

labels = {
        'sexo': ['hombre', 'mujer'],
        'estatura': ['baja', 'alta'],
#        'edad': ['joven', 'viejo'],
        'pie': ['chico', 'grande'],
#        'brazo': ['corto', 'largo'],
        'espalda': ['delgada', 'robusta'],
        'craneo': ['chico', 'grande'],
        'peso': ['ligero', 'pesado'],
        }

df = pd.read_csv('dataset.csv')

df = df.drop(['edad', 'brazo'], axis=1)

df['estatura'] = categorize_binary(df['estatura'], labels['estatura'], [120, 160, 200])
#df['edad'] = categorize_binary(df['edad'], labels['edad'], [1, 35, 100])
df['pie'] = categorize_binary(df['pie'], labels['pie'], [15, 24, 30])
#df['brazo'] = categorize_binary(df['brazo'], labels['brazo'], [45, 60, 75])
df['espalda'] = categorize_binary(df['espalda'], labels['espalda'], [30, 45, 60])
df['craneo'] = categorize_binary(df['craneo'], labels['craneo'], [45, 54, 60])
df['peso'] = categorize_binary(df['peso'], labels['peso'], [25, 70, 110])

#print(df)

# Test for gender 1
# In this case, we have a 100% for the 'hombre' class because the dataset.
# sample = {'estatura':'alta', 'pie':'grande', 'espalda':'robusta', 'craneo':'grande', 'peso':'pesado'}
# print(get_class_probs(df, 'sexo', sample, bn, labels))

# Test for gender 2
# In this case, we have 0% for both 'hombre' & 'mujer' because we do not have any case with these characteristics.
# BUT, if we add 'mujer,179,24,24,51.5,50,56.2,80' to our CSV, we will have 100%  for 'mujer'. So, the size and diversity
# of our dataset have a high impact in the bayesian network performance.
#sample = {'estatura':'alta', 'pie':'chico', 'espalda':'robusta', 'craneo':'grande', 'peso':'pesado'}
#print(get_class_probs(df, 'sexo', sample, bn, labels))

# Test for weight
# Finally, if we want to test the probability of any other node (i.e. 'peso'), we just have to put that node in the
# function 'get_class_probs', and take the 'sexo' node back to our sample.
sample = {'estatura':'alta', 'pie':'grande', 'espalda':'robusta', 'craneo':'grande', 'sexo':'hombre'}
print(get_class_probs(df, 'peso', sample, bn, labels))
