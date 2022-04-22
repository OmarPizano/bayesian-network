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

print(df)

# Test
sample = {'estatura':'alta', 'pie':'grande', 'espalda':'robusta', 'craneo':'grande', 'peso':'pesado'}
print(get_class_probs(df, 'sexo', sample, bn, labels))
