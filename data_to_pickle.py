import pandas as pd

print('Loading train...')
train = pd.read_csv('data/postulaciones_train.csv')
print(train.head())
print('Saving train to pickle...')
train.to_pickle('data/train.pkl')

print('Loading test...')
test = pd.read_csv('data/ejemplo_solution.csv')
print(test.head())
print('Saving test to pickle...')
test.to_pickle('data/test.pkl')

print('Loading Postulantes Educacion...')
educacion = pd.read_csv('data/postulantes_educacion.csv')
print(educacion.head())
print('Saving Postulantes Educacion to pickle...')
educacion.to_pickle('data/educacion_user_feat.pkl')

print('Loading Postulantes Genero Edad...')
genero_edad = pd.read_csv('data/postulantes_genero_edad.csv')
print(genero_edad.head())
print('Saving Postulantes Genero Edad to pickle...')
genero_edad.to_pickle('data/genero_edad_user_feat.pkl')

print('Loading Avisos...')
avisos = pd.read_csv('data/avisos_detalle.csv')
print(avisos.head())
print('Saving Avisos to pickle...')
avisos.to_pickle('data/avisos.pkl')

