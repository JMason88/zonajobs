import pandas as pd

print("Reading CSV...")
train = pd.read_csv('data/postulaciones_train.csv')

print("Grouping Results by Aviso...")
grouped = train.groupby('idaviso').sum().reset_index()
print("Sorting Results by Avisos with top interactions...")
top_ten = grouped.sort_values('idaviso', ascending=False).head(10)

print("Printing Top Ten Avisos...")
print(top_ten)
top_ten['key'] = 1

prediction = top_ten[['key', 'idaviso']]

print("Printing top ten prediction...")
print(prediction)

test = pd.read_csv('data/ejemplo_solution.csv')
test['key'] = 1

print(test.head(10))

submission = pd.merge(
    prediction[['key', 'idaviso']],
    test[['key', 'idpostulante']],
    left_on='key',
    right_on='key',
    how='right'
)

print(submission[['idpostulante', 'idaviso']].head(30))

print('Saving Results...')
submission[['idpostulante', 'idaviso']].to_csv('salidas/submission.csv', index=False)
