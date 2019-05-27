import numpy as np
import pandas as pd

df = pd.read_csv('submission.csv')

print('En la submission hay {} usuarios únicos'.format(df.idpostulante.nunique()))
train = pd.read_pickle('../data/train.pkl')

print('En train hay {} usuarios únicos'.format(train.idpostulante.nunique()))
print('En train hay {} avisos únicos'.format(train.idaviso.nunique()))
print(train.groupby('idaviso').count().sort_values(by='idpostulante', ascending=False))


join = pd.merge(
    left=df,
    right=train,
    on=['idpostulante','idaviso'],
    how='inner'
)

join_group = join.groupby('idpostulante').count()
print(join_group)