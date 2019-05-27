import numpy as np
import lightfm as lfm
from lightfm import data

T = [
    ('u1', 'i1'),
    ('u2', 'i1'), ('u2', 'i2'),
                  ('u3', 'i2'), ('u3', 'i3'),                ('u3', 'i5'),
                  ('u4', 'i2'), ('u4', 'i3'), ('u4', 'i4'),  ('u4', 'i5'),
                  ('u5', 'i2'),                              ('u5', 'i5'),
]
print(T)
print(50*'-')
F = [
    ('i1', [      'f1', 'f3']),
    ('i2', [      'f1', 'f3']),
    ('i3', ['f1', 'f2']),
    ('i4', ['f1', 'f2']),
    ('i5', ['f1', 'f2']),
]
print(F)
print(50*'-')


U = set([u for (u, _) in T])
I = set([i for (_, i) in T])

dataset = lfm.data.Dataset()

dataset.fit(users=U, items=I)
dataset.fit_partial(item_features=set([b for (_, f) in F for b in f]))

interactions, weights = dataset.build_interactions(T)
item_features = dataset.build_item_features(F, normalize=False)
user_id_mapping, user_feature_mapping, item_id_mapping, item_feature_mapping = dataset.mapping()

model = lfm.LightFM(no_components=3, loss='warp', learning_schedule='adagrad')
model.fit(interactions=interactions, sample_weight=weights, item_features=item_features, epochs=10, verbose=True)

avisos_a_predecir = np.array(['i1', 'i2', 'i3', 'i4', 'i5'])
avisos_a_predecir = np.array(['i4', 'i1', 'i3'])



for _ in range(10):
    print(50*'-')
    np.random.shuffle(avisos_a_predecir)
    print(avisos_a_predecir)
    p = model.predict(user_id_mapping['u5'], [item_id_mapping[a] for a in avisos_a_predecir])
    print(-p)
    print(np.argsort(-p))
    print(avisos_a_predecir[np.argsort(-p)])
    print(50 * '-')


A = np.array([0, 1, 2, 3, 4, 5])
B = np.array([1, 3, 5])

print(A[~np.in1d(A, B)])

