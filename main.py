import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
import recsys as rs
import scipy.sparse as sps
from collections import defaultdict
from top_ten_user_filter import top_ten_custom
import sys

if __name__ == "__main__":
    users = pd.read_pickle('preprocessing/entrada/user_features.pkl')
    train = pd.read_pickle('data/train.pkl')
    #train = train[:10000]
    test  = pd.read_pickle('data/test.pkl')
    #test  = test[:100]
    print("Users in users:%s" % (users.idpostulante.nunique()))
    print("Users in train:%s" % (train.idpostulante.drop_duplicates().nunique()))
    print("Users in test:%s" % (test.idpostulante.drop_duplicates().nunique()))

    join_train = pd.merge(
        left=users.idpostulante.to_frame(),
        right=train.idpostulante.drop_duplicates().to_frame(),
        on='idpostulante',
        how='inner'
    )
    print("Users in join_train:%s" % (join_train.idpostulante.nunique()))


    join_test = pd.merge(
        left=users.idpostulante.to_frame(),
        right=test.idpostulante.drop_duplicates().to_frame(),
        on='idpostulante',
        how='inner'
    )
    print("Users in join_test:%s" % (join_test.idpostulante.nunique()))



    join_test_train = pd.merge(
        left=train.idpostulante.drop_duplicates().to_frame(),
        right=test.idpostulante.drop_duplicates().to_frame(),
        on='idpostulante',
        how='inner'
    )
    print("Users in join_test_train:%s" % (join_test_train.idpostulante.nunique()))

    print(users.head())
    print(users.columns)

    print(50*'-')
    u_list = pd.concat([users.idpostulante, train.idpostulante]).drop_duplicates().reset_index().idpostulante
    #print(u_list)

    i_list = train.idaviso.drop_duplicates().reset_index().idaviso
    #print(i_list)

    print(50 * '-')
    print('Building LightFM Dataset...')
    print(50 * '-')
    lfm_dataset = Dataset(user_identity_features=False, item_identity_features=False)


    lfm_dataset.fit(
        users=u_list,
        items=i_list,
        user_features=np.concatenate((users.edad.drop_duplicates().values,
                                      users.sexo.drop_duplicates().values,
                                      users.educacion.drop_duplicates().values),
                                     axis=0)
    )

    print('Retrieving internal mappings and dictionaries...')
    u_map, u_feat_map, i_map, i_feat_map = lfm_dataset.mapping()

    print(50 * '-')
    print('Building Interactions...')
    print(50 * '-')
    interactions = train.groupby(['idpostulante','idaviso']).agg('count').rename(
        columns={'fechapostulacion': 'rating'}).reset_index()
    #print(interactions.sort_values('rating', ascending=False).head())

    interactions = np.array(
        [
            interactions.idpostulante.values,
            interactions.idaviso.values,
            interactions.rating.values
        ],
        dtype=np.object).T

    res_interactions, res_weights = lfm_dataset.build_interactions(data=interactions)

    print(50 * '-')
    print('Building User Features...')
    print(50 * '-')

    users_features = np.array([users.idpostulante.values, users[['edad', 'sexo', 'educacion']].values.tolist()], dtype=np.object).T
    print(users_features)
    print(50*'-')

    u_feat = lfm_dataset.build_user_features(data=users_features, normalize=False)


    print(50 * '-')

    print('Matrix factorization model...')
    mf_model = rs.runMF(interactions=res_interactions,
                        n_components=30,
                        loss='warp',
                        k=15,
                        epoch=50,
                        n_jobs=24)
    print(50 * "-")
    print('Building Recommendations...')

    users = pd.Series(data=list(u_map.keys()), name='idpostulante')
    print(users.to_frame().head())
    users = pd.merge(
        left=test.idpostulante.to_frame(),
        right=users.to_frame(),
        how='inner',
        on='idpostulante'
    )

    dict = defaultdict(list)
    users = users.values.T[0]
    print(50*'-')

    for i, user in enumerate(users):
        sys.stdout.write(
            "\rRecomendacion numero: " + str(i) + "/ " + str(len(users)))
        sys.stdout.flush()
        rec_list = rs.sample_recommendation_user(model=mf_model,
                                                 interactions=res_interactions,
                                                 user_id=user,
                                                 user_dict=u_map,
                                                 item_dict=i_map,
                                                 threshold=0,
                                                 nrec_items=10,
                                                 show=False,
                                                 user_feat=u_feat,
                                                 njobs=24)
        dict[user].append(rec_list)

    #print(dict)

    print('Grouping results per client...')
    lst = []
    for key in dict:
        for recommendation in dict[key][0]:
            lst.append([key, recommendation])

    #print(lst)

    prediction = pd.DataFrame(lst, columns=['idpostulante', 'idaviso'])
    prediction['idaviso'] = prediction['idaviso'].astype(int).astype('str')

    print(prediction.head(30))

    # test = pd.read_csv('data/ejemplo_solution.csv')

    print('Merging results...')
    submission = pd.merge(
        test[['idpostulante']],
        prediction,
        left_on='idpostulante',
        right_on='idpostulante',
        how='left'
    )
    print(submission[submission.idaviso.notnull()].head(15))
    print(50 * "-")
    print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idaviso.notnull()])))
    print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idaviso.isnull()])))

    top_ten_prediction = top_ten_custom()
    top_ten_prediction = pd.merge(
        left=submission[submission.idaviso.isnull()].reset_index(),
        right=top_ten_prediction,
        how='inner',
        on='idpostulante'
    )

    top_ten_prediction['idaviso'] = top_ten_prediction.idaviso_y
    top_ten_prediction = top_ten_prediction[['idpostulante', 'idaviso']]
    print(50 * "-")
    print("Revisar")
    print(50 * "-")

    print(top_ten_prediction.head(15))

    submission = pd.concat([submission[submission.idaviso.notnull()], top_ten_prediction])
    print(submission.head(15))
    print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idaviso.notnull()])))
    print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idaviso.isnull()])))

    print('Saving Results...')
    submission[['idpostulante', 'idaviso']].to_csv('salidas/submission.csv', index=False)





