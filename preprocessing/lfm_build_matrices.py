import pandas as pd
import sqlite_functions.sqlite as sql_fun
import scipy.sparse as sps
import numpy as np
from sklearn.feature_extraction import DictVectorizer


def build_user_item_mtrx(week_ID):
    """ Build user item matrix (for test and train datasets)
    (sparse matrix, Mui[u,i] = 1 if user u has purchase item i, 0 otherwise)

    arg : week_ID (str) validation week
    """

    print("Creating user_item matrix for LightFM")

    # For now, only consider the detail dataset
    cpdtr = pd.read_csv(
        "../Data/Validation/%s/coupon_detail_train_validation_%s.csv" %
        (week_ID, week_ID))
    cpltr = pd.read_csv(
        "../Data/Validation/%s/coupon_list_train_validation_%s.csv" %
        (week_ID, week_ID))
    cplte = pd.read_csv(
        "../Data/Validation/%s/coupon_list_test_validation_%s.csv" %
        (week_ID, week_ID))
    ulist = pd.read_csv(
        "../Data/Validation/%s/user_list_validation_%s.csv" %
        (week_ID, week_ID))

    # Build a dict with the coupon index in cpltr
    d_ci_tr = {}
    for i in range(len(cpltr)):
        coupon = cpltr["COUPON_ID_hash"].values[i]
        d_ci_tr[coupon] = i

    # Build a dict with the user index in ulist
    d_ui = {}
    for i in range(len(ulist)):
        user = ulist["USER_ID_hash"].values[i]
        d_ui[user] = i

    # Build the user x item matrices using scipy lil_matrix
    Mui_tr = sps.lil_matrix((len(ulist), len(cpltr)), dtype=np.int8)

    # Now fill Mui_tr with the info from cpdtr
    for i in range(len(cpdtr)):
        sys.stdout.write(
            "\rProcessing row " + str(i) + "/ " + str(cpdtr.shape[0]))
        sys.stdout.flush()
        user = cpdtr["USER_ID_hash"].values[i]
        coupon = cpdtr["COUPON_ID_hash"].values[i]
        ui, ci = d_ui[user], d_ci_tr[coupon]
        Mui_tr[ui, ci] = 1
    print

    # Save the matrix in the COO format
    spi.mmwrite(
        "../Data/Validation/%s/user_item_train_mtrx_%s.mtx" %
        (week_ID, week_ID), Mui_tr)


def build_user_features(filepath):
    """

    :param filepath:
    :return:
    """

    df = pd.read_pickle(filepath)
    vec = DictVectorizer(sparse=True)
    df = df.T.to_dict().values()
    for i in df:
        print(i)
    df = vec.fit_transform(df)
    print(df)

    u_feat = sps.csr_matrix(df, dtype=np.int32)

    return u_feat

