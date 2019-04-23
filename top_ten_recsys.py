import pandas as pd


def top_ten_recommender(test, verbose=True):
    """
    Function to generate top ten recommendations based on top interactions.

    :param test: test dataframe to be passed in order to predict top ten recommendations
    :param verbose: boolean for prints in stdout or not
    :return: a dataframe with idpostulante and top ten idaviso
    """

    if verbose:
        print("Reading CSV...")
    train = pd.read_pickle('data/train.pkl')

    if verbose:
        print("Grouping Results by Aviso...")
    grouped = train.groupby('idaviso').sum().reset_index()

    if verbose:
        print("Sorting Results by Avisos with top interactions...")
    top_ten = grouped.sort_values('idaviso', ascending=False).head(10)

    if verbose:
        print("Printing Top Ten Avisos...")
        print(top_ten)
    top_ten['key'] = 1

    prediction = top_ten[['key', 'idaviso']]

    if verbose:
        print("Printing top ten prediction...")
        print(prediction)

    test = test
    test['key'] = 1

    if verbose:
        print(test.head(10))

    submission = pd.merge(
        prediction[['key', 'idaviso']],
        test[['key', 'idpostulante']],
        left_on='key',
        right_on='key',
        how='right'
    )

    print(submission[['idpostulante', 'idaviso']].head(30))

    return submission[['idpostulante', 'idaviso']]
