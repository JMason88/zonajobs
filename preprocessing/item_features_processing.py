import pandas as pd
import sqlite_functions.sqlite as sql_fun
import scipy.sparse as sps
import numpy as np
import os

if __name__ == '__main__':
    con = sql_fun.create_connection()

    avisos = pd.read_csv('../data/avisos_detalle.csv')
    avisos.drop_duplicates(inplace=True)
    print(avisos.columns)
    print(50*'-')
    print(avisos.idpais.unique())
    print(50 * '-')
    print(avisos.nombre_area.unique())
    print(50 * '-')
    print(avisos.nivel_laboral.unique())
    print(50 * '-')
    print(avisos.tipo_de_trabajo.unique())
    print(50 * '-')
    print(avisos.ciudad.unique())
    print('Creating avisos table...')
    sql_fun.copy_table_from_df(con, '../data/avisos_detalle.csv', 'avisos', rm_duplicate=True)

    sql_1 = '''
    SELECT
        idaviso,
        nivel_laboral,
        tipo_de_trabajo
    FROM
        avisos
    '''

#    sql_fun.copy_table_from_df(con, '../data/postulaciones_train.csv', 'avisos', rm_duplicate=True)



    df = sql_fun.sql_to_pandas(con, sql_1)

    print(df.head())
    print(os.getcwd())
    df.to_pickle('entrada/item_features.pkl')

