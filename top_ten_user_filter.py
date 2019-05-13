import sqlite3
from sqlite3 import Error
import pandas as pd
import sqlite_functions.sqlite as sql_fun

def top_ten_custom():
    print('Creating SQlite DB in memory...')
    conn = sql_fun.create_connection()
    print(conn)
    print('Uploading df to SQlite Table...')
    sql_fun.copy_table_from_df(conn=conn, filepath='data/postulaciones_train.csv', table_name='train')
    sql_fun.copy_table_from_df(conn=conn, filepath='data/ejemplo_solution.csv', table_name='test')


    sql = """
    SELECT *
    FROM 
    (SELECT idpostulante
    FROM test)
    NATURAL JOIN
    (
    SELECT
        idaviso,
        count(*) AS cant_vista
    FROM train
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 50)
    WHERE
        (idpostulante || '-' || idaviso) 
            NOT IN
                (
                SELECT
                    (idpostulante || '-' || idaviso) as id_comp
                FROM train
                )
    ;
    """
    print('Executing SQL Statement...')
    c = conn.cursor()
    c.execute(sql)
    print('Fetching Results...')
    rows = c.fetchall()

    print('Reading rows...')
    from collections import defaultdict
    dict = defaultdict(list)
    for row in rows:
        dict[row[0]].append(row[1])

    print('Grouping results per client...')
    lst = []
    for key in dict:
        if len(dict[key]) > 10:
            for i in range(10):
                lst.append([key, dict[key][i]])
        else:
            for i in range(len(dict[key])):
                lst.append([key, dict[key][i]])

    #print(lst)
    df = pd.DataFrame(lst, columns=['idpostulante', 'idaviso'])
    df['idaviso'] = df['idaviso'].astype(int).astype('str')
    print(df.head(30))

    test = pd.read_csv('data/ejemplo_solution.csv')
    print(test.shape)
    print('Merging results...')
    submission = pd.merge(
        test[['idpostulante']],
        df,
        left_on='idpostulante',
        right_on='idpostulante',
        how='left'
    )
    print(submission[submission.idaviso.notnull()].head(30))
    print(len(submission[submission.idaviso.notnull()]))
    print(len(submission[submission.idaviso.isnull()]))


    #submission.to_csv('salidas/submission.csv', index=False)
    sql_fun.close_connection(conn=conn)
    return submission
