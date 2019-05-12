import sqlite3
from sqlite3 import Error
import pandas as pd

def create_connection(db_file=':memory:'):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def close_connection(conn):
    conn.close()



def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def copy_expert_sqlite(conn, filepath, table_name):
    cur = conn.cursor()

    with open(filepath, 'r') as f:
        cur.execute('COPY {} FROM STDIN WITH CSV HEADER'.format(table_name), f)

    conn.commit()

def copy_table_from_df(conn, filepath, table_name):
    file = pd.read_csv(filepath)
    file.to_sql(table_name, conn, index=False)

if __name__ == '__main__':
    print('Creating SQlite DB in memory...')
    conn = create_connection()
    print(conn)
    print('Uploading df to SQlite Table...')
    copy_table_from_df(conn=conn, filepath='data/postulaciones_train.csv', table_name='tabla')

    sql = """
    SELECT 
            idpostulante,
            idaviso,
            count(*) AS cant_vista
    FROM tabla 
    GROUP BY 1,2
    ORDER BY 1,3 DESC
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

    #top_ten = "a"

    #submission.loc[submission.idaviso.isnull(), ['idaviso']] = top_ten
    #print(len(submission[submission.idaviso.notnull()]))
    #print(len(submission[submission.idaviso.isnull()]))

    submission.to_csv('salidas/submission.csv', index=False)
    close_connection(conn=conn)
