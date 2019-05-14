import pandas as pd
import sqlite_functions.sqlite as sql_fun


if __name__ == '__main__':
    con = sql_fun.create_connection()

    genero = pd.read_csv('data/postulantes_genero_edad.csv')
    print(genero.fechanacimiento.unique())
    print(genero.sexo.unique())
    print('Creating genero_edad table...')
    sql_fun.copy_table_from_df(con, 'data/postulantes_genero_edad.csv', 'genero_edad')
    print('Creating educacion table...')
    #sql_fun.copy_table_from_df(con, 'data/postulantes_educacion.csv', 'educacion')

    sql_1 = '''
    SELECT
        idpostulante,
        sum(CASE WHEN nombre = 'Posgrado' AND estado = 'En Curso' THEN 1 ELSE 0 END) as posgrado_curso,
        sum(CASE WHEN nombre = 'Posgrado' AND estado = 'Graduado' THEN 1 ELSE 0 END) as posgrado_graduado,
        sum(CASE WHEN nombre = 'Posgrado' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as posgrado_abandonado,
        sum(CASE WHEN nombre = 'Universitario' AND estado = 'En Curso' THEN 1 ELSE 0 END) as universitario_curso,
        sum(CASE WHEN nombre = 'Universitario' AND estado = 'Graduado' THEN 1 ELSE 0 END) as universitario_graduado,
        sum(CASE WHEN nombre = 'Universitario' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as universitario_abandonado,
        sum(CASE WHEN nombre = 'Master' AND estado = 'En Curso' THEN 1 ELSE 0 END) as master_curso,
        sum(CASE WHEN nombre = 'Master' AND estado = 'Graduado' THEN 1 ELSE 0 END) as master_graduado,
        sum(CASE WHEN nombre = 'Master' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as master_abandonado,
        sum(CASE WHEN nombre = 'Otro' AND estado = 'En Curso' THEN 1 ELSE 0 END) as otro_curso,
        sum(CASE WHEN nombre = 'Otro' AND estado = 'Graduado' THEN 1 ELSE 0 END) as otro_graduado,
        sum(CASE WHEN nombre = 'Otro' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as otro_abandonado,
        sum(CASE WHEN nombre = 'Terciario/Técnico' AND estado = 'En Curso' THEN 1 ELSE 0 END) as terciario_curso,
        sum(CASE WHEN nombre = 'Terciario/Técnico' AND estado = 'Graduado' THEN 1 ELSE 0 END) as terciario_graduado,
        sum(CASE WHEN nombre = 'Terciario/Técnico' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as terciario_abandonado,
        sum(CASE WHEN nombre = 'Doctorado' AND estado = 'En Curso' THEN 1 ELSE 0 END) as doctorado_curso,
        sum(CASE WHEN nombre = 'Doctorado' AND estado = 'Graduado' THEN 1 ELSE 0 END) as doctorado_graduado,
        sum(CASE WHEN nombre = 'Doctorado' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as doctorado_abandonado,
        sum(CASE WHEN nombre = 'Secundario' AND estado = 'En Curso' THEN 1 ELSE 0 END) as secundario_curso,
        sum(CASE WHEN nombre = 'Secundario' AND estado = 'Graduado' THEN 1 ELSE 0 END) as secundario_graduado,
        sum(CASE WHEN nombre = 'Secundario' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as secundario_abandonado
    FROM
        educacion
    GROUP BY 1;
    '''

    sql_2 = '''
    SELECT
        idpostulante,
        strftime('%Y', date('now')) - strftime('%Y', fechanacimiento) as edad,
        sum(CASE WHEN nombre = 'Secundario' AND estado = 'Abandonado' THEN 1 ELSE 0 END) as genero
    FROM
        genero_edad
    ;
    '''

    #cur = con.cursor()
    #cur.execute(sql_2)
    #rows = cur.fetchall()

    #for row in rows:
    #    print(row)