#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle

import psycopg2 as postgres

from sklearn.preprocessing import LabelEncoder

def setup_view(cur, table_names, columns, join_atts=None):
    sql = """DROP TABLE IF EXISTS tmpview;"""
    print("Cleaning previous context...")
    #cur.execute(sql)
    
    sql = """SELECT column_name, data_type FROM information_schema.columns 
             WHERE table_schema = 'public' AND table_name IN ('{}') 
             AND column_name IN ('{}');""".format("','".join(table_names), "','".join(columns))
    cur.execute(sql)
    columns_types = cur.fetchall()
    
    if len(table_names) > 1:
        sql = """CREATE TABLE tmpview AS (SELECT {} from {} WHERE {});""".format(
              ",".join(["coalesce({col},'-1') AS {col}".format(col=col[0]) if "character" in col[1] 
                        else "{col}".format(col=col[0]) for col in columns_types]), 
              ",".join(["{} t{}".format(tab,i+1) for i,tab in enumerate(table_names)]),
              " AND ".join(["t{}.{} = t{}.{}".format(1, join[0], i+2, join[1]) 
                            for i,join in enumerate(join_atts)]))
        print("Setting up new context...")
        #cur.execute(sql)
        
    #sql = """SELECT count(*) FROM tmpview;"""
    #cur.execute(sql)
    #N = cur.fetchall()[0]
    
    return columns_types

def gather_meta(cur, columns):
    # columns with type
    min_max = {}
    encoders = {}
    
    print("Gather column information...")
    for col in columns:
        if "character" in col[1]:
            sql = """SELECT {col}, count(*) from tmpview GROUP BY {col};""".format(col=col[0])

            cur.execute(sql)
            tmp = cur.fetchall()

            attr = [x[0] for x in tmp]
        
            le = LabelEncoder()
            cats = [x.item() for x in le.fit_transform(sorted(attr))]

            encoders[col[0]] = le
            step = 1
        else:
            sql = """SELECT min({col}), max({col}) from tmpview;""".format(col=col[0])
            cur.execute(sql)
            cats = cur.fetchall()[0]
            if col[1] == "integer":
                step = 1
            else:
                step = 1/1000
        
        min_max[col[0]] = (min(cats), max(cats), step)
        
    return min_max, encoders

if __name__ == '__main__':
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # change login data accordingly
    conn = postgres.connect("dbname=imdb user=postgres password=postgres")
    conn.set_session(autocommit=True)
    cur = conn.cursor()

    cols = setup_view(cur, config["tables"], config["columns"], config["join_ids"])

    minmax, encoder = gather_meta(cur, cols)

    with open("min_max_{}.json".format(config["dbname"]), "w") as mm_file, \
    open("encoders_{}.pkl".format(config["dbname"]), "wb") as enc_file:
        json.dump(minmax, mm_file)
        pickle.dump(encoder, enc_file)

    conn.close()
