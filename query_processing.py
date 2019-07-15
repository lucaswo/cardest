#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle
import time
from argparse import ArgumentParser

import psycopg2 as postgres

import numpy as np

import pandas as pd

def generate_queries(cur, n_queries, min_max, encoders):
    SQL_set = set()
    SQL0_set = set()
    SQL = []
    cardinalities = []
    sql_body = """SELECT count(*) FROM tmpview WHERE {}"""

    total_columns = len(min_max)
    vectors = np.ndarray((n_queries, total_columns*4))
    columns = list(min_max.keys())
    while len(SQL) < n_queries:
        num_of_predictates = np.random.choice(range(1,total_columns+1))
        selected_predicates = np.random.choice(range(total_columns), size=num_of_predictates, replace=False)
        selected_predicates = [columns[i] for i in selected_predicates]
        
        selected_values = []
        for pred in selected_predicates:
            if pred in encoders.keys():
                sel = np.random.randint(len(encoders[pred].classes_))
                sel = encoders[pred].classes_[sel]
            else:
                choices = [-1] + list(np.arange(min_max[pred][0], min_max[pred][1]+1, min_max[pred][2]))
                sel = np.random.choice(choices)
            selected_values.append(sel)
        
        #[[0,0,1], [0,1,0], [1,0,0], [1,0,1], [0,1,1], [1,1,0]]
        # <>=
        #selected_operators = np.random.choice(["=", ">", "<", "<=", ">=", "!="], size=num_of_predictates)
        selected_operators = np.random.choice(["=", ">", "<"], size=num_of_predictates)
        #selected_operators = ["=" if "id" not in sp else np.random.choice(["=", ">", "<", "<=", ">=", "!="]) 
        #                      for sp in selected_predicates]
        #selected_operators = [np.random.choice(["IS", "IS NOT"]) if selected_values[i] == "-1" 
        #                      or selected_values == -1 else x for i,x in enumerate(selected_operators)]
        selected_operators = ["IS" if selected_values[i] == "-1" or selected_values == -1 else x 
                              for i,x in enumerate(selected_operators)]
        
        sql = sql_body.format(" AND ".join([" ".join([str(p), str(o), str(v) if not isinstance(v,str) or v == "-1"
                                                      else "'{}'".format(v)]) for p,o,v in zip(selected_predicates,
                                                                                               selected_operators, 
                                                                                               selected_values)]))
        check_len = len(SQL_set)
        sql = sql.replace("-1", "NULL")
        SQL_set.add(sql)
        if check_len != len(SQL_set) and sql not in SQL0_set:
            cur.execute(sql)
            card = cur.fetchone()[0]
            
            if card > 0:
                SQL.append(sql)
                cardinalities.append(card)
                vectors[len(SQL)-1] = vectorize_query(sql, min_max, encoders)
            else:
                SQL0_set.add(sql)
                SQL_set.remove(sql)

    return SQL, vectors, cardinalities

def vectorize_query(query_str, min_max, encoders):
    query_str = query_str.replace("NULL", "-1").replace("IS NOT", "!=")
    total_columns = len(min_max)
    vector = np.zeros(total_columns*4)
    predicates = query_str.split("WHERE", maxsplit=1)[1]
    operators = {
        "=": [0,0,1],
        ">": [0,1,0],
        "<": [1,0,0],
        "<=": [1,0,1],
        ">=": [0,1,1],
        "!=": [1,1,0],
        "IS": [0,0,1]
    }
    
    for exp in predicates.split("AND"):
        exp = exp.strip()
        pred, op, value = exp.split(" ")
        if pred in encoders.keys():
            value = encoders[pred].transform([value.replace("'", "")])[0]
        else:
            value = max(min_max[pred][0], float(value))
        idx = list(min_max.keys()).index(pred)
        vector[idx*4:idx*4+3] = operators[op]
        vector[idx*4+3] = (value-min_max[pred][0]+min_max[pred][2])/(min_max[pred][1]-min_max[pred][0]+min_max[pred][2])
    
    return vector

if __name__ == '__main__':
    parser = ArgumentParser(description='Query processing for local models')
    parser.add_argument("-v", "--vectorize", type=str, help="just vectorize queries without sampling them", default=0)
    args = parser.parse_args()

    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    with open("min_max_{}.json".format(config["dbname"]), "r") as mm_file, \
    open("encoders_{}.pkl".format(config["dbname"]), "rb") as enc_file:
        minmax = json.load(mm_file)
        encoder = pickle.load(enc_file)

    if args.vectorize:
        queries = pd.read_csv(config["query_file"])
        vectors = np.ndarray((len(queries), len(minmax)*4))

        start = time.time()
        i = 0
        for q in queries["SQL"]:
            vectors[i] = vectorize_query(q, minmax, encoder)
            i += 1
        end = time.time() - start

        print("Vectorized {} queries in {:.2f}s.".format(len(queries), end))

        vectors = np.column_stack([vectors, queries["cardinality"]])
        np.save(config["vector_file"], vectors)
    else:
        # change login data accordingly
        conn = postgres.connect("dbname=imdb user=postgres password=postgres")
        conn.set_session(autocommit=True)
        cur = conn.cursor()

        start = time.time()
        queries, vectors, card = generate_queries(cur, config["number_of_queries"], minmax, encoder)
        end = time.time() - start

        print("Sampled {} queries in {:.2f}s.".format(len(queries), end))

        vectors = np.column_stack([vectors, card])
        np.save(config["vector_file"], vectors)

        csv = pd.DataFrame({"SQL": queries, "cardinality": card})
        csv["cardinality"] = csv["cardinality"].astype(int)
        csv.to_csv(config["query_file"], index=False)

        conn.close()
