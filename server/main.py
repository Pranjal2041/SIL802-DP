from typing import Union

from fastapi import FastAPI, HTTPException

import subprocess
import os
import pydp as dp
from pydp.algorithms.laplacian import *
import pandas as pd


app = FastAPI()
all_data = None
# data = None

@app.get("/")
def read_root() -> str:
    return "Welcome to Differential Privacy Server!\nPlease use /query to query the database."

@app.get("/get_columns")
def get_columns(dataset : str) -> list:
    print('dataset in get_columns is',dataset, all_data[dataset].columns)
    return list(all_data[dataset].columns)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# A set of common functions applied on a column
@app.get("/common")
def common(dataset : str, function: str, column: str, epsilon: float = 0.1, delta = 1e-5, lower : Union[str, None] = None, upper : Union[str, None] = None) -> str:
    global all_data
    function_map = [
        "BoundedMean",
        "BoundedStandardDeviation",
        "BoundedSum",
        "BoundedVariance",
        "Count",
        "Max",
        "Min",
        "Median",
        "Percentile",
    ]
    if function not in function_map:
        print('function not in function_map')
        raise HTTPException(status_code=400, detail="Function not supported")
        # return {"error": "Function not supported"}, 400
    if column not in ['Age Bracket', 'Num Cases', 'Box', 'Unit sequence in call dispatch', 'Number of Alarms', 'Final Priority', 'Analysis Neighborhoods']:
        print("Column is not supported")
        raise HTTPException(status_code=400, detail="Column not supported")
    if epsilon <= 0:
        print("Epsilon should be > 0")
        raise HTTPException(status_code=400, detail="Epsilon must be positive")
        # return {"error": "Epsilon must be positive"}, 400
    dp_function = getattr(dp.algorithms.laplacian, function)
    try:
        algo = dp_function(epsilon, lower_bound = lower, upper_bound = upper, dtype = 'float')
    except:
        algo = dp_function(epsilon, dtype = 'float')

    if function == "Percentile":
        return {"result": algo.quick_result(list(all_data[dataset][column]), 0.9)}

    return {"result": algo.quick_result(list(all_data[dataset][column]))}

@app.get("/query")
def make_query(q: Union[str, None] = None, privacy : bool = True, epsilon : float = 0.1, delta : float = 1e-5, dataset : str = 'health'):

    print('Make Query called')

    if q.count('SELECT') != 1:
        # Return error code 400, bad request
        return {"error": "Query must contain exactly one SELECT statement."}, 400

    if privacy:
        # We will modify the query to include epsilon, sigma privacy
        # For this we will use zestsql library
        ANON_STR = 'WITH ANONYMIZATION OPTIONS(epsilon={}, delta={}, kappa={})'
        q = q.replace('SELECT', 'SELECT ' + ANON_STR.format(epsilon, delta, 1))
        
        for opers in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'VARIANCE', 'CORR', 'COVAR', 'COVARIANCE']:
            q = q.replace(opers, 'ANON_' + opers)
    else:
        q = q.replace('CLAMPED BETWEEN 10 AND 50', '').replace('CLAMPED BETWEEN 0 AND 20','')

    # q can contain whitespaces and new lines, sanitize to make it compatabile with subprocess
    q = q.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    print(q) # For debugging
    # result = subprocess.run(['./run_sql.sh', q], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    comm = './run_sql.sh ' + q + ' ' + 'covid_all_data.csv' if dataset == 'health' else 'fire_data.csv'
    comm += ' ' + '"State Patient Number"' if dataset == 'health' else "'Call Number'"
    print('See this', comm)
    ret_code = os.system(comm)
    text = open('sql_output.txt').read()
    if ret_code != 0:
        raise HTTPException(status_code=500, detail="Error in running query" + text)
        # return {"error": "Query failed: "+ text}, 500
    print(text)
    print('Return Code: ', ret_code)

    return {"output": text, 'query_run': q} 

def load_data():
    global all_data
    all_data = {
        'health': pd.read_csv('covid_all_data.csv'),
        'pums': pd.read_csv('titanic_clean.csv'),
        'fire': pd.read_csv('fire_data.csv')
    }
    print('Data Loaded')

def init():
    load_data()

init()