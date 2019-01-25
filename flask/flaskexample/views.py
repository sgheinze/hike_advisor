from flask import render_template
from flaskexample import app
from flask import request
from flaskexample.a_Model import ModelIt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import numpy as np
import psycopg2
import recommendation
from lightfm import data
from lightfm import cross_validation

columns = sorted(list(pd.read_pickle('column_names')))
hike_names = []
for hike_name in columns:
    hike_names.append({'name': hike_name})

data = pd.read_pickle('hike_data_filtered_190124')
dataset, interactions = recommendation.lightfm_implicit_matrix(data)
train, test = cross_validation.random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(seed=1))
model = recommendation.lightfm_train(train, 30, 30)

item_map = dataset.mapping()[2]

@app.route('/', methods=['POST', 'GET'])
@app.route('/input', methods=['POST', 'GET'])
def index():
    return render_template("input.html",
    hike_names = hike_names)

@app.route('/output', methods=['POST', 'GET'])
def output():
   selected_hikes = request.form['hikes']
   hike_index = item_map[selected_hikes]
   sim_scores = recommendation.calculate_similarities(model, hike_index)
   top_hikes = recommendation.get_top_hikes_by_similarity(sim_scores, item_map)
   return render_template("output.html", result=top_hikes)

