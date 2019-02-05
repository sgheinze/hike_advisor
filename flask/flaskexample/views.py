from flask import render_template
from flask import request
from flaskexample import app
import pandas as pd
import numpy as np
import recommendation as rec
import pickle
from lightfm import data
import re

# Get list of dictionaries containing all hike names
columns = sorted(list(pd.read_pickle('flaskexample/Data/column_names_190129')))
hike_names = []
for hike_name in columns:
    hike_names.append({'name': hike_name})


hike_urls_locs = pd.read_pickle('flaskexample/Data/hike_url&loc_df.pickle')

# Get list of dictionaries containing all feature names
feature_matrix = pd.read_pickle('flaskexample/Data/hike_features_190129b')
drop_columns = ['difficulty_easy', 'difficulty_hard', 'difficulty_moderate',
               'distance_short', 'distance_medium', 'distance_long',
               'elevation_easy', 'elevation_medium', 'elevation_hard',
               'out_and_back', 'point_to_point', 'loop']
tag_names = []
for tag in feature_matrix.drop(labels=drop_columns, axis=1).columns:
    tag_names.append({'name': tag.capitalize()})

# Read in dataset and model 

with open('flaskexample/Data/lightfm_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
with open('flaskexample/Data/lightfm_model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('flaskexample/Data/lightfm_featurematrix.pickle', 'rb') as f:
    lfm_feature_matrix = pickle.load(f)

item_features_map = dataset.mapping()[3] # get mapping for item features to lightfm indices
item_features_embeddings = model.item_embeddings # get item feature embeddings
item_map = dataset.mapping()[2]
item_embeddings = model.get_item_representations(features=lfm_feature_matrix)[1]

@app.route('/', methods=['POST', 'GET'])
@app.route('/input', methods=['POST', 'GET'])
def index():
    return render_template("input.html", hike_names = hike_names, tag_names = tag_names)

@app.route('/output', methods=['POST', 'GET'])
def output():
    
    # Get input from user
    selected_hikes_text = request.form['hikes']
    selected_hikes = rec.parse_selected_hikes(selected_hikes_text)
    hike_difficulty = request.form['hikedifficulty']
    hike_distance = request.form['hikedistance']
    hike_elevation = request.form['hikeelevation']
    hike_type = request.form['hiketype']
    tags = [ tag.lower() for tag in request.form['tags'].split(',') ]

    # Get hike recommendations
    top_hikes = rec.get_top_hikes(selected_hikes=selected_hikes, hike_difficulty=hike_difficulty, hike_distance=hike_distance, 
                                  hike_elevation=hike_elevation, hike_type=hike_type, tags=tags, item_embeddings=item_embeddings,
                                  item_map=item_map, item_features_embeddings=item_features_embeddings, 
                                  item_features_map=item_features_map, model=model)

    # Get hike recommendations AllTrails.com urls, urls for corresponding pictures, and hike locations.
    top_hikes_dict = rec.get_top_hike_urls(top_hikes=top_hikes, hike_urls_locs=hike_urls_locs)

    # Get matched tags for recommendations based on inputed attributes on webapp.
    matched_tags_for_hikes = rec.get_matched_tags(top_hikes=top_hikes, hike_difficulty=hike_difficulty, 
                                                  hike_distance=hike_distance, hike_elevation=hike_elevation,
                                                  hike_type=hike_type, tags=tags, feature_matrix=feature_matrix)
  
    return render_template("output.html", top_hikes_dict=top_hikes_dict, hike_difficulty = hike_difficulty,
           hike_distance = hike_distance, hike_elevation = hike_elevation, hike_type = hike_type,
           tags = tags, matched_tags_for_hikes = matched_tags_for_hikes)
