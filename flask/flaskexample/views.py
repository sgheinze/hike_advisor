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
    return render_template("input.html",
    hike_names = hike_names,
    tag_names = tag_names)

@app.route('/output', methods=['POST', 'GET'])
def output():
    selected_hikes_text = request.form['hikes']
    selected_hikes_list = selected_hikes_text.split('||')[:-1]
    selected_hikes = []
    for hike in selected_hikes_list:
         s1 = re.sub('^,', '', hike)
         s2 = re.sub('$,', '', s1)
         selected_hikes.append(s2)

    tags = [ tag.lower() for tag in request.form['tags'].split(',') ]
    hike_difficulty = request.form['hikedifficulty']
    hike_distance = request.form['hikedistance']
    hike_elevation = request.form['hikeelevation']
    hike_type = request.form['hiketype']

    hike_attributes = rec.get_attributes(tags=tags, hike_difficulty=hike_difficulty, hike_distance=hike_distance,
                      hike_elevation=hike_elevation, hike_type=hike_type)
    selected_item_features_embeddings= rec.get_item_feature_embeddings(hike_attributes, item_features_map, 
                                       item_features_embeddings)
    selected_hike_embeddings = rec.get_item_embeddings(selected_hikes=selected_hikes, item_map=item_map, 
                               item_embeddings=item_embeddings)
    av_hike_embedding = rec.get_average_item_embedding(selected_item_embeddings=selected_hike_embeddings)
    final_embedding = rec.get_final_embedding(av_item_embedding=av_hike_embedding, 
                      selected_item_features_embeddings=selected_item_features_embeddings)
    sim_scores = rec.calculate_similarities(model, final_embedding, item_embeddings)
    top_hikes = rec.get_top_hikes_by_similarity(sim_scores, item_map, num_hikes=10)

    top_hikes_dict = {}
    for hike in top_hikes:
        temp_dict = {}
        temp_dict['hike_url'] = hike_urls_locs.loc[hike, 'hike_url']
        temp_dict['location'] = hike_urls_locs.loc[hike, 'location']
        temp_dict['pic_url'] = hike_urls_locs.loc[hike, 'hike_pic_url']
        top_hikes_dict[hike] = temp_dict
  
    matched_tags_for_hikes = {}
    for hike in top_hikes:
        hike_features = rec.get_hike_features(hike, feature_matrix)
        convereted_hike_features = rec.convert_features(hike_features)
        matched_tags = rec.get_matched_features(convereted_hike_features, hike_difficulty=hike_difficulty, 
                                                hike_distance=hike_distance, hike_elevation=hike_elevation, 
                                                hike_type=hike_type, tags=[ tag.capitalize() for tag in tags])
        matched_tags_for_hikes[hike] = matched_tags
     
    return render_template("output.html", top_hikes_dict=top_hikes_dict, hike_difficulty = hike_difficulty,
           hike_distance = hike_distance, hike_elevation = hike_elevation, hike_type = hike_type,
           tags = tags, matched_tags_for_hikes = matched_tags_for_hikes)
