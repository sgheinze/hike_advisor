import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix
from scipy import sparse
from lightfm import data
from lightfm import cross_validation
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from numpy import dot
from numpy.linalg import norm

def lightfm_implicit_matrix(df):
    dataset = data.Dataset() # Instantiate class
    dataset.fit((user for user in df.index),
                (feature for feature in df.columns)) # Create users and items from df
    num_users, num_items = dataset.interactions_shape() # Get shape
    interaction_list = list(df[df > 0].stack().index) # Get interaction pairs
    interactions, weights = dataset.build_interactions((x[0], x[1]) for x in interaction_list) # Build interactions
    
    return dataset, interactions

def lightfm_train(train, num_components, num_epochs):
    NUM_THREADS = 1
    NUM_COMPONENTS = num_components
    NUM_EPOCHS = num_epochs
    ITEM_ALPHA = 1e-6

    # Let's fit a WARP model: these generally have the best performance.
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                   no_components=NUM_COMPONENTS)

    # Run 3 epochs and time it.
    model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    
    return model

def find_item(lfm_map, lfm_index):
    return [ key for key in lfm_map.keys() if lfm_map[key] == lfm_index ][0]

def return_top_10_new_user(interactions, model, item_map):
    '''This is for recommending based on interactions'''
    predictions = model.predict(interactions._shape[0]-1, np.arange(interactions._shape[1]))
    sorted_by_index = (-1*predictions).argsort()
    names = []
    
    for pred_num in range(10):
        name = find_item(item_map, sorted_by_index[pred_num])
        print('Prediction {} is {}.'.format(pred_num+1, name))
        names.append(name)
        
    return names
    
def cos_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def calculate_similarities(model, hike_index_of_interest):
    item_embeddings = model.get_item_representations()[1]
    sim_scores = []
    for item_index in range(item_embeddings.shape[0]):
        sim_scores.append(cos_sim(item_embeddings[item_index, :], item_embeddings[hike_index_of_interest, :]))
        
    return sim_scores

def get_top_hikes_by_similarity(sim_scores, item_map):
    '''This is for recommending based on item similarities'''
    top_items = (-1*np.array(sim_scores)).argsort()
    names = []
    for pred_num in range(10):
        name = find_item(item_map, top_items[pred_num])
        names.append(name)
    
    return names
   
   