import numpy as np
from numpy import dot
from numpy.linalg import norm
from lightfm import data
import re

def get_attributes(tags=None, hike_difficulty=None, hike_distance=None, hike_elevation=None, hike_type=None):
    '''Generates a list of hike attributes/tags the user has selected.
    
    Returns: list of individual hike attributes/tags.'''
    
    hike_attributes = []
    
    if tags != ['']:
        for tag in tags:
            hike_attributes.append(tag)
    
    if hike_difficulty == 'Easy':
        hike_attributes.append('difficulty_easy')
    elif hike_difficulty == 'Moderate':
        hike_attributes.append('difficulty_moderate')
    elif hike_difficulty == 'Hard':
        hike_attributes.append('difficulty_hard') 
        
    if hike_distance == 'Short':
        hike_attributes.append('distance_short')
    elif hike_difficulty == 'Medium':
        hike_attributes.append('distance_medium')
    elif hike_difficulty == 'Long':
        hike_attributes.append('distance_long')
        
    if hike_elevation == 'Easy':
        hike_attributes.append('elevation_easy')
    elif hike_elevation == 'Medium':
        hike_attributes.append('elevation_medium')
    elif hike_elevation == 'Hard':
        hike_attributes.append('elevation_hard')
        
    if hike_type == 'Out-and-Back':
        hike_attributes.append('out_and_back')
    elif hike_type == 'Point-to-Point':
        hike_attributes.append('point_to_point')
    elif hike_type == 'Loop':
        hike_attributes.append('loop')
        
    return hike_attributes

def get_item_feature_embeddings(hike_attributes, item_features_map, item_features_embeddings):
    '''Generate embeddings for selected hike attributes (item features).
    
    Returns: list of embeddings for each attribute (item feature).'''
    
    selected_item_features_embeddings = []
    for hike_attribute in hike_attributes:
        selected_item_features_embeddings.append(item_features_embeddings[item_features_map[hike_attribute]])
    
    return selected_item_features_embeddings

def get_item_embeddings(selected_hikes, item_map, item_embeddings):
    '''Generate embeddings for selected hikes (items)
    
    Returns: list of embeddings for each hike (item)'''
    
    selected_hike_embeddings = []
    for selected_hike in selected_hikes:
        selected_hike_embeddings.append(item_embeddings[item_map[selected_hike]])
        
    return selected_hike_embeddings

def get_average_item_embedding(selected_item_embeddings):
    '''Calculate the average embedding over all selected hikes (items).
    
    Returns: np.array of average embedding.'''

    av_item_embedding = np.zeros((10,))
    for item_embedding in selected_item_embeddings:
        av_item_embedding += item_embedding
    try:
        av_item_embedding = av_item_embedding / len(selected_item_embeddings)
        return av_item_embedding
    except:
        av_item_embedding = 0
        return av_item_embedding

def get_final_embedding(av_item_embedding, selected_item_features_embeddings):
    '''Add all selected item feature embeddings to the average item embedding to produce the final embedding.
    
    Returns: np.array of final embedding'''
    
    final_embedding = av_item_embedding.copy()
    for item_feature_embedding in selected_item_features_embeddings:
        final_embedding += item_feature_embedding
        
    return final_embedding

def find_item(item_map, item_index):
    '''Get hike (item) name given a LightFM index.
    
    Returns: hike name (str)'''
    
    return [ key for key in item_map.keys() if item_map[key] == item_index ][0]

def cos_sim(a, b):
    '''Calculate the cosine similarity between two vectors.
    
    Returns: float'''
    
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def calculate_similarities(model, embedding, item_embeddings):
    '''Calculate the similarity scores for a given embedding and all item embeddings.
    
    Returns: list of similarities, in order of LightFM index'''
    
    sim_scores = []
    for item_index in range(item_embeddings.shape[0]):
        sim_scores.append(cos_sim(item_embeddings[item_index], embedding))
        
    return sim_scores

def get_top_hikes_by_similarity(sim_scores, item_map, num_hikes):
    '''Find the top num_hikes hikes given a list of similarity scores.
    
    Returns: list of hike names.'''
    
    top_items = (-1*np.array(sim_scores)).argsort()
    names = []
    for pred_num in range(num_hikes):
        name = find_item(item_map, top_items[pred_num])
        names.append(name)
    
    return names

def get_hike_features(hike_name, hikes_features_df):
    '''Get a list of item features for a hike
        
    Returns: list of hike_features'''
    hike_features = []
    for feature in hikes_features_df.columns:
        if hikes_features_df.loc[hike_name, feature] == 1.0:
            hike_features.append(feature)
            
    return hike_features

def convert_features(hike_features):
    '''Converts selected hike attributes into the names used in the feature_matrix for LightFM training.
       This allows the features to be used for prediction in the LightFM recommendation system.

    Returns: a dict of selected features with key, value pairs corresponding to the attributes, values.'''

    converted_features = {}
    converted_tags = []
    for feature in hike_features:
        if feature.startswith('difficulty'):
            converted_features['Difficulty'] = feature.split('_')[1].capitalize()
        elif feature.startswith('distance'):
            converted_features['Distance'] = feature.split('_')[1].capitalize()
        elif feature.startswith('elevation'):
            converted_features['Elevation'] = feature.split('_')[1].capitalize()
        elif feature == 'out_and_back':
            converted_features['Type'] = 'Out-and-Back'
        elif feature == 'point_to_point':
            converted_features['Type'] = 'Point-to-Point'
        elif feature == 'loop':
            converted_features['Type'] = 'Loop'
        else:
            converted_tags.append(feature.capitalize())
    if converted_tags:
        converted_features['tags'] = converted_tags
        
    return converted_features

def parse_selected_hikes(selected_hikes_text):
    '''When a user selects hikes on the webapp, the hikes are returned as a single string. This function
       parses the string into a list of hikes (i.e. multiple strings where each string is a hike name)

      Returns: list of selected hikes'''

    selected_hikes_list = selected_hikes_text.split('||')[:-1]
    selected_hikes = []
    for hike in selected_hikes_list:
         s1 = re.sub('^,', '', hike)
         s2 = re.sub('$,', '', s1)
         selected_hikes.append(s2)

    return selected_hikes

def get_matched_features(converted_features, hike_difficulty, hike_distance, hike_elevation, hike_type, tags):
    '''Given a list of recommended hikes, finds the features which overlap with the user's selection.

    Returns: nested dict with key, value pairs where the keys are recommended hike names and the values
             are a dict with key, value pairs corresponding to hike attributes which overlap with the user's
             selection on the webapp.'''

    matched_features = {}
    matched_tags = []
    if converted_features['Difficulty'] == hike_difficulty:
        matched_features['Difficulty'] = hike_difficulty
    if converted_features['Distance'] == hike_distance:
        matched_features['Distance'] = hike_distance
    if converted_features['Elevation'] == hike_elevation:
        matched_features['Elevation'] = hike_elevation
    if converted_features['Type'] == hike_type:
        matched_features['Type'] = hike_type
    if converted_features['tags']:
        for tag in converted_features['tags']:
            if tag in tags:
                matched_tags.append(tag)
    if matched_tags:
        matched_features['Tags'] = matched_tags
    
    return matched_features

def get_top_hikes(selected_hikes, hike_difficulty, hike_distance, hike_elevation, hike_type, tags, item_embeddings, item_map,
                  item_features_embeddings, item_features_map, model):
    '''Get the top 10 recommendations for a user based on their selections on the HikeAdvisor webapp.

    Returns: A list of the top 10 recommendations'''

    hike_attributes = get_attributes(tags=tags, hike_difficulty=hike_difficulty, hike_distance=hike_distance,
                      hike_elevation=hike_elevation, hike_type=hike_type)
    selected_item_features_embeddings= get_item_feature_embeddings(hike_attributes, item_features_map,
                                       item_features_embeddings)
    selected_hike_embeddings = get_item_embeddings(selected_hikes=selected_hikes, item_map=item_map,
                               item_embeddings=item_embeddings)
    av_hike_embedding = get_average_item_embedding(selected_item_embeddings=selected_hike_embeddings)
    final_embedding = get_final_embedding(av_item_embedding=av_hike_embedding,
                      selected_item_features_embeddings=selected_item_features_embeddings)
    sim_scores = calculate_similarities(model, final_embedding, item_embeddings)
    top_hikes = get_top_hikes_by_similarity(sim_scores, item_map, num_hikes=10)

    return top_hikes

def get_top_hike_urls(top_hikes, hike_urls_locs):
    '''Retrieve the urls and locations for the top 10 recommendations.
    
    Returns: a nested dictionary where the keys are hike_name and the values are a dictionary
             with keys hike_url, location, and pic_url.'''

    top_hikes_dict = {}
    for hike in top_hikes:
        temp_dict = {}
        try:
           temp_dict['hike_url'] = hike_urls_locs.loc[hike, 'hike_url']
           temp_dict['location'] = hike_urls_locs.loc[hike, 'location']
           temp_dict['pic_url'] = hike_urls_locs.loc[hike, 'hike_pic_url']
           top_hikes_dict[hike] = temp_dict
        except:
           temp_dict['hike_url'] = None
           temp_dict['location'] = None
           temp_dict['pic_url'] = None
           top_hikes_dict[hike] = temp_dict

    return top_hikes_dict

def get_matched_tags(top_hikes, hike_difficulty, hike_distance, hike_elevation, hike_type, tags, feature_matrix):
    '''Given that a user selects specific hike attributes (distance, elevation, tags, etc.), computes
    the list of matched attributes for the top 10 recommended hikes.

    Returns: a nested dictionary where the keys are hike_name and the values are a dictionary
             with keys hike_distance, hike_elevation, etc. and the values are the values of those
             attributes.'''

    matched_tags_for_hikes = {}
    for hike in top_hikes:
        hike_features = get_hike_features(hike, feature_matrix)
        convereted_hike_features = convert_features(hike_features)
        matched_tags = get_matched_features(convereted_hike_features, hike_difficulty=hike_difficulty,
                                            hike_distance=hike_distance, hike_elevation=hike_elevation,
                                            hike_type=hike_type, tags=[ tag.capitalize() for tag in tags])
        matched_tags_for_hikes[hike] = matched_tags

    return matched_tags_for_hikes
