import torch
import pandas as pd
from model import GCNLinkPrediction, hit_ratio
import joblib
from data_prep import UserIdMapper, create_torch_data_object
import json

############ Set user/model parameters ################
########################################################
# User parameters
preprocessing_dir = './deep_learning/preprocessing'
model_name ='GCNLinkPrediction' # Swap out with any other model class/name we try (needs to be in model_dict below)
evaluation_directory = './deep_learning/evaluation'
index_cols = ['user_id']
model_name_dict = {
    'GCNLinkPrediction':GCNLinkPrediction
}
model_path = f'./deep_learning/models/{model_name}.pth'
# Note: robust scaler still has a lot of large values after scaling (ex: some are -0.1, some are 557.1). Not sure if this would cause problems with gradient descent, maybe just minmax would be better???
# Had to move all features to minmax to avoid INF training error. Large values ^^ are a problem...
standardization_dict = {
    'standard':['average_stars', 'days_since_yelping'],
    # 'robust':[],
    'minmax':['years_elite', 'years_since_elite', 'review_count', 'useful', 'funny', 'cool', 'fans', 'total_compliments', 'Health_Beauty_Rec', 'Food_Dining', 'Shopping_Retail', 'Home_Services', 'Professional_Services', 'Arts_Entertainment_Party', 'Public_Transportation_Education', 'Other']
}
# Model hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#batch_size = 64

################## Gather test data ####################
########################################################
print('Gathering data...')

# Read in NEEDS TO BE CHANGES TO TEST DATA SET 
df_edges = pd.read_parquet('./data/edges_double.parquet')
df_node_features = pd.read_parquet('./data/node_attributes.parquet')
print('Data read.')

# Convert user hash to an ID
user_id_mapper = UserIdMapper(dict_path='./deep_learning/preprocessing/user_to_id.pkl')
df_edges['source'] = df_edges['source'].apply(user_id_mapper.transform)
df_edges['target'] = df_edges['target'].apply(user_id_mapper.transform)
df_node_features['user_id'] = df_node_features['user_id'].apply(user_id_mapper.transform)

# Load scalers from training
standard_scaler = joblib.load(f'{preprocessing_dir}/standard_scaler.pkl')
minmax_scaler = joblib.load(f'{preprocessing_dir}/minmax_scaler.pkl')

# Apply standardization to node features
df_node_features[standardization_dict['standard']] = standard_scaler.transform(df_node_features[standardization_dict['standard']])
df_node_features[standardization_dict['minmax']] = minmax_scaler.transform(df_node_features[standardization_dict['minmax']])

# Create PTGeo Data object
index, data = create_torch_data_object(df_edges=df_edges, df_node_features=df_node_features, index_cols=index_cols)

################## Make predictions ####################
########################################################
# Load model
model = torch.load(model_path)
model.to(device)
model.eval()

# Load test set edges
test_set_edges = torch.load(f=f'{evaluation_directory}/test_edge_list.pt')
# print(test_set_edges.size())
# print(test_set_edges[0,:].unique().size()) # Too many people in test set, will need to take random sample of 1000 (plus the actual edges for that user)
# Find the maximum count
unique_values, counts = test_set_edges[0, :].unique(return_counts=True)
max_count = counts.max()
print(f'Maximum friend count in test set: {max_count.item()}\n')

# Make predictions on entire graph
with torch.no_grad():
    # Forward pass through model
    node_embeddings = model(data)
    #print(node_embeddings.size())

# Calculate hit ratio as a benchmark for each model
# ASSUMES USING DOT PRODUCT FOR EDGE SCORE
hit_ratio = hit_ratio(embeddings=node_embeddings, test_edge_index=test_set_edges)
print(f'Hit ratio: {hit_ratio}')

# Create summary evaluation file
evaluation = {
    'model':model_name,
    'max_test_set_friend_count':max_count.item(),
    'num_test_set_users':len(unique_values),
    'hit_ratio':hit_ratio
}

with open(f'{evaluation_directory}/{model_name}_eval.json', "w") as f:
    json.dump(evaluation, f)
