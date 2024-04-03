import torch
import pandas as pd
from torch_geometric.utils import negative_sampling
from model import GCNLinkPrediction, score_edges, compute_loss, get_graph_recommendations
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from data_prep import UserIdMapper, create_torch_data_object
import sys

############ Set user/model parameters ################
########################################################
# User parameters
preprocessing_dir = './deep_learning/preprocessing'
model_name ='GCNLinkPrediction' # Swap out with any other model class/name we try (needs to be in model_dict below)
output_model_directory = './deep_learning/models'
index_cols = ['user_id']
model_name_dict = {
    'GCNLinkPrediction':GCNLinkPrediction
}
model_path = f'./deep_learning/models/{model_name}.pth'
# Note: robust scaler still has a lot of large values after scaling (ex: some are -0.1, some are 557.1). Not sure if this would cause problems with gradient descent, maybe just minmax would be better???
# Had to move all features to minmax to avoid INF training error. Large values ^^ are a problem...
standardization_dict = {
    # 'standard':[],
    # 'robust':[],
    'minmax':['average_stars', 'days_since_yelping', 'years_elite', 'years_since_elite', 'review_count', 'useful', 'funny', 'cool', 'fans', 'total_compliments', 'Health_Beauty_Rec', 'Food_Dining', 'Shopping_Retail', 'Home_Services', 'Professional_Services', 'Arts_Entertainment_Party', 'Public_Transportation_Education', 'Other']
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

# Apply standardization to node features
minmax_scaler = MinMaxScaler()
df_node_features[standardization_dict['minmax']] = minmax_scaler.fit_transform(df_node_features[standardization_dict['minmax']])

# Create PTGeo Data object
index, data = create_torch_data_object(df_edges=df_edges, df_node_features=df_node_features, index_cols=index_cols)

################## Make predictions ####################
########################################################
# Load model
model = torch.load(model_path)
model.to(device)
model.eval()

# Make predictions on entire graph
with torch.no_grad():
    # Forward pass through model
    node_embeddings = model(data)

# Get matrix of similarity scores between all pairs of nodes
# ASSUMES USING DOT PRODUCT FOR EDGE SCORE
graph_recommendations = get_graph_recommendations(embeddings=node_embeddings, edge_index=data.edge_index)

print(graph_recommendations)
# process killed, may be a memory problem with such a large matrix multiplication??

# Gather the top 5 recommendations for each user (highest scores in each row/column)



