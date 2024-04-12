import torch
import pandas as pd
from model import GCNLinkPrediction, hit_ratio
import joblib
from data_prep import UserIdMapper, create_torch_data_object
import json
import argparse
import sys

#command line arguments for parameters
parser = argparse.ArgumentParser(description='Train and evaluate a GCN link prediction model.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability (default: 0.3)')
parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension size (default: 100)')
parser.add_argument('--num_layers', type=int, default=1, help='Number of GCN layers (default: 1)')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for regularization (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--counter', type=int, default=7, help='Number of training epochs (default: 7)')
args = parser.parse_args()

# Extract command-line arguments
dropout = args.dropout
hidden_dim = args.hidden_dim
num_layers = args.num_layers
weight_decay = args.weight_decay
learning_rate = args.learning_rate
counter = args.counter

#Create dictionary for hyperparamters to save
hp_dict = {
    'dropout': dropout,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers,
    'weight_decay': weight_decay,
    'learning_rate': learning_rate,
    'counter': counter
}

print(hp_dict,file=sys.stderr)

############ Set user/model parameters ################
########################################################
# User parameters
preprocessing_dir = '../deep_learning/preprocessing'
model_name =f'GCNLinkPrediction{counter}' # Swap out with any other model class/name we try (needs to be in model_dict below)
evaluation_directory = '../deep_learning/evaluation'
index_cols = ['user_id']
model_name_dict = {
    f'GCNLinkPrediction{counter}':GCNLinkPrediction
}
model_path = f'../deep_learning/models/{model_name}.pth'
sample_size_for_hit_ratio=10000
# Note: robust scaler still has a lot of large values after scaling (ex: some are -0.1, some are 557.1). Not sure if this would cause problems with gradient descent, maybe just minmax would be better???
# Had to move all features to minmax to avoid INF training error. Large values ^^ are a problem...
standardization_dict = {
    'standard':['average_stars', 'days_since_yelping'],
    # 'robust':[],
    'minmax':['years_elite', 'years_since_elite', 'review_count', 'useful', 'funny', 'cool', 'fans', 'total_compliments', 'Health_Beauty_Rec', 'Food_Dining', 'Shopping_Retail', 'Home_Services', 'Professional_Services', 'Arts_Entertainment_Party', 'Public_Transportation_Education', 'Other']
}
# Model hyperparameters
device = 'cpu'
#batch_size = 64

################## Gather test data ####################
########################################################
print('Gathering data...',file=sys.stderr)

# Read in NEEDS TO BE CHANGES TO TEST DATA SET 
df_edges = pd.read_parquet('../data/edges_double.parquet')
df_node_features = pd.read_parquet('../data/node_attributes.parquet')
print('Data read.',file=sys.stderr)

# Convert user hash to an ID
user_id_mapper = UserIdMapper(dict_path='../deep_learning/preprocessing/user_to_id.pkl')
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
print(f'Maximum friend count in test set: {max_count.item()}\n',file=sys.stderr)

# Make predictions on entire graph
with torch.no_grad():
    # Forward pass through model
    node_embeddings = model(data)
    #print(node_embeddings.size())

# Calculate hit ratio as a benchmark for each model
# ASSUMES USING DOT PRODUCT FOR EDGE SCORE
hit_ratio = hit_ratio(embeddings=node_embeddings, test_edge_index=test_set_edges, sample_size=sample_size_for_hit_ratio)
print(f'Hit ratio: {hit_ratio}',file=sys.stderr)

# Create summary evaluation file
evaluation = {
    'model':model_name,
    'max_test_set_friend_count':max_count.item(),
    'num_test_set_users':len(unique_values),
    'hit_ratio_sample_size':sample_size_for_hit_ratio,
    'hit_ratio':hit_ratio,
    'Hyperparameters': hp_dict
}

with open(f'{evaluation_directory}/{model_name}_eval.json', "w") as f:
    json.dump(evaluation, f)
