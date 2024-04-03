import torch
import pandas as pd
from torch_geometric.utils import negative_sampling
from model import GCNLinkPrediction, score_edges, compute_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from data_prep import UserIdMapper, create_torch_data_object

############ Set user/model parameters ################
########################################################
# User parameters
preprocessing_dir = './deep_learning/preprocessing'
model_name ='GCNLinkPrediction' # Swap out with any other model class/name we try (needs to be in model_dict below)
output_model_directory = f'./deep_learning/models'
index_cols = ['user_id']
model_name_dict = {
    'GCNLinkPrediction':GCNLinkPrediction
}
# Note: robust scaler still has a lot of large values after scaling (ex: some are -0.1, some are 557.1). Not sure if this would cause problems with gradient descent, maybe just minmax would be better???
# Had to move all features to minmax to avoid INF training error. Large values ^^ are a problem...
standardization_dict = {
    # 'standard':[],
    # 'robust':[],
    'minmax':['average_stars', 'days_since_yelping', 'years_elite', 'years_since_elite', 'review_count', 'useful', 'funny', 'cool', 'fans', 'total_compliments', 'Health_Beauty_Rec', 'Food_Dining', 'Shopping_Retail', 'Home_Services', 'Professional_Services', 'Arts_Entertainment_Party', 'Public_Transportation_Education', 'Other']
}

# Model hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 4
#batch_size = 64
learning_rate = 0.005
#momentum = 0.9

################## Gather data #########################
# Note: Right now, not splitting into train/val/test yet.
# Just want to know that the code works correctly for training.
# Splitting may require using 'train_mask', 'val_mask', etc. that I haven't figured out yet.
########################################################
print('Gathering data...')

# Read in
df_edges = pd.read_parquet('./data/edges_double.parquet')
df_node_features = pd.read_parquet('./data/node_attributes.parquet')
print('Data read.')

# Convert user hash to an ID
unique_user_in_edges = df_edges['source'].unique().sort()
unique_users = df_node_features['user_id'].unique().sort()
assert (unique_user_in_edges == unique_users), "All users part of edges must be in node attribute data frame"

#assert len(list3) == 0, "All users part of edges must be in node attribute data frame"
# List of unique users
unique_users = df_node_features['user_id'].unique()
user_id_mapper = UserIdMapper(unique_users=unique_users)
# Conversion to integer
df_edges['source'] = df_edges['source'].apply(user_id_mapper.transform)
df_edges['target'] = df_edges['target'].apply(user_id_mapper.transform)
df_node_features['user_id'] = df_node_features['user_id'].apply(user_id_mapper.transform)
# Save this dictionary to use in predict.py
user_id_mapper.save_dict('./deep_learning/preprocessing/user_to_id.pkl')

# Split into train/val/test

# Apply standardization to node features
# Note: this will need to change after data splitting
# standard_scaler = StandardScaler()
# df_node_features[standardization_dict['standard']] = standard_scaler.fit_transform(df_node_features[standardization_dict['standard']])

minmax_scaler = MinMaxScaler()
df_node_features[standardization_dict['minmax']] = minmax_scaler.fit_transform(df_node_features[standardization_dict['minmax']])

# robust_scaler = RobustScaler()
# df_node_features[standardization_dict['robust']] = robust_scaler.fit_transform(df_node_features[standardization_dict['robust']])

# Save preprocessing objects to use in predict script
# joblib.dump(standard_scaler, f'{preprocessing_dir}/standard_scaler.pkl')
# joblib.dump(robust_scaler, f'{preprocessing_dir}/robust_scaler.pkl')
joblib.dump(minmax_scaler, f'{preprocessing_dir}/minmax_scaler.pkl')

# Create PTGeo Data object
# This will require sorting the df_node_features by user id integer inside of this function
index, data = create_torch_data_object(df_edges=df_edges, df_node_features=df_node_features, index_cols=index_cols)

# print(index[0:20])
# print(data.x)
# print(data.edge_index)

################## Train model #########################
########################################################
# Instantiate model
model_class = model_name_dict[model_name]
model = model_class(input_dim=data.x.shape[1], output_dim=25).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define a function to perform link prediction training
def train_loop(model, optimizer, data):
    model.train()

    # Forward pass through model
    node_embeddings = model(data)
    
    # Sample positive and negative edges for making predictions
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes) # Generate same number of negative samples as positive samples

    # Compute similarity scores for each pos/neg edge
    pos_pred = score_edges(node_embeddings, pos_edge_index)  # Compute scores for positive edges
    neg_pred = score_edges(node_embeddings, neg_edge_index)  # Compute scores for negative edges

    # Compute BCE loss (average of pos + average of neg)
    loss = compute_loss(pos_pred, neg_pred)  # Compute loss
    print(f'Training loss: {loss}')

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for i in range(epochs):
    print(f'===== Epoch {i} ======================================================================================')
    # Run the train loop
    # data should contain node features (data.x) and edge indices (data.edge_index)
    train_loop(model=model, optimizer=optimizer, data=data)

print(f"Training complete\n")

################## Save model ##########################
########################################################
try:
    torch.save(model, f'{output_model_directory}/{model_name}.pth')
    print('Model successfully saved')
except:
    print(f'Error saving model\n')
    raise


################## NEED TO ADD #########################
########################################################

# A test loop that acts on the validation set (we calculate the loss on validation set at the end of each epoch)
# A final model prediciton on the test set, along with the loss of the test set (calculated only once at the end of the training)???