import torch
import pandas as pd
from torch_geometric.utils import negative_sampling
from model import GCNLinkPrediction, score_edges, compute_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from data_prep import UserIdMapper, create_torch_data_object, split_edges
import sys
import argparse

#command line arguments for parameters
parser = argparse.ArgumentParser(description='Train and evaluate a GCN link prediction model.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability (default: 0.3)')
parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension size (default: 100)')
parser.add_argument('--num_layers', type=int, default=1, help='Number of GCN layers (default: 1)')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for regularization (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--counter', type=int, default=1, help='Number of training epochs (default: 1)')
args = parser.parse_args()

# Extract command-line arguments
dropout = args.dropout
hidden_dim = args.hidden_dim
num_layers = args.num_layers
weight_decay = args.weight_decay
learning_rate = args.learning_rate
counter = args.counter

############ Set user/model parameters ################
########################################################
# User parameters
preprocessing_dir = '../deep_learning/preprocessing'
model_name =f'GCNLinkPrediction{counter}' # Swap out with any other model class/name we try (needs to be in model_dict below)
output_model_directory = '../deep_learning/models'
evaluation_dir = '../deep_learning/evaluation'
index_cols = ['user_id']
model_name_dict = {
    f'GCNLinkPrediction{counter}':GCNLinkPrediction
}
# Note: robust scaler still has a lot of large values after scaling (ex: some are -0.1, some are 557.1). Not sure if this would cause problems with gradient descent, maybe just minmax would be better???
standardization_dict = {
    'standard':['average_stars', 'days_since_yelping'],
    # 'robust':[],
    'minmax':['years_elite', 'years_since_elite', 'review_count', 'useful', 'funny', 'cool', 'fans', 'total_compliments', 'Health_Beauty_Rec', 'Food_Dining', 'Shopping_Retail', 'Home_Services', 'Professional_Services', 'Arts_Entertainment_Party', 'Public_Transportation_Education', 'Other']
}

# Model hyperparameters
device = "cpu"
epochs = 7
#batch_size = 64
learning_rate = learning_rate
#momentum = 0.9

#Empty cache prior to training model
#torch.cuda.empty_cache()

################## Gather data #########################
# Note: Right now, not splitting into train/val/test yet.
# Just want to know that the code works correctly for training.
# Splitting may require using 'train_mask', 'val_mask', etc. that I haven't figured out yet.
########################################################
print('Gathering data...',file=sys.stderr)

# Read in
df_edges = pd.read_parquet('../data/edges_double.parquet')
df_node_features = pd.read_parquet('../data/node_attributes.parquet')
print('Data read.',file=sys.stderr)

# Convert user hash to an ID
unique_user_in_edges = df_edges['source'].unique().sort()
unique_users = df_node_features['user_id'].unique().sort()
assert (unique_user_in_edges == unique_users), "All users part of edges must be in node attribute data frame"

# List of unique users
unique_users = df_node_features['user_id'].unique()
user_id_mapper = UserIdMapper(unique_users=unique_users)
# Conversion to integer
df_edges['source'] = df_edges['source'].apply(user_id_mapper.transform)
df_edges['target'] = df_edges['target'].apply(user_id_mapper.transform)
df_node_features['user_id'] = df_node_features['user_id'].apply(user_id_mapper.transform)
# Save this dictionary to use in predict.py
user_id_mapper.save_dict('../deep_learning/preprocessing/user_to_id.pkl')

# Apply standardization to node features
# Note: Does this need to change after data splitting?? We are masking edges not nodes...
standard_scaler = StandardScaler()
df_node_features[standardization_dict['standard']] = standard_scaler.fit_transform(df_node_features[standardization_dict['standard']])

minmax_scaler = MinMaxScaler()
df_node_features[standardization_dict['minmax']] = minmax_scaler.fit_transform(df_node_features[standardization_dict['minmax']])

# robust_scaler = RobustScaler()
# df_node_features[standardization_dict['robust']] = robust_scaler.fit_transform(df_node_features[standardization_dict['robust']])

# Save preprocessing objects to use in predict script
joblib.dump(standard_scaler, f'{preprocessing_dir}/standard_scaler.pkl')
# joblib.dump(robust_scaler, f'{preprocessing_dir}/robust_scaler.pkl')
joblib.dump(minmax_scaler, f'{preprocessing_dir}/minmax_scaler.pkl')

# Create PTGeo Data object
# This will require sorting the df_node_features by user id integer inside of this function
index, data = create_torch_data_object(df_edges=df_edges, df_node_features=df_node_features, index_cols=index_cols)
data = data.to(device)

# Split edge indices into train and test sets
train_edge_index, val_edge_index, test_edge_index = split_edges(data.edge_index)
data.train_edge_index = train_edge_index
data.val_edge_index = val_edge_index
data.test_edge_index=test_edge_index

# print(index[0:20])
# print(data.x)
# print(data.edge_index)

# Save the test set so that we can calculate hit ratio later in predict.py
torch.save(data.test_edge_index, f=f'{evaluation_dir}/test_edge_list.pt')

print(data.train_edge_index.size(),file=sys.stderr)
print(data.val_edge_index.size(),file=sys.stderr)
print(data.test_edge_index.size(),file=sys.stderr)

################## Train model #########################
########################################################
# Instantiate model
model_class = model_name_dict[model_name]
model = model_class(input_dim=data.x.shape[1], output_dim=25, num_layers=num_layers,hidden_dim=hidden_dim, p_dropout = dropout).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

print(f"dropout = {dropout}, hidden_dim = {hidden_dim}, num_layers = {num_layers}, weight_decay = {weight_decay}, learning_rate = {learning_rate}, counter = {counter}",file=sys.stderr)


# Define a function to perform link prediction training
train_losses = []
def train_loop(model, optimizer, data):
    global train_losses
    model.train()

    # Forward pass through model
    node_embeddings = model(data)
    
    # Sample positive and negative edges FROM TRAINING EDGE INDEX for making predictions
    # Negative edge index is so large that we dont need to care about overlap between train and test edge sets?
    pos_edge_index = data.train_edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes) # Generate same number of negative samples as positive samples

    # Compute similarity scores for each pos/neg edge
    pos_pred = score_edges(node_embeddings, pos_edge_index)  # Compute scores for positive edges
    neg_pred = score_edges(node_embeddings, neg_edge_index)  # Compute scores for negative edges

    # Compute BCE loss
    loss = compute_loss(pos_pred, neg_pred)  # Compute loss
    print(f'Training loss: {loss}',file=sys.stderr)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_losses.append(loss.item())

# Define test loop using positive test edges
validation_losses = []
def test_loop(model, data):
    global validation_losses

    model.eval()

    with torch.no_grad():
        # Forward pass through model
        node_embeddings = model(data)

        # Sample negative edges for val set
        pos_edge_index = data.val_edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)

        # Compute similarity scores for positive and negative edges in test set
        pos_pred = score_edges(node_embeddings, pos_edge_index)
        neg_pred = score_edges(node_embeddings, neg_edge_index)

        # Compute BCE loss
        loss = compute_loss(pos_pred, neg_pred)  # Compute loss
        validation_losses.append(loss.item())
        print(f'Validation loss: {loss}',file=sys.stderr)

for i in range(epochs):
    print(f'===== Epoch {i} ======================================================================================',file=sys.stderr)
    # Run the train loop
    # data should contain node features (data.x) and edge indices (data.edge_index)
    train_loop(model=model, optimizer=optimizer, data=data)
    test_loop(model=model, data=data)

print(f"Training complete\n",file=sys.stderr)

################## Loss on test set ####################
########################################################
print("Evaluating performance on test set...",file=sys.stderr)
model.eval()

with torch.no_grad():
    # Forward pass through model
    node_embeddings = model(data)

    # Sample negative edges for val set
    pos_edge_index = data.test_edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)

    # Compute similarity scores for positive and negative edges in test set
    pos_pred = score_edges(node_embeddings, pos_edge_index)
    neg_pred = score_edges(node_embeddings, neg_edge_index)

    # Compute BCE loss
    test_loss = compute_loss(pos_pred, neg_pred).item()  # Compute loss
    print(f'Test loss: {test_loss}',file=sys.stderr)

################## Save loss list ######################
########################################################
df_losses = pd.DataFrame({
    'epoch':[i+1 for i in range(epochs)],
    'train_loss':train_losses,
    'validation_losses':validation_losses,
    'test_loss':[test_loss for i in range(epochs)]
})

df_losses.to_csv(f'{evaluation_dir}/{model_name}_loss.csv', header=True, index=False)

################## Save model ##########################
########################################################
try:
    torch.save(model, f'{output_model_directory}/{model_name}.pth')
    print('Model successfully saved',file=sys.stderr)
except:
    print(f'Error saving model\n',file=sys.stderr)
    raise