import pandas as pd
from torch_geometric.data import Data
import torch
import pickle
import numpy as np

class UserIdMapper:
    """Convert user hash to integer for PyTorch Geometric data object"""
    def __init__(self, unique_users=None, dict_path=None):
        # If dict_path specified, load from this, otherwise look for the list of unique user hashes
        self.dict_path = dict_path
        if dict_path:
            self.load_from_dict()
        elif unique_users is not None:
            self.user_to_id = {user: idx for idx, user in enumerate(sorted(unique_users))}
            self.id_to_user = {idx: user for user, idx in self.user_to_id.items()}
        else:
            raise ValueError('Either unique users or dict path required.')

    def load_from_dict(self):
        """Load this object class using a dictionary mapping already saved"""
        with open(self.dict_path, 'rb') as f:
            self.user_to_id = pickle.load(f)
        self.id_to_user = {idx: user for user, idx in self.user_to_id.items()}

    def save_dict(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.user_to_id, f)
         
    def transform(self, user):
        """Convert hash to int"""
        return self.user_to_id[user]
    
    def get_original_user(self, id):
        """Regather hash from int"""
        return self.id_to_user[id]

def prepare_edge_index(df):
    """Create torch tensor the way Data object expects edge list"""
    df_copy = df.copy()[['target', 'source']].rename(columns={'source':'target', 'target':'source'})
    df_combined = pd.concat([df, df_copy], axis=0).drop_duplicates()
    assert all(df_combined.groupby(['source', 'target']).size().unique()) == 1
    edge_index = torch.tensor(df_combined[['source', 'target']].values, dtype=torch.long).t().contiguous()
    return edge_index

def prepare_node_features(df):
    """Create torch tensor the way Data object expects node features"""
    return torch.tensor(df.values, dtype=torch.float)

def create_torch_data_object(df_edges, df_node_features, index_cols):
    # Ensure that the df_node_features is sorted based on user (so that feature_tensor[0,:] == user #0)
    df_node_features_sorted = df_node_features.sort_values('user_id', ascending=True)
    # Separate all index related information (player name, date, team, etc.)
    index = pd.concat([df_node_features_sorted.pop(col) for col in index_cols], axis=1)
    # Get tensor of edge indexes
    edge_index = prepare_edge_index(df_edges)
    # Get tensor of node features
    node_features_sorted = prepare_node_features(df_node_features_sorted)
    # Prepare Data object
    data = Data(x=node_features_sorted, edge_index=edge_index)
    return index, data

# Define a function to split edge indices into train and test sets
def split_edges(edge_index, train_ratio=0.65, val_ratio=0.20):
    num_edges = edge_index.shape[1]
    num_train = int(num_edges * train_ratio)
    num_val = int(num_edges * val_ratio)

    # Randomly permute edges
    indices = np.random.permutation(num_edges)

    # Create sets
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:]
    return edge_index[:, train_indices], edge_index[:, val_indices], edge_index[:, test_indices]

