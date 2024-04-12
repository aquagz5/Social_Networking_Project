import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import numpy as np
import sys

class GCNLinkPrediction(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2,hidden_dim=100, p_dropout = 0.3):
        super(GCNLinkPrediction, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.p_dropout = p_dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if num_layers == 1:
            self.conv1 = GCNConv(input_dim, hidden_dim)
        else:
            #start with one layer
            self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)])
            #add layer if there are more than one extra layer
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.conv_out = GCNConv(hidden_dim, output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
       
        if self.num_layers == 1:
            x = self.conv1(x, edge_index)
        else:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x,p=self.p_dropout, training = self.training)

        x = F.normalize(x, p=2, dim=1) # Normalize node embeddings
        return x

def score_edges(embeddings, edge_index):
    source, destination = edge_index
    source_embedding = embeddings[source]
    destination_embedding = embeddings[destination]
    scores = (source_embedding * destination_embedding).sum(dim=1)  # Dot product scoring function
    return scores

def compute_loss(pos_pred, neg_pred):
    pos_loss = -torch.log(torch.sigmoid(pos_pred)).mean()  # Binary cross-entropy loss for positive edges
    neg_loss = -torch.log(1 - torch.sigmoid(neg_pred)).mean()  # Binary cross-entropy loss for negative edges
    loss = (pos_loss + neg_loss) / 2.0
    return loss

def hit_ratio(embeddings, test_edge_index, sample_size=10000):
    print('Beginning hit ratio calculation...',file=sys.stderr)
    # Find the uniqe users that are in the test set
    unique_test_users = sorted(test_edge_index[0,:].unique().tolist())

    # Take sample of users that we will calculate hit ratios for... can't do this for 240,000
    test_users_sample = random.sample(unique_test_users, sample_size)

    hit_ratios = []
    hit_ratio_weights = []
    # For each user, sample and calculate hit ratio
    for idx, user_id in enumerate(test_users_sample):
        if (idx+1) % 100 == 0:
            print(f'Percent complete: {100.0 * (idx+1) / len(test_users_sample)}%',file=sys.stderr)
        # User embedding
        user_embedding = embeddings[user_id]
        #print(user_embedding)

        # Find the test set users that they are actually connected to
        existing_connections = test_edge_index[1, test_edge_index[0] == user_id].tolist()
        #print(existing_connections)
        hit_ratio_weights.append(min(len(existing_connections), 10))

        # Construct a sample of the embedding matrix
        # Generate a pool of random integers between (excluding the existing connections)
        pool = set(range(min(unique_test_users), max(unique_test_users)+1)) - set(existing_connections)

        # Sample 3 distinct user ids from the pool
        sampled_users = existing_connections + random.sample(pool, 1000-len(existing_connections))
        #print(sampled_users)

        # Sample embeddings
        sample_embeddings = embeddings[sampled_users]
        # print(sample_embeddings)
        # print(sample_embeddings.size())

        # Compute edge scores between user id and the other sampled users
        user_scores = torch.matmul(user_embedding, sample_embeddings.t())
        # print(user_scores)
        # print(user_scores.size())

        # Find the top 10 indices
        top_10_indices = torch.argsort(user_scores, descending=True)[0:10].tolist()
        #print(top_10_indices)

        # Find the users that the top 10 indices corresponds to
        top_10_users = [sampled_users[idx] for idx in top_10_indices]
        #print(top_10_users)

        # Extend the hit list to include results for this user
        hit_list = [1 if user in existing_connections else 0 for user in top_10_users]
        #print(hit_list)

        # Calculate hit ratio (percentage of actual edges that were located in top 10 indices)
        hit_ratio = sum(hit_list) / min(len(hit_list), len(existing_connections))
        hit_ratios.append(hit_ratio)

    # print(hit_ratios)
    # print(hit_ratio_weights)
    # Weighted final hit ratio
    weighted_hit_ratio = np.sum((np.array(hit_ratios) * np.array(hit_ratio_weights))) / np.sum(hit_ratio_weights)

    return float(weighted_hit_ratio)

# def get_graph_recommendations(embeddings, edge_index, batch_size=1000):
#     """Compute recommendations for entire graph in batches"""
#     # How many batches are required?
#     num_nodes = embeddings.size(0)
#     num_batches = (num_nodes + batch_size - 1) // batch_size

#     print(f'Required batches: {num_batches}')

#     # Initialize an empty matrix to store scores
#     # scores = torch.zeros(num_nodes, num_nodes)
#     # print('Scores iniitalized with zeros.')

#     # Initialize an empty list to store scores
#     recommendations_list = []

#     # Perform matrix multiplication in batches
#     for i in range(num_batches):
#         print(f'Starting batch {i}')
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, num_nodes)
#         batch_embeddings = embeddings[start_idx:end_idx]

#         # Compute scores for the current batch
#         batch_scores = torch.matmul(batch_embeddings, embeddings.t())


#         # Set recommendations for connected nodes within the batch to -inf
#         edge_list = edge_index.t().tolist()
#         edge_list_filtered = [[src, dest] for src, dest in edge_list if ((src >= start_idx) and (src < end_idx))]
#         for i, src_dest in edge_list_filtered:
#             batch_scores[src_dest[0] - start_idx, src_dest[1]] = float('-inf')
#             #batch_scores[i, start_idx+i] = float('-inf') # Self recommendation

#         # Sort scores for each row and get top 10 column indices
#         top_10_indices = torch.argsort(batch_scores, dim=1, descending=True)[:, :10]

#         # Append top 10 recommendations to the list
#         recommendations_list.append(top_10_indices)

#     # Concatenate batch_scores along the rows to form the final scores matrix
#     recommendations = torch.cat(recommendations_list, dim=0)

#     # # Diagonal elements represent self-similarity, which should be set to negative infinity
#     # scores.fill_diagonal_(float('-inf'))
#     # # Set similarity scores between connected nodes to negative infinity
#     # for src, dest in edge_index.t().tolist():
#     #     scores[src, dest] = float('-inf')
#     #     scores[dest, src] = float('-inf')

#     return recommendations

# ========== Adjusting scoring (pairwise node similarity = the edge score) to be MLP / NN instead of just using dot product ==========
# import torch.nn as nn

# class MLPScore(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(MLPScore, self).__init__()
#         self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # Concatenate node embeddings, hence input_dim * 2
#         self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single scalar score

#     def forward(self, src_embedding, dst_embedding):
#         combined_embedding = torch.cat([src_embedding, dst_embedding], dim=1)  # Concatenate node embeddings
#         x = F.relu(self.fc1(combined_embedding))
#         x = self.fc2(x)
#         return x.squeeze()

# # Instantiate the MLP scoring function
# mlp_score = MLPScore(input_dim=16, hidden_dim=8)  # Assuming input_dim=16 and hidden_dim=8

# # Modify the score_edges function to use MLP scoring
# def score_edges(embeddings, edge_index):
#     src, dst = edge_index
#     src_embedding = embeddings[src]
#     dst_embedding = embeddings[dst]
#     scores = mlp_score(src_embedding, dst_embedding)  # Use MLP scoring function
#     return scores



