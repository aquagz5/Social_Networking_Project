import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLinkPrediction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLinkPrediction, self).__init__()
        self.conv1 = GCNConv(input_dim, 100)
        self.conv2 = GCNConv(100, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
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

def get_graph_recommendations(embeddings, edge_index, batch_size=1000):
    """Compute recommendations for entire graph in batches"""
    # How many batches are required?
    num_nodes = embeddings.size(0)
    num_batches = (num_nodes + batch_size - 1) // batch_size

    print(f'Required batches: {num_batches}')

    # Initialize an empty matrix to store scores
    # scores = torch.zeros(num_nodes, num_nodes)
    # print('Scores iniitalized with zeros.')

    # Initialize an empty list to store scores
    recommendations_list = []

    # Perform matrix multiplication in batches
    for i in range(num_batches):
        print(f'Starting batch {i}')
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_nodes)
        batch_embeddings = embeddings[start_idx:end_idx]

        # Compute scores for the current batch
        batch_scores = torch.matmul(batch_embeddings, embeddings.t())

        # Diagonal elements represent self-recommendations, which should be set to -inf
        batch_scores.fill_diagonal_(float('-inf'))

        # Set recommendations for connected nodes within the batch to -inf
        edge_list = edge_index.t().tolist()
        edge_list_filtered = [[src, dest] for src, dest in edge_list if ((src >= start_idx) and (src < end_idx))]
        for src, dest in edge_list_filtered:
            batch_scores[src - start_idx, dest] = float('-inf')

        # Sort scores for each row and get top 10 column indices
        top_10_indices = torch.argsort(batch_scores, dim=1, descending=True)[:, :10]

        # Append top 10 recommendations to the list
        recommendations_list.append(top_10_indices)

    # Concatenate batch_scores along the rows to form the final scores matrix
    recommendations = torch.cat(recommendations_list, dim=0)

    # # Diagonal elements represent self-similarity, which should be set to negative infinity
    # scores.fill_diagonal_(float('-inf'))
    # # Set similarity scores between connected nodes to negative infinity
    # for src, dest in edge_index.t().tolist():
    #     scores[src, dest] = float('-inf')
    #     scores[dest, src] = float('-inf')

    return recommendations

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



