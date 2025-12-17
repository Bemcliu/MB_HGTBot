import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import RGCNConv


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dimension, num_heads, mlp_hidden_size, dropout=0.5):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiheadAttention(embed_dim=embedding_dimension, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dimension)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_size, embedding_dimension)
        )
        self.layer_norm2 = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(embedding_dimension)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        x = self.batch_norm(x.squeeze(1)).unsqueeze(1)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm2(x)
        return x


class MB_HGTBot(nn.Module):
    def __init__(self, tweet_size=768, num_prop_size=6, comments_prop_size=7, embedding_dimension=192, dropout=0.5,
                 num_heads=8, mlp_hidden_size=192, num_layers=6, temperature=0.5):
        super(MB_HGTBot, self).__init__()
        self.temperature = temperature
        feature_dim = int(embedding_dimension / 3)

        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, feature_dim), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, feature_dim), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(comments_prop_size, feature_dim), nn.LeakyReLU())

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embedding_dimension, num_heads, mlp_hidden_size, dropout) for _ in
             range(num_layers)])

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.mlp_output = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_size, embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, 2)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def supervised_contrastive_loss(self, features, labels):
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.unsqueeze(1)
        matches = (labels == labels.T).float().to(features.device)
        positives = matches * torch.exp(similarity_matrix)
        all_pairs = torch.exp(similarity_matrix)
        loss = -torch.log(positives.sum(dim=1) / all_pairs.sum(dim=1))
        return loss.mean()

    def forward(self, tweet, num_prop, comments_prop, edge_index, edge_type, labels, train_idx):
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(comments_prop)
        features = torch.cat((t, n, c), dim=1).unsqueeze(1)

        for layer in self.layers:
            features = layer(features)
        features = features.squeeze(1)

        contrastive_loss = 0
        if labels is not None:
            contrastive_loss = self.supervised_contrastive_loss(features[0:train_idx], labels[0:train_idx])

        x = self.rgcn(features, edge_index, edge_type)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.mlp_output(x)
        return x, contrastive_loss
