import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import RGCNConv


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dimension, num_heads, mlp_hidden_size, dropout=0.5):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=num_heads,
            batch_first=True
        )
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
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 embedding_dimension=192, dropout=0.5, num_heads=8, mlp_hidden_size=256,
                 num_layers=6, temperature=0.5):
        super(MB_HGTBot, self).__init__()
        self.temperature = temperature
        feature_dim = int(embedding_dimension / 4)

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, feature_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, feature_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, feature_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, feature_dim),
            nn.LeakyReLU()
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dimension, num_heads, mlp_hidden_size, dropout)
            for _ in range(num_layers)
        ])

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.mlp_output = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_size, embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, 2)
        )

        self.initialize_weights()

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

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

        positive_sum = positives.sum(dim=1) + 1e-10
        all_pair_sum = all_pairs.sum(dim=1) + 1e-10

        loss = -torch.log(positive_sum / all_pair_sum)
        return loss.mean()

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type, labels,
                batch_size=10000, train_idx=None):

        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        features = torch.cat((d, t, n, c), dim=1).unsqueeze(1)

        accumulated_features = []
        total_contrastive_loss = 0

        num_samples = labels.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        contrastive_sample_limit = train_idx if train_idx is not None else num_samples
        actual_contrastive_samples = 0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            if end_idx > num_samples:
                end_idx = num_samples

            batch_features = features[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            for layer in self.layers:
                batch_features = layer(batch_features)

            batch_features = batch_features.squeeze(1)

            # Contrastive Loss Calculation
            current_indices = torch.arange(start_idx, end_idx, device=features.device)
            train_mask = current_indices < contrastive_sample_limit

            if train_mask.any():
                train_features = batch_features[train_mask]
                train_labels = batch_labels[train_mask]

                if train_features.size(0) > 1:
                    contrastive_loss = self.supervised_contrastive_loss(train_features, train_labels)
                    total_contrastive_loss += contrastive_loss.item() * train_features.size(0)
                    actual_contrastive_samples += train_features.size(0)

            accumulated_features.append(batch_features)

        all_features = torch.cat(accumulated_features, dim=0)

        if actual_contrastive_samples > 0:
            total_contrastive_loss /= actual_contrastive_samples
        else:
            total_contrastive_loss = 0

        x = self.rgcn(all_features, edge_index, edge_type)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.mlp_output(x)

        return x, total_contrastive_loss