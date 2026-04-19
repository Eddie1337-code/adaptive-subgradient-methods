"""
Neural network model for Click-Through Rate (CTR) prediction.

Architecture: Feature Hashing → Embedding (with per-field offsets) → MLP → Logit
"""

import torch
import torch.nn as nn


class CTRModel(nn.Module):
    """
    CTR prediction model using a single shared embedding table with
    per-field offsets, followed by a multi-layer perceptron.

    Each of the n_fields feature fields maps into its own region of the
    embedding table via:  index_for_lookup = field_offset + hashed_index

    This replaces n_fields separate nn.Embedding modules with one large
    embedding, enabling a single fused lookup per forward pass.
    """

    def __init__(self, n_fields, hash_size=100000, embed_dim=8,
                 hidden_dims=(128, 64)):
        super().__init__()
        self.n_fields = n_fields
        self.hash_size = hash_size
        self.embed_dim = embed_dim

        # Single embedding table: n_fields * hash_size rows
        self.embedding = nn.Embedding(n_fields * hash_size, embed_dim)

        # Offsets: field i uses rows [i*hash_size, (i+1)*hash_size)
        offsets = torch.arange(n_fields, dtype=torch.long) * hash_size
        self.register_buffer('field_offsets', offsets)

        # MLP head
        input_dim = n_fields * embed_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, field_indices):
        """
        Args:
            field_indices: (batch_size, n_fields) LongTensor of hashed feature indices

        Returns:
            logits: (batch_size,) float tensor
        """
        # Add per-field offsets: (batch, n_fields) + (n_fields,) → (batch, n_fields)
        global_indices = field_indices + self.field_offsets
        # Single lookup: (batch, n_fields) → (batch, n_fields, embed_dim)
        embeds = self.embedding(global_indices)
        # Flatten: (batch, n_fields * embed_dim)
        x = embeds.view(embeds.size(0), -1)
        return self.mlp(x).squeeze(-1)
