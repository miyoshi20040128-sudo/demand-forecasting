import torch
import torch.nn as nn


class ITransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_features,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        target_idx=0,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.target_idx = target_idx

        # 入力: [batch, seq_len, num_features]
        # iTransformer風に [batch, num_features, seq_len] にして、
        # 各変数を1トークンとして扱う
        self.value_embedding = nn.Linear(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        x = x.permute(0, 2, 1)
        # x: [batch, num_features, seq_len]

        x = self.value_embedding(x)
        # x: [batch, num_features, d_model]

        x = self.encoder(x)
        # x: [batch, num_features, d_model]

        out = self.projection(x)
        # out: [batch, num_features, pred_len]

        target_out = out[:, self.target_idx, :]
        # target_out: [batch, pred_len]

        return target_out