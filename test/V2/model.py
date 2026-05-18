import torch
import torch.nn as nn

class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch First: (batch, seq, feature)
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Predict a single return value
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # out shape: (batch_size, seq_len, hidden_size)
        
        # We only need the output of the last time step
        last_out = out[:, -1, :] 
        
        pred = self.fc(last_out)
        return pred.squeeze(-1) # shape: (batch_size)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model) for batch_first
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(StockTransformer, self).__init__()
        self.d_model = d_model
        
        # Linear projection from input_size (e.g. 56) to d_model (e.g. 64)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Fully connected layer for the final prediction
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Project features to d_model
        x = self.input_projection(x) # shape: (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer encoder
        out = self.transformer_encoder(x) # shape: (batch_size, seq_len, d_model)
        
        # We only need the output of the last time step for forecasting
        last_out = out[:, -1, :] # shape: (batch_size, d_model)
        
        # Final prediction
        pred = self.fc(last_out)
        return pred.squeeze(-1) # shape: (batch_size)
