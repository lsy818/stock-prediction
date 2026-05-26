import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, gru_out):
        # gru_out shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores
        scores = self.attn(gru_out) # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # (batch_size, seq_len, 1)
        
        # Weighted sum of gru outputs
        context = torch.sum(weights * gru_out, dim=1) # (batch_size, hidden_size)
        return context, weights

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(AttentionGRU, self).__init__()
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
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size)
        
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
        out = self.layer_norm(out)
        
        # Apply attention over the sequence
        context, _ = self.attention(out)
        # context shape: (batch_size, hidden_size)
        
        pred = self.fc(context)
        return pred.squeeze(-1) # shape: (batch_size)


class EnsembleAttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, num_models=3):
        super(EnsembleAttentionGRU, self).__init__()
        self.num_models = num_models
        
        # Initialize multiple AttentionGRU models
        self.models = nn.ModuleList([
            AttentionGRU(input_size, hidden_size, num_layers, dropout)
            for _ in range(num_models)
        ])
        
    def forward(self, x):
        preds = []
        for model in self.models:
            preds.append(model(x))
            
        # preds is a list of tensors of shape (batch_size,)
        stacked = torch.stack(preds, dim=1) # (batch_size, num_models)
        
        # Average the predictions
        avg_pred = torch.mean(stacked, dim=1) # (batch_size,)
        return avg_pred
