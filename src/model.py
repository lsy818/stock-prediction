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
