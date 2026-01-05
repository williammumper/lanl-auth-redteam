import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Decoder 
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        _, (hidden, _) = self.encoder(x)
        
        # Take the last hidden state and repeat it for the decoder
        # hidden[-1] is the last layer's hidden state
        repeat_hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        out, _ = self.decoder(repeat_hidden)
        return self.output_layer(out)