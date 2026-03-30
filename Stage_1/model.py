import torch.nn as nn
import torch

class LSTMAttentionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, sentence_embeddings):
        lstm_out, _ = self.lstm(sentence_embeddings)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector, attn_weights


class EssayScoringModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(EssayScoringModel, self).__init__()
        self.lstm_attention = LSTMAttentionModel(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, sentence_embeddings):
        context_vector, _ = self.lstm_attention(sentence_embeddings)
        score = self.fc(context_vector)
        return score
