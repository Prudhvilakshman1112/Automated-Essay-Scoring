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
    """
    Stage 3 — DOL-enhanced model.
    The final FC layer accepts (hidden_dim + 16) inputs:
      - hidden_dim (140) features from the LSTM attention context vector
      - 16 projected density features from nn.Linear(1→16) + ReLU
    This forces the model to consider content richness, not just length.
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(EssayScoringModel, self).__init__()
        self.lstm_attention = LSTMAttentionModel(embedding_dim, hidden_dim)
        # Learnable projection for density scalar (1 -> 16)
        self.density_projection = nn.Linear(1, 16)
        # Final FC layer now takes hidden_dim + 16 (projected density features)
        self.fc = nn.Linear(hidden_dim + 16, 1)

    def forward(self, sentence_embeddings, density_scalar):
        """
        Args:
            sentence_embeddings : Tensor [batch, max_sentences, embedding_dim]
            density_scalar      : Tensor [batch, 1]  — information density score
        """
        context_vector, _ = self.lstm_attention(sentence_embeddings)
        
        # Project density scalar and apply ReLU (non-linearity allows thresholding)
        density_feat = torch.relu(self.density_projection(density_scalar))
        
        # Concatenate projected density features to context vector before scoring
        enriched_vector = torch.cat([context_vector, density_feat], dim=1)
        score = self.fc(enriched_vector)
        return score
