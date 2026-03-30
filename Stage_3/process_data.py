import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# Download stopwords on first run
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_model.to(device)

STOPWORDS = set(stopwords.words('english'))


def get_info_density(essay_text):
    """
    Density-Over-Length (DOL) metric.
    Calculates the ratio of unique content words (non-stopwords) to total words.
    High density = rich, varied vocabulary.
    Low density = repetitive filler / padding.

    Returns a float in [0, 1].
    """
    words = essay_text.lower().split()
    total_words = len(words)
    if total_words == 0:
        return 0.0
    content_words = [w for w in words if w.isalpha() and w not in STOPWORDS]
    unique_content_words = set(content_words)
    density = len(unique_content_words) / total_words
    return density


def get_sentence_embeddings(essay_text, show_progress_bar=False):
    sentences = essay_text.split('. ')
    sentence_embeddings = sbert_model.encode(sentences, convert_to_tensor=True, show_progress_bar=show_progress_bar)
    return sentence_embeddings


def prepare_data(essays, scores, batch_size=32):
    """
    Returns:
        padded_embeddings : Tensor [N, max_sentences, 384]
        scores_tensor     : Tensor [N, 1]
        density_tensor    : Tensor [N, 1]  ← NEW in Stage 3
    """
    essay_embeddings = []
    density_scores = []

    for i in range(0, len(essays), batch_size):
        batch_essays = essays[i:i + batch_size]
        for essay in batch_essays:
            essay_embeddings.append(get_sentence_embeddings(essay))
            density_scores.append(get_info_density(essay))

    padded_embeddings = nn.utils.rnn.pad_sequence(essay_embeddings, batch_first=True)
    scores_tensor = torch.tensor(scores, dtype=torch.float32).view(-1, 1)
    density_tensor = torch.tensor(density_scores, dtype=torch.float32).view(-1, 1)
    return padded_embeddings, scores_tensor, density_tensor
