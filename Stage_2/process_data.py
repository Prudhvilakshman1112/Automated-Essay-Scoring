import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_model.to(device)


def get_sentence_embeddings(essay_text, show_progress_bar=False):
    sentences = essay_text.split('. ')
    sentence_embeddings = sbert_model.encode(sentences, convert_to_tensor=True, show_progress_bar=show_progress_bar)
    return sentence_embeddings


def prepare_data(essays, scores, batch_size=32):
    essay_embeddings = []
    for i in range(0, len(essays), batch_size):
        batch_essays = essays[i:i + batch_size]
        batch_embeddings = [get_sentence_embeddings(essay) for essay in batch_essays]
        essay_embeddings.extend(batch_embeddings)
    padded_embeddings = nn.utils.rnn.pad_sequence(essay_embeddings, batch_first=True)
    scores = torch.tensor(scores, dtype=torch.float32).view(-1, 1)
    return padded_embeddings, scores
