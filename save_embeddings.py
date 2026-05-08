import torch
from load_model import return_model
import joblib

model = return_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

book_embeddings = torch.load('./book_embeddings.pt', map_location=device)

print(book_embeddings.shape)