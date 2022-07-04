import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(text, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    vocab = torch.load(model_path + '/vocab.pt')
    tokenizer = torch.load(model_path + '/tokenizer.pt')
    model = TextClassificationModel(len(vocab), 128, 2)
    model.load_state_dict(torch.load(model_path + '/model.pt')['model_state_dict'])
    model.eval()

    # Predict
    text = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64).to(device)
    offset = torch.tensor([0], dtype=torch.int64).to(device)
    predited_label = model(text, offset)

    return predited_label.argmax(1).to('cpu').tolist()[0]
