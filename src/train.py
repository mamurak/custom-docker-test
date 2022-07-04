import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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


class CollateBatch():
    def __init__(self, device, text_pipeline, label_pipeline):
        self.device = device
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def __call__(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)    


def load_data(path):
    df_train = pd.read_csv(path + '/train.csv')
    df_valid = pd.read_csv(path + '/valid.csv')
    df_test = pd.read_csv(path + '/test.csv')

    train_data = list(zip(df_train['label'], df_train['titletext']))
    valid_data = list(zip(df_valid['label'], df_valid['titletext']))
    test_data = list(zip(df_test['label'], df_test['titletext']))
    return train_data, valid_data, test_data


def create_vocab(train_iter):
    tokenizer = get_tokenizer('basic_english')
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab, tokenizer


def data_loaders(path, device, batch_size):
    train_iter, valid_iter, test_iter = load_data(path)
    vocab, tokenizer = create_vocab(train_iter)

    label_pipeline = lambda x: int(x)
    text_pipeline = lambda x: vocab(tokenizer(x))
    collate_batch = CollateBatch(device, text_pipeline, label_pipeline)
    train_loader = DataLoader(train_iter, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_iter, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_iter, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_batch)
    return train_loader, valid_loader, test_loader, vocab, tokenizer


def train_step(device, epoch, model, dataloader, optimizer, criterion):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        label = label.to(device)
        text = text.to(device)
        offsets = offsets.to(device)
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
    return model


def evaluate_step(device, model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            label = label.to(device)
            text = text.to(device)
            offsets = offsets.to(device)
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def main(input_path, output_path, epochs=5, batch_size=32, learning_rate = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_accu = None
    
    # Set data
    train_loader, valid_loader, test_loader, vocab, tokenizer = \
        data_loaders(input_path, device, batch_size)

    # Training
    model = TextClassificationModel(len(vocab), embed_dim=128, num_class=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_step(device, epoch, model, train_loader, optimizer, criterion)
        accu_val = evaluate_step(device, model, valid_loader, criterion)

        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)

    # Test
    accu_test = evaluate_step(device, model, test_loader, criterion)
    print('test accuracy {:8.3f}'.format(accu_test))

    # Save model
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state_dict, output_path + "/model.pt")
    torch.save(vocab, output_path + '/vocab.pt')
    torch.save(tokenizer, output_path + '/tokenizer.pt')