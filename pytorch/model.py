import argparse

import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from nltk.tokenize import word_tokenize


CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class NNClassifier(nn.Module): 

    def __init__(self, num_classes, vocab_size, hidden_dim, embedding_dim): 
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.prediction = nn.Sigmoid()


    def forward(self, seq): 
        embed = self.embedding(seq)
        output, _ = self.lstm(embed)
        feature = output[-1, :, :]
        feature = self.linear(feature)
        feature = self.dropout(feature)
        out = self.out(feature)
        return self.prediction(out)


class BatchWrapper(object): 

    def __init__(self, data, x_var, y_vars): 
        self.data = data
        self.x_var = x_var
        self.y_vars =y_vars

    def __iter__(self): 
        for batch in self.data: 
            x = getattr(batch, self.x_var)
            y = torch.cat([getattr(batch, col).unsqueeze(1) for col in self.y_vars], dim=1).float()
            yield (x, y)

    def __len__(self): 
        return len(self.data)


def train(args): 
    TEXT = data.Field(sequential=True, tokenize=word_tokenize, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    fields = [('id', None), 
              ('comment_text', TEXT), ('toxic', LABEL), 
              ('severe_toxic', LABEL), ('obscene', LABEL),
              ('threat', LABEL), ('insult', LABEL), 
              ('identity_hate', LABEL)]

    dataset = data.TabularDataset(path=args.input_path, 
                                  format='csv', 
                                  fields=fields,
                                  skip_header=True)

    train, validation = dataset.split(split_ratio=0.8)

    TEXT.build_vocab(train)

    train_inter, val_inter = data.BucketIterator.splits(
        (train, validation), 
        batch_sizes=(args.batch_size, args.batch_size), 
        device=-1, 
        sort_key=lambda x: len(x.comment_text),
        sort_within_batch=False,
        repeat=False,
    )

    train_data = BatchWrapper(train_inter, 'comment_text', CLASSES)
    val_data = BatchWrapper(val_inter, 'comment_text', CLASSES)

    model = NNClassifier(len(CLASSES), len(TEXT.vocab), args.hidden_units, args.embedding_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1): 
        running_loss = 0.0
        model.train()
        for x, y in tqdm.tqdm(train_data): 
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_func(y, preds)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0] * x.size(0)
        epoch_loss = running_loss / len(train)

        val_loss = 0.0
        model.eval()
        for x, y in val_data:
            preds = model(x)
            loss = loss_func(x, preds)
            val_loss += loss.data[0] * x.size(0)

        val_loss /= len(validation)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, 
                                                                                 epoch_loss, 
                                                                                 val_loss))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple toxic comments classifier')

    parser.add_argument('--input-path', dest='input_path', type=str, 
                        help='Path to input ')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=300)
    parser.add_argument('--hidden-units', dest='hidden_units', type=int, default=256)

    args = parser.parse_args()
    train(args)