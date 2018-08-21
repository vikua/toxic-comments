import os
import argparse

import pandas as pd
import numpy as np

from model import NNClassifier, VocabularyProcessor


TEXT_COL = 'comment_text'
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def train(args): 
    data = pd.read_csv(os.path.join(args.input_path, 'train.csv'))

    indices = np.random.permutation(data.shape[0])

    split = int(data.shape[0] * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    vp = VocabularyProcessor()
    X_train = vp.fit_transform(train_data[TEXT_COL].values)
    y_train = train_data[CLASSES].values

    X_test = vp.transform(test_data[TEXT_COL].values)
    y_test = test_data[CLASSES].values

    clf = NNClassifier(len(CLASSES), vp,
                       embedding_dim=args.embedding_dim, 
                       dropout_keep_prob=args.dropout_keep_prob)
    clf.fit(X_train, y_train, X_test, y_test, 
            epochs=args.epochs,
            batch_size=args.batch_size)

    clf.save()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple toxic comments classifier')

    parser.add_argument('--input-path', dest='input_path', type=str, 
                        help='Path to input ')
    parser.add_argument('--output-path', dest='output_path', type=str,
                        help='Path to save the model')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dropout-keep-prob', dest='dropout_keep_prob', type=float, 
                        default=0.5, help='Dropout keep probability')
    parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=100)

    args = parser.parse_args()
    train(args)