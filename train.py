import os
import argparse

import pandas as pd
import numpy as np

from model import NNClassifier, VocabularyProcessor


TEXT_COL = 'comment_text'
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def train(args): 
    data = pd.read_csv(os.path.join(args.input_path, 'train.csv'))

    vp = VocabularyProcessor()
    texts = vp.fit_transform(data[TEXT_COL].values)

    labels = data[CLASSES].values

    clf = NNClassifier(len(CLASSES), vp,
                       embedding_dim=args.embedding_dim, 
                       dropout_keep_prob=args.dropout_keep_prob)
    clf.fit(texts, labels, 
            epochs=args.epochs,
            batch_size=args.batch_size)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple toxic comments classifier')

    parser.add_argument('--input-path', dest='input_path', type=str, 
                        help='Path to input ')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dropout-keep-prob', dest='dropout_keep_prob', type=float, 
                        default=0.5, help='Dropout keep probability')
    parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=100)

    args = parser.parse_args()
    train(args)