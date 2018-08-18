import os
import argparse
import pickle

import pandas as pd
import numpy as np

from model import NNClassifier, VocabularyProcessor


TEXT_COL = 'comment_text'
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def prepare_data(args): 
    x_path = os.path.join(args.output_dir, 'X.npy')
    y_path = os.path.join(args.output_dir, 'y.npy')
    vocab_path = os.path.join(args.output_dir, 'vocab.pkl')
    if os.path.exists(x_path): 
        os.remove(x_path)
    if os.path.exists(y_path): 
        os.remove(y_path)
    if os.path.exists(vocab_path): 
        os.remove(vocab_path)

    data = pd.read_csv(args.data_path)

    vocab_processor = VocabularyProcessor()
    vocab_processor.fit(data[TEXT_COL].values)
    X = vocab_processor.transform(data[TEXT_COL].values)
    y = data[CLASSES].values

    with open(vocab_path, 'wb') as f: 
        pickle.dump(vocab_processor, f)
    np.save(x_path, X)
    np.save(y_path, y)


def train(args): 
    X = np.load(os.path.join(args.input_path, 'X.npy'))
    y = np.load(os.path.join(args.input_path, 'y.npy'))

    clf = NNClassifier(len(CLASSES), 
                       embedding_dim=args.embedding_dim, 
                       dropout_keep_prob=args.dropout_keep_prob)
    clf.fit(texts, labels, 
            epochs=args.epochs,
            batch_size=args.batch_size)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple toxic comments classifier')
    subs = parser.add_subparsers(title='command', dest='command')

    vocab_parser = subs.add_parser('prepare_data')
    vocab_parser.add_argument('--data-path', dest='data_path', type=str, 
                              help='Path to raw train data')
    vocab_parser.add_argument('--output-dir', dest='output_dir', type=str, 
                              help='Path to output directory')

    train_parser = subs.add_parser('train')
    train_parser.add_argument('--input-path', dest='input_path', type=str, 
                              help='Path to input ')
    train_parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                              help='Number of epochs')
    train_parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                              help='Batch size')
    train_parser.add_argument('--dropout-keep-prob', dest='dropout_keep_prob', type=float, 
                        default=0.5, help='Dropout keep probability')
    train_parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=100)

    args = parser.parse_args()
    if args.command == 'prepare_data': 
        prepare_data(args)
    elif args.command == 'train': 
        train(args)