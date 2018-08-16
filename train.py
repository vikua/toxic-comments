import os
import argparse

import pandas as pd 

from model import NNClassifier


def main(args): 
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    texts = train['comment_text'].values
    labels = train[classes].values

    clf = NNClassifier(len(classes), embedding_dim=args.embedding_dim, 
                       dropout_keep_prob=args.dropout_keep_prob)
    clf.fit(texts, labels)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple toxic comments classifier')
    
    parser.add_argument('--data-path', dest='data_path', type=str, 
                        help='Path to train/test data')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size')
    parser.add_argument('--dropout-keep-prob', dest='dropout_keep_prob', type=float, 
                        default=0.5, help='Dropout keep probability')
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=100)

    args = parser.parse_args()
    main(args)
