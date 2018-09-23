import os
import argparse

import pandas as pd
import numpy as np
import cloudpickle

from model import NNClassifier, VocabularyProcessor, E2EClassifier, CLASSES
from utils import make_keras_picklable


TEXT_COL = 'comment_text'


make_keras_picklable()


def train(args): 
    data = pd.read_csv(os.path.join(args.input_path, 'train.csv'))

    indices = np.random.permutation(data.shape[0])

    split = int(data.shape[0] * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    vp = VocabularyProcessor(args.max_features)

    X_train = vp.fit_transform(train_data[TEXT_COL].values)
    y_train = train_data[CLASSES].values

    X_test = vp.transform(test_data[TEXT_COL].values)
    y_test = test_data[CLASSES].values

    clf = NNClassifier(len(CLASSES), vp,
                       embedding_dim=args.embedding_dim, 
                       dropout=args.dropout, 
                       hidden_units=args.hidden_units, 
                       model_path=os.path.join(args.output_path, 'toxic.h5'))

    clf.fit(X_train, y_train, X_test, y_test, 
            epochs=args.epochs, batch_size=args.batch_size)

    with open(os.path.join(args.output_path, 'vocab.pkl'), 'wb') as f: 
        cloudpickle.dump(vp, f)

    # creating classifier wrapper for e2e predictions
    e2e = E2EClassifier(vocab_processor, clf)
    with open(os.path.join(args.output_path, 'keras.pkl'), 'wb') as f: 
        cloudpickle.dump(e2e, f)


def predict(args): 
    with open(args.model_path, 'rb') as f: 
        clf = cloudpickle.load(f)

    x = [args.sentence]

    print('Predictions for: "{}"'.format(args.sentence))
    print(clf.predict_proba(x))
    print(clf.predict(x))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple toxic comments classifier')

    parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=100)
    parser.add_argument('--hidden-units', dest='hidden_units', type=int, default=50)

    subparser = parser.add_subparsers(dest='command')

    train_parser = subparser.add_parser('train')
    train_parser.add_argument('--input-path', dest='input_path', type=str, 
                              help='Path to input ')
    train_parser.add_argument('--output-path', dest='output_path', type=str,
                              help='Path to save the model')
    train_parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                              help='Number of epochs')
    train_parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                              help='Batch size')
    train_parser.add_argument('--dropout', dest='dropout', type=float, 
                              default=0.5, help='Dropout - fraction of units to drop')
    train_parser.add_argument('--max-features', dest='max_features', type=int, default=None)

    predict_parser = subparser.add_parser('predict')
    predict_parser.add_argument('--model-path', dest='model_path', type=str,
                                help='Path to trained model')
    predict_parser.add_argument('--sentence', dest='sentence', type=str, 
                                help='Sentence to predict for', default="Sample input sentence")

    args = parser.parse_args()
    if args.command == 'train': 
        train(args)
    elif args.command == 'predict': 
        predict(args)
    else:
        raise ValueError('Unknown command {}'.format(args.command))
