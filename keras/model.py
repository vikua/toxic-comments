import os
import argparse
import multiprocessing as mp
import tempfile
from functools import partial

import pandas as pd
import numpy as np
import cloudpickle
import tensorflow as tf 

from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


TEXT_COL = 'comment_text'
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def auc_roc(y_true, y_pred): 
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    for v in metric_vars: 
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]): 
        value = tf.identity(value)
        return value



class VocabularyProcessor(object): 

    def __init__(self, max_features=None): 
        self._max_features = max_features
        self._max_seq_len = None
        self._tokenizer = None

    @property
    def word_index(self): 
        if not self._tokenizer: 
            raise ValueError('Tokenizer was not created. Please call fit method')
        return self._tokenizer.word_index

    @property
    def max_seq_len(self): 
        return self._max_seq_len

    @property
    def max_features(self): 
        return self._max_features

    def fit(self, data): 
        self._tokenizer = Tokenizer(num_words=self._max_features, 
                                    oov_token='<unk>')

        self._tokenizer.fit_on_texts(list(data))

        if not self._max_features: 
            self._max_features = len(self._tokenizer.word_index)

        transformed = self._tokenizer.texts_to_sequences(data)
        self._max_seq_len = max([len(x) for x in transformed])

    def transform(self, data): 
        seq = self._tokenizer.texts_to_sequences(data)
        seq = pad_sequences(seq, maxlen=self._max_seq_len, padding='post')
        return seq

    def fit_transform(self, data): 
        self.fit(data)
        return self.transform(data)


class NNClassifier(object):

    def __init__(self, num_clases, vocab_processor, **kwargs):
        self._num_classes = num_clases
        self._vocab_processor = vocab_processor

        self._model = None

        # hyperparameters
        self._embedding_dim = kwargs.pop('embedding_dim', 100)
        self._dropout = kwargs.pop('dropout', 0)
        self._lstm_units = kwargs.pop('lstm_units', 128)
        self._hidden_units = kwargs.pop('hidden_units', 50) 
        self._model_path = kwargs.pop('model_path', '/tmp/toxic')

    def build_model(self): 
        inputs = tf.keras.Input(shape=(self._vocab_processor.max_seq_len, ), dtype='int32', name='inputs')

        embed = tf.keras.layers.Embedding(self._vocab_processor.max_features + 1, 
                                          self._embedding_dim,
                                          mask_zero=False,
                                          trainable=True)(inputs)
        
        lstm = tf.keras.layers.LSTM(self._lstm_units, return_sequences=True)(embed)
        
        max_pool = tf.keras.layers.GlobalMaxPool1D()(lstm)
        
        dropout = tf.keras.layers.Dropout(self._dropout)(max_pool)
        
        dense = tf.keras.layers.Dense(self._hidden_units, activation='relu')(dropout)
        
        dropout = tf.keras.layers.Dropout(self._dropout)(dense)

        predictions = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')(dropout)

        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model

    def save(self, path): 
        self.model.save(path)

    def load(self, path): 
        self.model.load_weights(path)

    @property
    def model(self):
        if not self._model:
            self._model = self.build_model()
        return self._model

    def fit(self, X_train, y_train, X_test, y_test, **kwargs): 
        epochs = kwargs.pop('epochs', 5)
        batch_size = kwargs.pop('batch_size', 128)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                                    patience=3, 
                                                                    min_lr=0.0001, 
                                                                    verbose=1)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self._model_path, save_best_only=True)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
                           loss='binary_crossentropy', 
                           metrics=['accuracy', auc_roc])

        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_test, y_test),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stopping, reduce_learning_rate, 
                                            model_checkpoint])
        return history

    def predict(self, X): 
        predictions = self.model.predict(X)
        return predictions

    def __getstate__(self): 
        state = self.__dict__.copy()
        state['session'] = None 

        if state.get('_model'): 
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                tf.keras.models.save_model(state['_model'], fd.name, overwrite=True)
                state['_model'] = fd.read()       
        
        return state

    def __setstate__(self, state): 
        if state.get('_model'): 
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd: 
                fd.write(state['_model'])
                fd.flush()

                state['_model'] = tf.keras.models.load_model(fd.name, 
                                                             custom_objects={'auc_roc': auc_roc})
        self.__dict__.update(state)


class E2EClassifier(object): 

    def __init__(self, vocab_processor, clf):
        self._vocab_processor = vocab_processor
        self._clf = clf

    def predict_proba(self, arr): 
        x = self._vocab_processor.transform(arr)
        return self._clf.predict(x)

    def predict(self, arr, threshold=0.5):
        predictions = self.predict_proba(arr)
        result = []
        for pred in predictions:
            labels = [l for p, l in zip(pred, CLASSES) if p >= threshold]
            result.append(labels)
        return result



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
    e2e = E2EClassifier(vp, clf)
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
