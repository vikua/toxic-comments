import multiprocessing as mp
from functools import partial

import numpy as np
import tensorflow as tf 

from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


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
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/toxic', save_best_only=True)

        def auc_roc(y_true, y_pred): 
            value, update_op = tf.metrics.auc(y_true, y_pred)

            metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

            for v in metric_vars: 
                tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

            with tf.control_dependencies([update_op]): 
                value = tf.identity(value)
                return value

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


class ToxicCommentsClassifier(object):

    def __init__(self, classifier, vocab_processor):
        self.classifier = classifier
        self.vocab_processor = vocab_processor

    def fit(self, X_train, y_train, X_test, y_test, **kwargs): 
        epochs = kwargs.pop('epochs', 5)
        batch_size = kwargs.pop('batch_size', 128)

        X_train = self.vocab_processor.fit_transform(X_train)
        X_test = self.vocab_processor.transform(X_test)

        classifier.fit(X_train, y_train, X_test, y_test, 
                       epochs=epochs, batch_size=batch_size)

    def predict(self, X): 
        X = self.vocab_processor.transform(X)
        predict = self.classifier.predict(X)