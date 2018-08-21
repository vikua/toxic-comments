import multiprocessing as mp
from functools import partial

import numpy as np
import tensorflow as tf 

from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences


def chunks(lst, chunk_size): 
    for i in range(0, len(lst), chunk_size): 
        yield lst[i:i + chunk_size]


def tokenize_list(lst): 
    return [word_tokenize(s.lower()) for s in lst]


class VocabularyProcessor(object):

    def __init__(self): 
        self._max_seq_len = None
        self._vocab = None
        self._word_to_index = None

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def vocab(self):
        return self._vocab

    @property
    def word_to_index(self):
        return self._word_to_index

    def _execute_parallel(self, data, func): 
        pool = mp.Pool(mp.cpu_count())
        pool_results = pool.map(func, list(chunks(data, 1000)))
        pool.close()
        pool.join()

        results = []
        for result in pool_results:
            results.extend(result)

        return results

    def fit_transform(self, data): 
        print('Building vocabulary...')
        sentences = self._execute_parallel(data, tokenize_list)
        self._max_seq_len = max([len(sentence) for sentence in sentences])

        words = set([word for sentence in sentences for word in sentence])

        self._vocab = ['<unk>'] + sorted(words)
        self._word_to_index = {word: i + 1 for i, word in enumerate(self._vocab)}

        print('Mapping words to indices...')
        sentences = self._execute_parallel(sentences, self._vectorize)

        print('Padding sequences...')
        sentences = pad_sequences(sentences, maxlen=self._max_seq_len, value=0, padding='post')

        print('Done!')
        return sentences

    def transform(self, data): 
        assert self._vocab, 'Vocabulary is not initialized'
        assert self._word_to_index, 'Vocabulary is not initialized'

        sentences = [
            self._vectorize_sentence(word_tokenize(sentence)) 
            for sentence in data
        ]
        sentences = pad_sequences(sentences, maxlen=self._max_seq_len, value=0, padding='post')

        return sentences

    def _vectorize(self, sentences): 
        return [self._vectorize_sentence(sentence) for sentence in sentences]

    def _vectorize_sentence(self, sentence): 
        unk = self._word_to_index['<unk>']
        return np.array([self._word_to_index.get(word, unk) for word in sentence])


class NNClassifier(object):

    def __init__(self, num_clases, vocab_processor, **kwargs):
        self._num_classes = num_clases
        self._vocab_processor = vocab_processor

        self._model = None

        # hyperparameters
        self._embedding_dim = kwargs.pop('embedding_dim', 100)
        self._dropout = 1.0 - kwargs.pop('dropout_keep_prob', 1.0)
        self._lstm_units = kwargs.pop('lstm_units', 128)

    def build_model(self): 
        inputs = tf.keras.Input(shape=(self._vocab_processor.max_seq_len, ), dtype='int32', name='inputs')
        embed = tf.keras.layers.Embedding(len(self._vocab_processor.vocab) + 1, 
                                          self._embedding_dim,
                                          mask_zero=True,
                                          trainable=True)(inputs)
        dropout = tf.keras.layers.Dropout(self._dropout)(embed)
        lstm = tf.keras.layers.LSTM(self._lstm_units)(dropout)

        predictions = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')(lstm)

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

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                                    patience=2, 
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


class ClassifierWrapper(object):

    def __init__(self, classifier, vocab_processor):
        self.classifier = classifier
        self.vocab_processor = vocab_processor

    def fit(self, X_train, y_train, X_test, y_test): 
        pass
    def predict(self, X): 
        pass