import numpy as np
import tensorflow as tf 

from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences


class NNClassifier(object):

    def __init__(self, num_clases, debug=False, **kwargs):
        self._num_classes = num_clases
        self._debug = debug

        self._model = None

        self._max_seq_len = None
        self._vocab = None
        self._word_to_index = None

        # hyperparameters
        self._embedding_dim = kwargs.pop('embedding_dim', 100)
        self._dropout = 1.0 - kwargs.pop('dropout_keep_prob', 1.0)
        self._lstm_units = kwargs.pop('lstm_units', 256)

    def build_model(self): 
        inputs = tf.keras.Input(shape=(self._max_seq_len, ), dtype='int32', name='inputs')
        embed = tf.keras.layers.Embedding(len(self._vocab), 
                                          self._embedding_dim,
                                          trainable=True)(inputs)
        dropout = tf.keras.layers.Dropout(self._dropout)(embed)
        lstm = tf.keras.layers.LSTM(self._lstm_units)(dropout)

        predictions = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')(lstm)

        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        return model

    @property
    def model(self):
        if not self._model:
            self._model = self.build_model()
        return self._model

    def fit(self, data, labels, **kwargs): 
        epochs = kwargs.pop('epochs', 5)
        batch_size = kwargs.pop('batch_size', 128)

        print('Building vocabulary and vectorizing data...')
        self._build_vocabulary(data)
        X = self._vectorize(data)
        print('Done!')


        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                                    patience=2, 
                                                                    min_lr=0.0001)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/toxic', save_best_only=True)

        def auc_roc(y_true, y_pred): 
            value, update_op = tf.metrics.auc(y_true, y_pred)

            metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

            for v in metric_vars: 
                tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

            with tf.control_dependencies([update_op]): 
                value = tf.identity(value)
                return value

        self.model.compile(optimizer=tf.train.AdamOptimizer(0.01), 
                           loss='binary_crossentropy', 
                           metrics=['accuracy', auc_roc])

        history = self.model.fit(X, labels, 
                                 validation_split=0.25,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stopping, reduce_learning_rate, 
                                            model_checkpoint])
        return history

    def predict(self, data): 
        X = self._vectorize(data)
        predictions = self.model.predict(X)

    def _build_vocabulary(self, data): 
        """ Building vocabulary and mapping from word to index and requred for 
        transforming words into embeddings

        Parameters
        ----------
        data : list of sorted
            List of raw sentences
        """
        self._max_seq_len = max([len(sentence) for sentence in data])

        words = set([word for sentence in data for word in sentence])

        self._vocab = ['<unk>'] + sorted(words)
        self._word_to_index = {word: i + 1 for i, word in enumerate(self._vocab)}

    def _vectorize_sentence(self, sentence): 
        array = np.empty_like(sentence, dtype=np.int32)
        for i, word in enumerate(sentence): 
            if word not in self._vocab: 
                word = '<unk>'
            array[i] = self._word_to_index[word]
        return array

    def _vectorize(self, data):
        assert self._vocab, 'Vocabulary is not initialized'
        assert self._word_to_index, 'Vocabulary is not initialized'

        sentences = [
            self._vectorize_sentence(word_tokenize(sentence)) 
            for sentence in data
        ]
        sentences = pad_sequences(sentences, maxlen=self._max_seq_len, value=0, padding='post')

        return sentences
