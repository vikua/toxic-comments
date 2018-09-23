import tempfile 
import types 

import tensorflow as tf

from model import NNClassifier, E2EClassifier

import cloudpickle


def make_keras_picklable(): 
    def __getstate__(self): 
        model_str = ''
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tf.keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd: 
            fd.write(state['model_str'])
            fd.flush()
            model = tf.keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = tf.keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
