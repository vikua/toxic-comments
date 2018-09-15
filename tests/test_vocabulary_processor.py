import numpy as np
import pytest 
from nltk import word_tokenize

from model import VocabularyProcessor


@pytest.fixture
def sentences():
    return [
        "Why the edits made under my username Hardcore Metallica Fan were reverted?",
        "D'aww! He matches this background colour I'm seemingly stuck with.", 
        "Hey man, I'm really not trying to edit war.",
        "He seems to care more about the formatting than the actual info."
    ]


@pytest.fixture
def vocab_processor(sentences): 
    vocab_processor = VocabularyProcessor()
    vocab_processor.fit_transform(sentences)
    return vocab_processor


def test_fit_creates_vocabulary(vocab_processor): 
    assert vocab_processor.max_seq_len == 12
    assert vocab_processor.max_features == 39
    assert vocab_processor.word_index == {
        'the': 1, 'he': 2, "i'm": 3, 'to': 4, 'why': 5, 'edits': 6, 'made': 7, 
        'under': 8, 'my': 9, 'username': 10, 'hardcore': 11, 'metallica': 12, 'fan': 13, 
        'were': 14, 'reverted': 15, "d'aww": 16, 'matches': 17, 'this': 18, 'background': 19, 
        'colour': 20, 'seemingly': 21, 'stuck': 22, 'with': 23, 'hey': 24, 'man': 25, 
        'really': 26, 'not': 27, 'trying': 28, 'edit': 29, 'war': 30, 'seems': 31, 
        'care': 32, 'more': 33, 'about': 34, 'formatting': 35, 'than': 36, 'actual': 37, 
        'info': 38, '<unk>': 39,
    }

def test_transform(vocab_processor, sentences):
    expected = [
        [ 5,  1,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
        [16,  2, 17, 18, 19, 20,  3, 21, 22, 23,  0,  0],
        [24, 25,  3, 26, 27, 28,  4, 29, 30,  0,  0,  0],
        [ 2, 31,  4, 32, 33, 34,  1, 35, 36,  1, 37, 38],
    ]
    x = vocab_processor.transform(sentences)
    assert np.array_equal(np.array(expected), x) 