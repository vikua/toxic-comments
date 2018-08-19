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
    assert vocab_processor.max_seq_len == 13
    assert vocab_processor.vocab == [
        '<unk>', '!', "'m", ',', '.', '?', 'about', 'actual', 'background', 
        'care', 'colour', "d'aww", 'edit', 'edits', 'fan', 'formatting', 'hardcore', 
        'he', 'hey', 'i', 'info', 'made', 'man', 'matches', 'metallica', 'more', 'my', 
        'not', 'really', 'reverted', 'seemingly', 'seems', 'stuck', 'than', 'the', 'this', 
        'to', 'trying', 'under', 'username', 'war', 'were', 'why', 'with',
    ]
    assert vocab_processor.word_to_index == {
        '<unk>': 1, '!': 2, "'m": 3, ',': 4, '.': 5, '?': 6, 'about': 7, 
        'actual': 8, 'background': 9, 'care': 10, 'colour': 11, "d'aww": 12, 
        'edit': 13, 'edits': 14, 'fan': 15, 'formatting': 16, 'hardcore': 17, 
        'he': 18, 'hey': 19, 'i': 20, 'info': 21, 'made': 22, 'man': 23, 'matches': 24, 
        'metallica': 25, 'more': 26, 'my': 27, 'not': 28, 'really': 29, 'reverted': 30, 
        'seemingly': 31, 'seems': 32, 'stuck': 33, 'than': 34, 'the': 35, 'this': 36, 
        'to': 37, 'trying': 38, 'under': 39, 'username': 40, 'war': 41, 'were': 42, 
        'why': 43, 'with': 44
    }

def test_transform(vocab_processor, sentences):
    expected = [
        [1, 35, 14, 22, 39, 27, 40,  1,  1,  1, 42, 30,  6],
        [1,  2,  1, 24, 36,  9, 11,  1,  3, 31, 33, 44,  5],
        [1, 23,  4,  1,  3, 29, 28, 38, 37, 13, 41,  5,  0],
        [1, 32, 37, 10, 26,  7, 35, 16, 34, 35,  8, 21,  5]
    ] 
    x = vocab_processor.transform(sentences)
    assert np.array_equal(np.array(expected), x) 