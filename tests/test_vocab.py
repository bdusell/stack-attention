import pytest

from vocab2 import build_to_int_vocabulary, build_to_string_vocabulary

def build_complex_vocab(builder):
    return (
        builder.content(['red', 'orange', 'yellow']) +
        builder.catchall('unk') +
        builder.reserved(['eos']) +
        builder.content(['alpha', 'beta']) +
        builder.reserved(['bos'])
    )

def test_complex_to_int():
    v = build_to_int_vocabulary(build_complex_vocab)
    assert len(v) == 8
    assert v.to_int('red') == 0
    assert v.to_int('orange') == 1
    assert v.to_int('yellow') == 2
    assert v.to_int('alpha') == 5
    assert v.to_int('beta') == 6
    assert v.to_int('asdf') == 3
    assert v.to_int('dummy') == 3
    assert v.to_int('<unk>') == 3
    assert v.to_int('<eos>') == 3
    assert v.to_int('<bos>') == 3

def test_complex_to_string():
    v = build_to_string_vocabulary(build_complex_vocab)
    assert len(v) == 8
    with pytest.raises(IndexError):
        v.to_string(-1)
    assert v.to_string(0) == 'red'
    assert v.to_string(1) == 'orange'
    assert v.to_string(2) == 'yellow'
    assert v.to_string(3) == '<unk>'
    assert v.to_string(4) == '<eos>'
    assert v.to_string(5) == 'alpha'
    assert v.to_string(6) == 'beta'
    assert v.to_string(7) == '<bos>'
    with pytest.raises(IndexError):
        v.to_string(8)
    assert v.unk_index == 3
    assert v.eos_index == 4
    assert v.bos_index == 7
    with pytest.raises(AttributeError):
        v.asdf_index
