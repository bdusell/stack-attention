from lib.levenshtein import levenshtein_distance

def test_levenshtein_distance():
    assert levenshtein_distance('intention', 'execution') == 5
