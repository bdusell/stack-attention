import typing

def levenshtein_distance(source: typing.Sequence, target: typing.Sequence) -> int:
    """Compute the minimum edit distance, aka Levenshtein distance, between a
    source and target sequence. There are three operations: insert, replace,
    and delete, each with a weight of 1.

    Source: Speech and Language Processing, 2nd edition (Jurafsky and Martin, 2009).
    """
    n = len(target)
    m = len(source)
    # The value of prev_row[i] after iteration j is the minimum edit
    # distance from source[:j] to target[:i].
    # This implementation stores just two rows of the table at a time; it isn't
    # necessary to hold the whole table in memory at once.
    prev_row = range(n+1)
    for j in range(1, m+1):
        curr_row = [j]
        for i in range(1, n+1):
            curr_row.append(min(
                # Insert
                curr_row[i-1] + 1,
                # Replace
                prev_row[i-1] + int(source[j-1] != target[i-1]),
                # Delete
                prev_row[i] + 1
            ))
        prev_row = curr_row
    return prev_row[-1]
