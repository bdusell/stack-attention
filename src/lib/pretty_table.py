import shutil

def align(table, min_col_width=4, max_table_width=None, cut_middle=True,
        print=print):
    if max_table_width is None:
        max_table_width = shutil.get_terminal_size().columns
    table = [
        [str(item) for item in row]
        for row in table
    ]
    num_cols = max(map(len, table))
    for row in table:
        while len(row) < num_cols:
            row.append('')
    widths = [
        max(max(_ansi_len(row[j]) for row in table), min_col_width)
        for j in range(num_cols)
    ]
    for row in table:
        if max_table_width is None:
            for cell, w in zip(row, widths):
                print(_ansi_pad(cell, w), end='')
            print()
        else:
            row_str = ' '.join(
                _ansi_pad(cell, w)
                for cell, w in zip(row, widths)
            )
            row_str_len = _ansi_len(row_str)
            if row_str_len > max_table_width:
                if cut_middle:
                    mid = max_table_width // 2
                    left = mid - 1
                    right = row_str_len - (mid - 2)
                    print('{}...{}'.format(
                        _ansi_truncate_end(row_str, left),
                        _ansi_truncate_start(row_str, right)))
                else:
                    print('{}...'.format(
                        _ansi_truncate_end(row_str, max_table_width - 3)))
            else:
                print(row_str)

_OUTSIDE, _SAW_ESC, _INSIDE = range(3)

def _analyze_ansi_chars(s):
    curr_mode = '0'
    state = _OUTSIDE
    for i, c in enumerate(s):
        if state == _OUTSIDE:
            if c == ESC:
                state = _SAW_ESC
            else:
                yield i, c, curr_mode
        elif state == _SAW_ESC:
            if c == '[':
                state = _INSIDE
                curr_mode = ''
            else:
                state = _OUTSIDE
                yield i, c, curr_mode
        else:
            if c == 'm':
                state = _OUTSIDE
            else:
                curr_mode += c

def _ansi_len(s):
    n = 0
    for _ in _analyze_ansi_chars(s):
        n += 1
    return n

def _ansi_truncate_end(s, index):
    n = 0
    prev_i = 0
    prev_mode = '0'
    for i, c, mode in _analyze_ansi_chars(s):
        if n < index:
            n += 1
            prev_i = i
            prev_mode = mode
        else:
            break
    result = s[:prev_i + 1]
    if prev_mode != '0':
        result += '{}[0m'.format(ESC)
    return result

def _ansi_truncate_start(s, index):
    n = 0
    for i, c, mode in _analyze_ansi_chars(s):
        if n < index:
            n += 1
        else:
            break
    else:
        i = len(s)
        mode = '0'
    result = s[i:]
    if mode != '0':
        result = '{}[{}m'.format(ESC, mode) + result
    return result

def _ansi_split(s, index):
    n = 0
    left_i = 0
    left_mode = '0'
    for i, c, mode in _analyze_ansi_chars(s):
        if n < index:
            n += 1
            left_i = i
            left_mode = mode
        else:
            right_i = i
            right_mode = mode
            break
    else:
        right_i = len(s)
        right_mode = left_mode
    left = s[:left_i + 1]
    if left_mode != '0':
        left += '{}[0m'.format(ESC)
    right = s[right_i:]
    if right_mode != '0':
        right = '{}[{}m'.format(ESC, right_mode) + right
    return left, right

def _ansi_pad(s, n):
    ansi_len = _ansi_len(s)
    pad_len = max(0, n - ansi_len)
    return s + ' ' * pad_len

ESC = '\x1b'

def color(s, code):
    return '{}[{}m{}{}[0m'.format(ESC, code, s, ESC)

BLACK = 30
RED = 31
GREEN = 32
YELLOW = 33
BLUE = 34
MAGENTA = 35
CYAN = 36
WHITE = 37

def green(s):
    return color(s, 32)

def red(s):
    return color(s, 31)
