import sys
import re

STARTS_WITH_WHITESPACE_RE = re.compile(r'^\s+')

saw_tree = False
for line in sys.stdin:
    line = line.rstrip()
    if line:
        m = STARTS_WITH_WHITESPACE_RE.match(line)
        if m is None:
            if saw_tree:
                print()
            else:
                saw_tree = True
            print(line, end='')
        else:
            assert saw_tree
            print(' ', end='')
            print(line.lstrip(), end='')
if saw_tree:
    print()
