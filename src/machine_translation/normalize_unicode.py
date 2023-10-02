import sys
import unicodedata

NORMAL_FORM = 'NFKC'

def main():
    for line in sys.stdin:
        sys.stdout.write(unicodedata.normalize(NORMAL_FORM, line))

if __name__ == '__main__':
    main()
