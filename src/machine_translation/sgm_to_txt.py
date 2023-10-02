import argparse
import pathlib
import sys

import bs4

def parse_xml_file(fin):
    return bs4.BeautifulSoup(fin, 'html.parser')

def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    doc = parse_xml_file(sys.stdin)
    for el in doc.find_all('seg'):
        line = el.string
        if line is None or '\n' in line:
            raise ValueError(f'invalid <seg> element: {el}')
        print(line)

if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        pass
