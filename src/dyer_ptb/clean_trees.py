import fileinput

import bllipparser

def main():
    for line in fileinput.input():
        tree = bllipparser.Tree(line)
        for subtree in tree.all_subtrees():
            subtree.label_suffix = ''
        print(tree)

if __name__ == '__main__':
    main()
