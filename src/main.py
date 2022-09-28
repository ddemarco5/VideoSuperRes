"""

"""
import sys
import argparse

import src.util as util
import src.train as train
import src.options as options


def main(argv):

    parser = argparse.ArgumentParser()
    args = options.parse(parser, argv)

    util.print_args(parser, args)

    t = train.Train(args)
    t.train()


if __name__ == '__main__':
    main(sys.argv)

