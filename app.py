import argparse

import src.preproc as preproc
import src.train as train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('step')
    args = parser.parse_args()

    if args.step == 'preproc':
        preproc.main('./data/raw', "./data/dataset")
    elif args.step == 'train':
        train.main('./data/dataset', "./data/model")
    else:
        print(f'Step "{args.step}" not found.')


