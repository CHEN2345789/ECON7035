import pandas as pd


def clean(input_file):
    df = pd.read_csv(input_file)
    df = df.drop(labels='school', axis=1)
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Data file (CSV)')
    parser.add_argument('output', help='Cleaned data file (CSV)')
    args = parser.parse_args()

    cleaned = clean(args.input)
    cleaned.to_csv(args.output, index=False)
