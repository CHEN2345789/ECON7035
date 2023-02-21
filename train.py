from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train(df):
    X = df.drop(labels='price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = {
        'R2': model.score(X_test, y_test),
        'MAE': mean_absolute_error(y_test, model.predict(X_test))
    }
    return metrics, model


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Cleaned data file (CSV)')
    parser.add_argument('output', help='Model file (pickle)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    metrics, model = train(df)
    print(metrics)

    with open(args.output, 'wb+') as out:
        pickle.dump(model, out)
