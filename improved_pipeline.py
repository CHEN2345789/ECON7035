import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def clean(input_file):
    df = pd.read_csv(input_file)
    df = df.drop(labels='school', axis=1)
    return df


def train(df):
    X = df.drop(labels='price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = {
        'R2': model.score(X_test, y_test),
        'MAE': mean_absolute_error(y_test, model.predict(X_test))
    }
    return metrics, model


if __name__ == '__main__':
    print("cleaning data")
    data = clean("house_price.csv")

    print("training model")
    metrics, model = train(data)
    print(metrics)

    with open('model.pkl', 'wb') as out:
        pickle.dump(model, out)