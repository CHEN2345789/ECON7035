from clean import clean
from train import train

if __name__ == '__main__':
    import pickle

    print("cleaning data")
    data = clean("house_price.csv")

    print("training model")
    metrics, model = train(data)
    print(metrics)

    with open('model.pkl', 'wb') as out:
        pickle.dump(model, out)


