import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("house_price.csv")
df = df.drop(labels='school', axis=1)

X = df.drop(labels='price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print(f' R2:{model.score(X_test, y_test):f}')
print(f'MAE:{mean_absolute_error(y_test, model.predict(X_test)):f}')

with open('model.pkl', 'wb') as out:
    pickle.dump(model, out)

