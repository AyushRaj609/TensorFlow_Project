import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# data import
insurance=pd.read_csv("insurance.csv")
data=pd.DataFrame(insurance)

ct = make_column_transformer(
    (MinMaxScaler(),["age","bmi","children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)

# chreate x and y
X=insurance.drop("charges",axis=1)
y=insurance["charges"]

# build train and test set
# X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# fit the column
ct.fit(X_train)

X_train_normal=ct.transform(X_train)
X_test_normal=ct.transform(X_test)

print(X_train_normal[0])
