import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# data import
insurance=pd.read_csv("insurance.csv")
data=pd.DataFrame(insurance)
# print(data)
# print(data.info())

# print(data["smoker"], data["age"])

insurance_one_hot=pd.get_dummies(data)
print(insurance_one_hot.head())

X=insurance_one_hot.drop("charges",axis=1)
y=insurance_one_hot["charges"]

# print(X.head())
# print(y.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(len(X),len(X_train),len(X_test))

# print(X_test)
tf.random.set_seed(42)

# create a model
insurance_model=tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# compole the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"])

# fit the model
# insurance_model.fit(X_train,y_train,epochs=100)

# check the result
print(insurance_model.evaluate(X_test,y_test))

print(y_train.median(),y_train.mean())

tf.random.set_seed(42)
insurance_model_2=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

insurance_model_2.fit(X_train,y_train,epochs=100,verbose=1)
insurance_model_2.evaluate(X_test,y_test)


tf.random.set_seed(42)
# create a model
insurance_model_3=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])
# fit the model
history=insurance_model_3.fit(X_train,y_train,epochs=200)

insurance_model_3.evaluate(X_test,y_test)

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()