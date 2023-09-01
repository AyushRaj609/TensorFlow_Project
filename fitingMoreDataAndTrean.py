import tensorflow as tf
import tensorflow.python.keras.callbacks
from tensorflow import keras
# from keras.utils import vis_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X = tf.range(-100,100,4)
y = X + 10
print(X)
print(y)

plt.plot(X,y)
# plt.show()

plt.scatter(X,y)
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]
plt.figure(figsize=(10,7))
plt.scatter(X_train,y_train,c="b",label="Training Data")
plt.scatter(X_test,y_test,c="g",label="Testing Data")
plt.legend()
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# model.fit(X_train,y_train,epochs=100)

tf.random.set_seed(42)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()
model.fit(X_train,y_train,epochs=100,verbose=1)

# plot_model(model=model,show_shapes=True)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[1],name="input_layer"),
    tf.keras.layers.Dense(1,name="output_layer")
],name="model_1")

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(X_train,y_train,epochs=100,verbose=1)
model.summary()

y_pred=model.predict(X_test)
y_pred=tf.constant(y_pred)
y_pred=tf.squeeze(y_pred)

print(y_pred)
print(y_test)

def plot_prediction(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_pred):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",label="t\Training data")
    plt.scatter(test_data,test_labels,c="g",label="Testing data")
    plt.scatter(test_data,predictions,c="r",label="Predictions")
    plt.legend()
    return plt.show()

# plot_prediction()


model.evaluate(X_test,y_test)

mae=tf.metrics.mean_absolute_error(y_true=y_test,
                                   y_pred=tf.squeeze(y_pred))
print(mae)

mse=tf.metrics.mean_squared_error(y_true=y_test,
                                  y_pred=tf.squeeze(y_pred))
print(mse)

def mae(y_true,y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,
                                   y_pred=y_pred)
def mse(y_true,y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,
                                  y_pred=y_pred)

y_preds_1=model.predict(X_test)
plot_prediction(predictions=y_preds_1)

mae_1=mae(y_test,y_preds_1)
mse_1=mse(y_test,y_preds_1)
print(mae_1,mse_1)

model_2=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model_2.fit(X_train,y_train,epochs=100)

y_pred_2=model_2.predict(X_test)
plot_prediction(predictions=y_pred_2)

mae_2=mae(y_test,y_pred_2)
mse_2=mse(y_test,y_pred_2)
print(f"mae: {mae_2}\nmse: {mse_2}")

model_3=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])

model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

model_3.fit(X_train,y_train,epochs=500)

y_pred_3=model_3.predict(X_test)
plot_prediction(predictions=y_pred_3)

# y_test=tf.squeeze(y_test)
# print(y_test)

mae_3=mae(y_test,y_pred_3)
mse_3=mse(y_test,y_pred_3)
print(f"mae: {mae_3}\nmse: {mse_2}")

model_result=[["model"]]