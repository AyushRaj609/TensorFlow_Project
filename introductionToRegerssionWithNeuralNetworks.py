import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt

X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])
# y=x+10
plt.scatter(X,y)
# plt.show()

#creat a demo tensor for house price prediction problem

house_info = tf.constant(["bedroom","bathroom","garage"])
house_price = tf.constant([939700])
print(house_info,house_price)

input_shape = X.shape
output_shape = y.shape
print(input_shape,output_shape)

print(X[0],y[0])
print(X[1],y[1])
print(X[0].ndim)

#Turn our NumPy arrays into tensors

X = tf.cast(tf.constant(X),dtype=tf.float32)
y = tf.cast(tf.constant(y),dtype=tf.float32)
print(X,y)
# [2,3,4,5,6,7,8,9]
# reshape(3,3)
# [[1,2,3],
# [4,5,6],
# [7,8,9]]
plt.scatter(X,y)
# plt.show()
X = tf.reshape(X, (-1,1))
# print(X)
tf.random.set_seed(42)

# create a model using the Sequestial API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50),
    # tf.keras.layers.Dense(100,activation="relu"),
    # tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(1)
])

# compile the model
model.compile(loss=tf.keras.losses.mae,
          optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=["mae"])

# fit the model
model.fit(X,y,epochs=100)

# print(X,y)

# try and make a prediction using our model
y_pred=model.predict([17.0])
print(y_pred)

# improveing our model
