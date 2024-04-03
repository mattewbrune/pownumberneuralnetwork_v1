import tensorflow as tf
import numpy as np

def activation(x):
    return x
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
aa = []
for i in range(len(a)):
  aa.append(a[i]**2)

x_train = np.array(a)
y_train = np.array(aa)

#defining architecture   Can change Sequential on Sigmoid or Relu
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation="relu"),
])
#reshaping x_train so its 2dimensional
x_train = x_train.reshape(-1, 1)
#compiling model
#sgd - GradientDescent (градиентный спуск)
#mse - MeanSquadError (среднеквадратическая ошибка)
mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=tf.keras.optimizers.experimental.SGD(
    learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())

#training parameters
epochs = 40
batch_size = 1

#Model training
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

#Accuracy check (loss check)
loss = model.evaluate(x_train, y_train)
#print(f"Loss: {loss}")

#Prediction (Prognose)
predictions = model.predict(x_train)
print(f"Predictions: {predictions}")
