import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

x = np.arange(0, 2*math.pi, .005)
y = np.sin(x)

model = keras.Sequential([
    layers.Dense(20, input_shape=(1,), activation='tanh'),
    layers.Dense(6, activation='tanh'),
    layers.Dense(1, activation='tanh'),
])

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mean_squared_error'])
model.fit(x, y, epochs=50, batch_size=5)

scores = model.evaluate(x, y, verbose=0)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
model.save("arch.h5", include_optimizer=False)
preds = model.predict(x)
plt.plot(x, y, 'b', x, preds, 'r--')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.show()
