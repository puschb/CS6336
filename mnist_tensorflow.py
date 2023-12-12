import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)


@keras.saving.register_keras_serializable()
class Convolutional(Model):
    def __init__(self, activation="sigmoid", layer_count=2, layer_sizes=(32, 64)):
        super(Convolutional, self).__init__()
        self.network = keras.Sequential([keras.Input(shape=(28, 28, 1))])
        for i in range(layer_count):
            if i != layer_count - 1:
                self.network.add(
                    layers.Conv2D(layer_sizes[i], (3, 3), activation=activation)
                )
                self.network.add(layers.MaxPooling2D((2, 2)))
            else:
                self.network.add(
                    layers.Conv2D(layer_sizes[i], (3, 3), activation=activation)
                )
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(10))

    def call(self, x):
        return self.network(x)


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Convolutional()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    metrics = [
        "accuracy",
        keras.metrics.Precision(top_k=1),
        # keras.metrics.Recall(),
        # keras.metrics.F1Score(),
        # keras.metrics.AUC(),
    ]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)
