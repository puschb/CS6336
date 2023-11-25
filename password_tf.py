import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
from password_data import get_password_data

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

if __name__ == "__main__":
    # builder = tfds.builder("rock_you")
    # x_train = builder.as_dataset(split="train[:50%]")
    # x_test = builder.as_dataset(split="train[50%:]")
    data = get_password_data()
