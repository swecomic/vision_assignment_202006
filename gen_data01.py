from keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def get_CIFAR10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0  # sCaling
    y_test_label = y_test
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


