from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import os
import shutil


def dataset_loader():
    # data preprocessing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Nomalize the images
    x_train = (x_train / 255)
    x_test = (x_test / 255)
    # one_hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return (x_train, y_train), (x_test, y_test)


def create_model():
    # parameters for LSTM
    nb_lstm_outputs = 30    # 输出神经元个数
    nb_time_steps = 28    # 时间序列的长度
    nb_input_vectors = 28    # 每个输入序列的向量维度

    # building model
    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(
        nb_time_steps, nb_input_vectors)))
    model.add(Dense(10, activation='softmax'))
    return model


def train(model, epochs=20, saved_model='mnist_lstm', logs='logs'):

    # compile:loss, optimizer, metrics
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        verbose=1,
        callbacks=[EarlyStopping(patience=2), TensorBoard(logs)]
    )
    model.save(saved_model)


def inference(model_name, x_test, y_test, batch_size=128):
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    return score


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = dataset_loader()
    saved_model = 'mnist_lstm'
    model = create_model()
    if not os.path.exists(saved_model):
        train(model, epochs=30, saved_model=saved_model)
    inference(saved_model, x_test, y_test)
