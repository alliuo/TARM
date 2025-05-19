import sys

tfapprox_path = sys.argv[1]
lut_path = sys.argv[2]
gpu_index = sys.argv[3]

sys.path.append(tfapprox_path)

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization
from tf2.python.keras.layers.fake_approx_convolutional import FakeApproxConv2D


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def basicBlock(block_inputs, num_fiters, strides=1, src=""):
    x = FakeApproxConv2D(filters=num_fiters, kernel_size=3, strides=strides, padding='same', activation='relu', mul_map_file=src)(block_inputs)
    x = BatchNormalization()(x)

    x = FakeApproxConv2D(filters=num_fiters, kernel_size=3, strides=1, padding='same', activation='relu', mul_map_file=src)(x)
    x = BatchNormalization()(x)
    if strides != 1:
        block_inputs = Conv2D(num_fiters, (1, 1), strides=strides)(block_inputs)

    block_output = layers.add([x, block_inputs])
    return block_output


def buildBlock(x, filter_num, block_num, strides, src=""):
    x = basicBlock(x, filter_num, strides, src=src)
    for _ in range(1, block_num):
        x = basicBlock(x, filter_num, strides=1, src=src)
    return x


def AppResNet(shape, layer_dim, class_num, src=""):
    img_input = tf.keras.Input(shape=shape, name='img_input')
    x = FakeApproxConv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', mul_map_file=src)(img_input)
    x = BatchNormalization()(x)

    x = buildBlock(x, 64, layer_dim[0], strides=1, src=src)
    x = buildBlock(x, 128, layer_dim[1], strides=2, src=src)
    x = buildBlock(x, 256, layer_dim[2], strides=2, src=src)
    x = buildBlock(x, 512, layer_dim[3], strides=2, src=src)
    x = layers.GlobalAveragePooling2D()(x)
    x = Dense(class_num, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(img_input, x, name='resnet')
    return model


def appBuildResNetI(shape, name, nb_clssses=100, src=""):
    nets = {'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3]}
    return AppResNet(shape, nets[name], nb_clssses, src)


def cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.squeeze(y_test, axis=1)
    x_test /= 255
    shape = (32, 32, 3)
    nb_classes = 10
    return x_test, y_test, shape, nb_classes


def TestApp(lut_path):
    x_test, y_test, shape, nb_classes = cifar10()
   
    model = appBuildResNetI(shape=shape, name='resnet18', nb_clssses=nb_classes, src=lut_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    tf.keras.backend.clear_session()
    model.load_weights('resnet18_cifar10_weights.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]


if __name__ == '__main__':
    accuracy = TestApp(lut_path)
    print(accuracy)
