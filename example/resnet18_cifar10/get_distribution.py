import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization


os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


tmp = 0
def basicBlock(block_inputs, num_fiters, strides=1):
    x = Conv2D(filters=num_fiters, kernel_size=3, strides=strides, padding='same', activation='relu')(block_inputs)
    x = BatchNormalization()(x)

    x = Conv2D(filters=num_fiters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    if strides != 1:
        global tmp
        block_inputs = Conv2D(num_fiters, (1, 1), strides=strides, name='1x1_' + str(tmp))(block_inputs)
        tmp += 1

    block_output = layers.add([x, block_inputs])
    return block_output


def buildBlock(x, filter_num, block_num, strides):
    x = basicBlock(x, filter_num, strides)
    for _ in range(1, block_num):
        x = basicBlock(x, filter_num, strides=1)
    return x


def ResNet(shape, layer_dim, class_num):
    img_input = tf.keras.Input(shape=shape, name='img_input')
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(img_input)
    x = BatchNormalization()(x)

    x = buildBlock(x, 64, layer_dim[0], strides=1)
    x = buildBlock(x, 128, layer_dim[1], strides=2)
    x = buildBlock(x, 256, layer_dim[2], strides=2)
    x = buildBlock(x, 512, layer_dim[3], strides=2)
    x = layers.GlobalAveragePooling2D()(x)
    x = Dense(class_num, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(img_input, x, name='resnet')
    return model


def BuildResNetI(shape, name, nb_clssses=100):
    nets = {'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3]}
    return ResNet(shape, nets[name], nb_clssses)


def quant(weight):
    """
    Fake Quantization, referred to TFApprox (https://github.com/ehw-fit/tf-approximate)
    """
    tensor = tf.convert_to_tensor(weight)
    tensor_min = tf.math.reduce_min(tensor)
    tensor_max = tf.math.reduce_max(tensor)
    tensor = tf.quantization.fake_quant_with_min_max_vars(tensor, tensor_min, tensor_max, num_bits=8)
    weight = tensor.numpy()
    weight = np.array(weight)
    # Quantization untrain weights
    max = np.max(weight)
    min = np.min(weight)
    scale = (max - min) / 255.

    zero_point_from_min = 0 - min/scale
    if zero_point_from_min < 0:
        nudged_zero_point = 0
    elif zero_point_from_min > 255:
        nudged_zero_point = 255.
    else:
        nudged_zero_point =  zero_point_from_min
    nudged_min = (0 - nudged_zero_point) * scale

    weight = (weight - nudged_min) / scale
    return weight, scale


def cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    x_train /= 255
    x_test /= 255
    shape = (32, 32, 3)
    nb_classes = 10
    return x_train, y_train, x_test, y_test, shape, nb_classes


def getDistribution():
    x_train, y_train, x_test, y_test, shape, nb_classes = cifar10()
    model = BuildResNetI(shape=shape, name='resnet18', nb_clssses=nb_classes)
    model.load_weights('resnet18_cifar10_weights.h5')

    # Random select images
    random_indices = np.random.choice(50000, size=10000, replace=False)
    x_train_part = np.array(x_train)[random_indices]
    x_train_part = tf.convert_to_tensor(x_train_part)

    prob_x = np.zeros(256)
    freq_w = np.zeros(256)
    bins = np.arange(-0.5, 256, 1)

    for layer in model.layers:
        layer_name = layer.name
        if('conv' in layer_name):
            if layer_name == 'conv2d':
                stride = 1
            elif int(layer_name.strip().split('_')[-1]) in [5, 9, 13]:
                stride = 2
            else:
                stride = 1
            print(layer_name)
            tf.keras.backend.clear_session()
            # Get weights
            weights, bias = layer.get_weights()
            # Quantization
            weights, scale_w = quant(weights)
            # Get weights frequency
            hist_w, bin_edges = np.histogram(weights.flatten(), bins=bins, density=False)
            freq_w += hist_w
            # Get input activations
            intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=layer.input)
            input_activation = intermediate_layer_model.predict(x_train_part)
            # Padding
            if(stride == 1):
                input_activation = np.pad(input_activation, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
            else:
                input_activation = np.pad(input_activation, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='constant')
            # Quantization
            input_activation, scale_a = quant(input_activation)
            # Get activation probability
            hist_x, bin_edges = np.histogram(input_activation.flatten(), bins=bins, density=False)
            prob_x += hist_x
            
    freq_w = freq_w / np.sum(freq_w)
    prob_x = prob_x / np.sum(prob_x)
    dist = np.outer(prob_x, freq_w)

    # Save distribution
    with open('resnet18_cifar10_dist.txt', 'w') as f:
        for i in dist:
            for j in i:
                f.write(str(j)+'\n')


if __name__ == '__main__':
    getDistribution()