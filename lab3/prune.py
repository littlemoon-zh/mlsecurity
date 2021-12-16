'''
python3 eval.py data/cl/valid.h5 data/bd/bd_valid.h5 models/bd_net.h5

Clean Classification accuracy: 98.64899974019225
Attack Success Rate: 100.0

prune:
    2%  45
    5%  47  97.7
    10% 52

'''
import h5py
from tensorflow import keras
import numpy as np
from keras.models import Model


# copy from eval.py
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    return x_data, y_data


def main():
    # load data
    cl_x_test, cl_y_test = data_loader('./data/cl/valid.h5')
    # load model
    model = keras.models.load_model('models/bd_net.h5')
    # go through the model
    predict = model.predict(cl_x_test)

    # print(model)
    print(model.summary())

    conv_3 = Model(inputs=model.input, outputs=model.get_layer('conv_3').output)
    activations = conv_3.predict(cl_x_test)

    # print(activations)
    print(activations.shape)

    # print(activations.mean(axis=(0, 1, 2)))
    print(activations.mean(axis=(0, 1, 2)).shape)

    sorted = np.argsort(activations.mean(axis=(0, 1, 2)))

    print(sorted)

    print(model.get_layer('conv_3'))
    print(model.get_layer('conv_3').get_weights()[0].shape)
    print(model.get_layer('conv_3').get_weights()[1].shape)
    print(len(model.get_layer('conv_3').get_weights()))

    prune_mask = np.zeros(60, dtype=bool)

    p = 0
    for idx in sorted:
        prune_mask[idx] = True
        a, b = model.get_layer('conv_3').get_weights()
        a[:, :, :, prune_mask] = 0
        b[prune_mask] = 0
        print(prune_mask)
        model.get_layer('conv_3').set_weights(weights=(a, b))
        p += 1
        filename = f'models/prune{p}.h5'
        model.save(filename)


if __name__ == '__main__':
    main()
