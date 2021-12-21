import tensorflow.keras as keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from lab3.eval import data_loader


def method_1():
    clean_data_filename = 'data/cl/valid.h5'
    poisoned_data_filename = 'data/bd/bd_valid.h5'

    path_bd = 'models/bd_net.h5'
    bd_model = keras.models.load_model(path_bd)
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    shift = cl_x_test[0]
    num_shift = 20
    for x in cl_x_test[1:20]:
        shift += x

    import collections

    pred = bd_model.predict(cl_x_test)

    shift = shift / num_shift

    return
    for i, x in enumerate(cl_x_test):
        cl_x_test[i] = x + shift
    for i, x in enumerate(bd_x_test):
        bd_x_test[i] = x + shift

    y1 = bd_model.predict(cl_x_test)
    y2 = bd_model.predict(bd_x_test)
    print(min([v.max() for v in y1]))
    print(max([v.max() for v in y1]))
    print(min([v.max() for v in y2]))
    print(max([v.max() for v in y2]))
    data1 = [np.argmax(v) for v in y1]
    data2 = [v.max() for v in y2]
    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
    plt.figure()
    # Plot

    plt.hist(data1, **kwargs, color='r', label='Ideal')
    # plt.hist(data2, **kwargs, color='g', label='Fair')
    plt.show()
    print('debug')


def method_2():
    clean_data_filename = 'data/cl/valid.h5'
    poisoned_data_filename = 'data/bd/bd_valid.h5'

    path_bd = 'models/bd_net.h5'
    bd_model = keras.models.load_model(path_bd)
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)
    import collections
    import random
    cls2img = collections.defaultdict(list)
    for i, cls in enumerate(cl_y_test):
        cls2img[int(cls)].append(i)
    for k, v in cls2img.items():
        print(k, v)

    def predict(x):
        newxs = []
        origianl = bd_model.predict()

        for k, v in cls2img.items():
            newxs.append(x + cl_x_test[random.choice(v)])
        newxs = np.array(newxs)
        ps = bd_model.predict(newxs)
        res = [np.argmax(p) for p in ps]

        prob = collections.Counter(res)

        entropy = 0
        for k, v in prob.items():
            p = v / len(newxs)
            entropy += -p * np.log2(p)

        if entropy > 5:
            return

    for x in bd_x_test:
        predict(x)
    print(cl_x_test.shape)


if __name__ == '__main__':
    method_2()
