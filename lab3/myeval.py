import h5py
from tensorflow import keras
import numpy as np
import sys
from matplotlib import image



def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def predict(imgname):
    print(f'read {imgname}')
    img = image.imread(imgname)
    x = img.reshape(1, 55, 47, 3)
    model = keras.models.load_model('models/bd_net.h5')
    pruned = keras.models.load_model('models/prune55.h5')

    model_y = model.predict(x)
    pruned_y = pruned.predict(x)

    cls_1 = np.argmax(model_y[0])
    cls_2 = np.argmax(pruned_y[0])

    if cls_1 == cls_2:
        return cls_1
    else:
        return 1283


if __name__ == '__main__':
    file = str(sys.argv[1])
    print(predict(file))
