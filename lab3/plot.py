import numpy as np
import matplotlib.pyplot as plt
from eval import main

if __name__ == '__main__':
    frac = []
    accs = []
    asrs = []
    for i in range(1, 61):
        modelname = f'models/prune{i}.h5'
        clean = './data/cl/test.h5'
        bad = './data/bd/bd_test.h5'

        acc, asr = main(modelname, clean, bad)

        frac.append(i / 60)
        accs.append(acc)
        asrs.append(asr)
        print(i, acc, asr)
        print(i)

    plt.plot(frac, accs, label='acc')
    plt.plot(frac, asrs, label='asr')
    plt.legend()

    plt.savefig('res.png')
    plt.show()
