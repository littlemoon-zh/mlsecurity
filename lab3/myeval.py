from tensorflow import keras
def eval():
    model = keras.models.load_model('models/bd_net.h5')
    pruned = keras.models.load_model('models/pruned40.h5')



if __name__ == '__main__':
    pass
