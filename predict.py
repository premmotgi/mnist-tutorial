from cnvrg import Endpoint
from scipy.misc.pilutil import imread, imresize
import numpy as np
#perform the prediction
from keras.models import load_model
#include custom charts in logging
e = Endpoint()
model = load_model('output/mnist_model.h5')

# load an image and predict the class
def predict(file_path):
    x = imread(file_path, mode='L')
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    #make it the right size
    x = imresize(x,(28,28))
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255
    # predict the class
    out = model.predict(x)
    #log the predicted digit
    e.log_metric("digit", np.argmax(out))
    return str(np.argmax(out))
