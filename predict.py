from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
 
model = load_model('mnist_model.h5')

# load an image and predict the class
def predict(file_path):
    img = load_img(file_path, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    # predict the class
    digit = model.predict_classes(img)
    return str(digit[0])
 
