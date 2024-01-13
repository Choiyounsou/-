import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


loaded_model = load_model('model_filter.h5')

img_path = 'photo.png'  
img = image.load_img(img_path, target_size=(48, 48), grayscale=True)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

custom = loaded_model.predict(x)

objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
ind = np.argmax(custom[0])
predicted_emotion = objects[ind]

print('Predicted Emotion:', predicted_emotion)
