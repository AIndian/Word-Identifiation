import os
from flask import Flask, request, Response, render_template,jsonify,redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

app = Flask(__name__)


fulllet_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',
             16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z',26: 'a', 27: 'b', 28: 'c', 29: 'd', 30: 'e',
            31: 'f', 32: 'g', 33: 'h', 34: 'i', 35: 'j', 36: 'k', 37: 'l', 38: 'm', 39: 'n', 40: 'o', 41: 'p', 42: 'q', 43: 'r',
            44: 's', 45: 't', 46: 'u', 47: 'v', 48: 'w', 49: 'x', 50: 'y', 51: 'z', 52: '0', 53: '1', 54: '2', 55: '3', 56: '4',
            57: '5', 58: '6', 59: '7', 60: '8', 61: '9'}

let_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',
             16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

#For word model

word_dict = {0: 'last', 1: 'Ned', 2: 'bhoy', 3: 'bide'}

size = 28

print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('base_letter_model.h5')
global graph
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (size,size))
    my_image_arr= image.img_to_array(my_image_re)
    my_image_arr /= 255
    img = my_image_arr.flatten().reshape(-1,size,size,1)
    img = 1-img
    #Step 3
    with sess.as_default():
        with sess.graph.as_default():
            probabilities = model.predict_classes(img)
            print(f"Predicted outcome: {probabilities}")
            #Step 4
            outcome = let_dict[probabilities[0]]
            predictions = {"class1":outcome
                  }
    #Step 5
    return render_template('predict.html', predictions=predictions)


if __name__ == '__main__':
    app.debug = True
    app.run()