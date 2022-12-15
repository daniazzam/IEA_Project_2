from flask import Flask, flash, request, redirect, render_template
import base64
from io import BytesIO
from PIL import Image
import joblib
from ImageToFeature import *
from GetBoundingRectange import *
from segmentation import segment
import tensorflow as tf


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


CNNmodel = tf.keras.models.load_model('./Models/CNNmodel3.h5')
ANNmodelImages = tf.keras.models.load_model('./Models/ANN_Images.h5')
ANNmodelFeatures = tf.keras.models.load_model('./Models/ANN_Features.h5')
Model_M_m = tf.keras.models.load_model('./Models/Model_M_m.h5')
Model_9_g_q = tf.keras.models.load_model('./Models/Model_9_g_q.h5')


def Top1(predictions_proba):
    # predictions_proba = predictions_proba.tolist()
    lea = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    predictions=[]

    # predictions_proba = pd.DataFrame(predictions_proba)
    index1 = predictions_proba.index(max(predictions_proba))
    predictions.append(lea[index1])
    predictions.append(predictions_proba[index1]) 
    predictions_proba[index1] = 0

    return predictions_proba, predictions


def Ensemble(image, feature):
    cnn_pred = CNNmodel.predict(image)
    ann_feature_pred = ANNmodelFeatures.predict(feature)
    ann_image_pred = ANNmodelImages.predict(img_feature)

    final_y_proba = sum = [x + y for (x, y) in zip(cnn_pred, ann_feature_pred)]
    final_y_proba = sum = [x + y for (x, y) in zip(final_y_proba, ann_image_pred[0])]

    predictions = [value / 2 for value in final_y_proba]
    # predcitions = [round(member, 2) for member in final_y_proba]
    predictions_proba, predictions= Top1(predictions)

    if predictions[0]=='M' or predictions[0] == 'm':
        output = Model_M_m.predict(feature)
        index = output.argmax(axis = 1)
        if index == 0:
            return 'M'
        else:
            return 'm'
    if predictions[0]=='9' or predictions[0] == 'g' or predictions[0] == 'q':
        output = Model_9_g_q.predict(feature)
        index = output.argmax(axis = 1)
        if index == 0:
            return '9'
        elif index == 1:
            return 'g'
        else:
            return 'q'
    print('Hi'+str(predictions))
    return predictions[0]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    for filename in os.listdir('static/uploads/drawing'):
        if 'canvasImage' not in filename:
            os.remove(f'static/uploads/drawing/{filename}')
    inputType = request.form['inputType']
    if inputType == 'drawing':
        model = request.form['model'] 
        letterList = segment('static/uploads/drawing/canvasImage.png')
        featureList = {}
        for idx, letter in enumerate(letterList):
            cv2.imwrite(f'static/uploads/drawing/canvas{idx}.png', letter)
            featureList[f'canvas{idx}'] = getFeatures(f'static/uploads/drawing/canvas{idx}.png')
        predictions_list = []
        if model == 'cnn':
            for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                img = tf.keras.utils.load_img(data_dir,target_size = (28, 28))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array,0)
                predictions_proba, predictions= Top1(CNNmodel.predict(img_array)[0].tolist())
                predictions_list.append(predictions[0])
        elif model == 'ann-features':
            for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                features = featureList[letter]
                ann_feature_pred = ANNmodelFeatures.predict(features)
                predictions_proba, predictions= Top1(ANNmodelFeatures.predict(features)[0].tolist())
                predictions_list.append(predictions[0])
        elif model == 'ann-images':
           for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                IMG = cv2.imread(data_dir)
                IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
                # image_feature = cv2.imread(data_dir)[0].flatten()
                csvFile = {}
                # for idx, bit in enumerate(image_feature):
                #     if bit>0:
                #         image_feature[idx] = 255
                counter = 0
                for row in IMG:
                    for el in row:
                        if el > 0:
                            el = 255
                        csvFile[f'Pixel {counter}'] = el
                        counter += 1
                ann_feature_pred = ANNmodelFeatures.predict(csvFile)
                predictions_proba, predictions= Top1(ANNmodelImages.predict(csvFile)[0].tolist())
                predictions_list.append(predictions[0])
        elif model == 'ensemble':
             for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                img = tf.keras.utils.load_img(data_dir,target_size = (28, 28))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array,0)
                features = featureList[letter]
                predictions_list.append(Ensemble(img_array, features))
        _, _, boundingRectInput = processUserImage('static/uploads/drawing/canvasImage.png')
        cv2.imwrite('static/uploads/canvasBox.png', boundingRectInput)
        return render_template('index.html', outputLetter=predictions_list, model=model, inputImage='static/uploads/canvasBox.png')
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = 'img.png'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        model = request.form['model'] 
        letterList = segment('static/uploads/drawing/canvasImage.png')
        featureList = {}
        for idx, letter in enumerate(letterList):
            cv2.imwrite(f'static/uploads/drawing/canvas{idx}.png', letter)
            featureList[f'canvas{idx}'] = getFeatures(f'static/uploads/drawing/canvas{idx}.png')
        predictions_list = []
        if model == 'cnn':
            for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                img = tf.keras.utils.load_img(data_dir,target_size = (28, 28))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array,0)
                predictions_proba, predictions= Top1(CNNmodel.predict(img_array)[0].tolist())
                predictions_list.append(predictions[0])
        elif model == 'ann-features':
            for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                features = featureList[letter]
                ann_feature_pred = ANNmodelFeatures.predict(features)
                predictions_proba, predictions= Top1(ANNmodelFeatures.predict(features)[0].tolist())
                predictions_list.append(predictions[0])
        elif model == 'ann-images':
           for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                IMG = cv2.imread(data_dir)
                IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
                # image_feature = cv2.imread(data_dir)[0].flatten()
                csvFile = {}
                # for idx, bit in enumerate(image_feature):
                #     if bit>0:
                #         image_feature[idx] = 255
                counter = 0
                for row in IMG:
                    for el in row:
                        if el > 0:
                            el = 255
                        csvFile[f'Pixel {counter}'] = el
                        counter += 1
                ann_feature_pred = ANNmodelFeatures.predict(csvFile)
                predictions_proba, predictions= Top1(ANNmodelImages.predict(csvFile)[0].tolist())
                predictions_list.append(predictions[0])
        elif model == 'ensemble':
             for letter in featureList:
                data_dir = f'static/uploads/drawing/{letter}.png'
                img = tf.keras.utils.load_img(data_dir,target_size = (28, 28))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array,0)
                features = featureList[letter]
                predictions_list.append(Ensemble(img_array, features))
        _, _, boundingRectInput = processUserImage('static/uploads/img.png')
        cv2.imwrite('static/uploads/uploadBox.png', boundingRectInput)
        return render_template('index.html', outputLetter=probas, model=model, inputImage='static/uploads/uploadBox.png')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/draw')
def display_image():
    return render_template('drawing.html')


@app.route('/draw', methods=['GET', 'POST'])
def draw():
    canvasImage = request.form['js_data']
    offset = canvasImage.index(',') + 1
    img_bytes = base64.b64decode(canvasImage[offset:])
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    cv2.imwrite('static/uploads/drawing/canvasImage.png', img)
    return render_template('drawing.html')


if __name__ == "__main__":
    app.run(host='localhost', port=8000,debug=True)
