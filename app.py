from concurrent.futures import thread
from flask import Flask, render_template, request, redirect

from keras.models import  load_model
import numpy as np
import librosa
from sklearn import preprocessing
from statistics import mode

from waitress import serve

model = load_model("model_emotion_2.h5")
emotions_list = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad'}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    emotion = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            #-----------
            y, sr = librosa.load(file, duration=3, offset=0.1)
            mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            mfcc_max = np.max(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            mfcc_min = np.min(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            xxx = np.concatenate((mfcc_mean,mfcc_max,mfcc_min), axis=0)
            min_max_scaler = preprocessing.MinMaxScaler()
            xx = min_max_scaler.fit_transform(xxx.reshape(-1, 1))
            X=xx.reshape((1,xx.shape[0],xx.shape[1]))

            predictions=model.predict(X,verbose=0)
            max_index_col = np.argmax(predictions, axis=1)
            max_index=mode(max_index_col)
            emotion = emotions_list[max_index]

    #print(file,emotion)    
    return render_template('index.html', emotion=emotion)


#if __name__ == "__main__":
#    app.run(debug=True, threaded=True)

serve(app, host='0.0.0.0',port=7000, threads=1)