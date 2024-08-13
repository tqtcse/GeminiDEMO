from flask import Flask, request, jsonify, session
from urllib.parse import unquote
from flask_cors import CORS
import os
import firebase_admin
from firebase_admin import credentials, storage
import requests
import numpy as np
import librosa
import io
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai
from AI import Model_AI


app = Flask(__name__)

in_memory_storage = {}

cors = CORS(app, resources={r"/submit":{"origins":"http://127.0.0.1:5500"},
                            r"/upload":{"origins":"http://127.0.0.1:5500"},
                            # r"/uploads/*":{"origins":"http://127.0.0.1:5500"}
                            })

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'tune2tab.appspot.com'
})


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
       
@app.route('/submit', methods=['POST'])
def submit():
    data = request.json['data']
    # print(data)
    return jsonify({'message': f'Received:{data}'})

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    bucket = storage.bucket()
    blob = bucket.blob('audio/' + file.filename)
    blob.upload_from_file(file)
    blob.make_public()
    url = blob.public_url
    # download_audio(url, 'downloads')
    data = str(extract_features('D:\Gemini\js\downloads\song.mp3'))
    model = genai.GenerativeModel(model_name=f'tunedModels/{"test75"}')
    result = model.generate_content("Đây là thông số của loại âm thanh nào?" + data)
    print(result.text)
    return jsonify({'message': 'Kết quả là: ' + result.text, 'number': data }), 200

@app.route('/')
def hello_world():
    return 'AAAa'

def download_audio(url, save_folder):

    os.makedirs(save_folder, exist_ok=True)

    response = requests.get(url)

    file_path = os.path.join(save_folder, 'song.mp3')

    with open(file_path, 'wb') as audio_file:
        audio_file.write(response.content)
    


    

def play_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    play(audio)

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

 

    features_dict = {
        'tempo': tempo[0],
        'chroma_stft_mean': np.mean(chroma_stft),
        'rmse_mean': np.mean(rmse),
        'spec_cent_mean': np.mean(spec_cent),
        'spec_bw_mean': np.mean(spec_bw),
        'rolloff_mean': np.mean(rolloff),
        'zcr_mean': np.mean(zcr)
    }

    for i, e in enumerate(mfcc):
        features_dict[f'mfcc_{i+1}_mean'] = np.mean(e)

    return features_dict

if __name__ == '__main__':
    app.run(debug=False)