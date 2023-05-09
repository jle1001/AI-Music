from flask import Flask, render_template, request
from app.src import plot_features, predict_genre
import os
import librosa

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        track = request.files['upload']
        track.save(f'./app/upload/upload{os.path.splitext(track.filename)[1]}')
    
    return render_template('upload.html')

@app.route('/analysis', methods=['POST', 'GET'])
def analysis():
    waveform_img_src = plot_features.show_waveform()
    spectogram_img_src = plot_features.show_spectogram()
    chromagram_img_src = plot_features.show_chromagram()
    MFCC_img_src = plot_features.show_MFCC()

    prediction = predict_genre.predict()
    
    return render_template('analysis.html', 
                           waveform_image=waveform_img_src,
                           spectogram_image=spectogram_img_src,
                           chromagram_image=chromagram_img_src,
                           MFCC_image=MFCC_img_src,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)