from flask import Flask, render_template, request
from app.src import plot_features, predict_genre
import librosa
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        track = request.files['upload']
        track.save(f'./app/static/upload/upload{os.path.splitext(track.filename)[1]}')
    
    y, sr = librosa.load(f'./app/static/upload/upload{os.path.splitext(track.filename)[1]}')
    sample_rate = sr
    filename = os.path.splitext(track.filename)[0]
    format = os.path.splitext(track.filename)[1]
    duration = librosa.get_duration(y=y)
    minutes, seconds = divmod(duration, 60)

    return render_template('upload.html', 
                           filename=filename, 
                           sample_rate=sample_rate, 
                           format=format, 
                           minutes=int(minutes), 
                           seconds=int(seconds))

@app.route('/analysis', methods=['POST', 'GET'])
def analysis():
    if request.method == 'POST':
        n_model = request.form['algorithm-selection']

    print(n_model)
    waveform_img_src = plot_features.show_waveform()
    spectogram_img_src = plot_features.show_spectogram()
    chromagram_img_src = plot_features.show_chromagram()
    MFCC_img_src = plot_features.show_MFCC()

    prediction = predict_genre.predict(n_model=n_model)
    model_name = predict_genre.get_model(n_model=n_model)
    
    return render_template('analysis.html', 
                           waveform_image=waveform_img_src,
                           spectogram_image=spectogram_img_src,
                           chromagram_image=chromagram_img_src,
                           MFCC_image=MFCC_img_src,
                           model_name=model_name,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)