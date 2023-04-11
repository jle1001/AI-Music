from flask import Flask, render_template, request
from app.src import process_audio
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
    audio = request.form
    return f'{audio}'
    # process_audio.show_waveform()

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)