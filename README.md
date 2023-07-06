<h1 align="center">AI-Music</h1>
<p align="center">
  <a href="https://github.com/jle1001/AI-Music/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/jle1001/AI-Music"></a>
  <a href="https://github.com/jle1001/AI-Music/issues?q=is%3Aopen"><img alt="GitHub issues" src="https://img.shields.io/github/issues/jle1001/AI-Music"></a>
  <a href="https://github.com/jle1001/AI-Music/issues?q=is%3Aissue+is%3Aclosed"><img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/jle1001/AI-Music"></a>
</p>

Repositorio para el TFG: **AI-Music**

Alumno: López Estrada, José Ángel

Tutores: César Ignacio García Osorio, Alicia Olivares Gil

---

<img src="https://i.imgur.com/MpiFtzq.png"/>

## Description

AI-Music is an application that predicts music genres from audio files. The application includes a user-friendly interface that enables the users to upload files and visualizate the predicted genres with the main features of the song.

## Usage

AI-Music project can be used <a href="https://jle1001.pythonanywhere.com/">online</a> or locally. For local use, follow the next steps:

### 1. Clone the repository

```bash
git clone https://github.com/jle1001/AI-Music.git
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
flask run
```

## External files

Dataset and metadata used to train the model is availiable in this repository: <a href="https://github.com/mdeff/fma">FMA: A Dataset For Music Analysis</a>

Last version of the application has been trained using: 
1. **`fma_small.zip`**: audio tracks used to train the model.
2. **`fma_metadata.zip`**: audio tracks information to obtain the genres asociated with the audio files.

# License

<img alt="GitHub" src="https://img.shields.io/github/license/jle1001/AI-Music">