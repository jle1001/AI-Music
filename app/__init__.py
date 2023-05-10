from flask import Flask, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_security import LoginForm, RegisterForm, Security, SQLAlchemySessionUserDatastore, UserMixin, RoleMixin, roles_required, login_required
from app.src import plot_features, predict_genre
import secrets
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECURITY_PASSWORD_SALT'] = os.environ.get('SECURITY_PASSWORD_SALT')

db = SQLAlchemy(app)
roles_users = db.Table('roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(32), unique=True)
    description = db.Column(db.String(128))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(48), unique=True)
    password = db.Column(db.String(16))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users, backref=db.backref('users', lazy='dynamic'))

user_datastore = SQLAlchemySessionUserDatastore(db, User, Role)
security = Security(app, user_datastore)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegisterForm()

    if form.validate_on_submit():
        user_datastore.create_user(
            email=form.email.data,
            password=form.password.data
        )
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/user_login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()

    if form.validate_on_submit():
        user = user_datastore.get_user(form.email.data)
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))

    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/admin')
@roles_required('admin')
def admin():
    email = current_user.email
    return render_template('admin.html', email=email)

@app.route('/user')
@login_required
def user():
    email = current_user.email
    return render_template('user.html', email=email)

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