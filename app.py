from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 세션 관리를 위한 시크릿 키

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/signup-check')
def signup_check():
    return render_template('signup-check.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/menu-existing')
def menu_existing():
    return render_template('menu-existing.html')

@app.route('/menu-new')
def menu_new():
    return render_template('menu-new.html')

@app.route('/menu-simple')
def menu_simple():
    return render_template('menu-simple.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

if __name__ == '__main__':
    app.run(debug=True)