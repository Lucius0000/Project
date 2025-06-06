from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import shutil
import subprocess
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL配置
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql1234'
app.config['MYSQL_DB'] = 'zhibanyun'

mysql = MySQL(app)

UPLOAD_FOLDER = 'uploaded_files'
RESULT_FOLDER = 'result_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 注册
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, password))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))
    return render_template('register.html')

# 登录
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('index'))
        else:
            return '登录失败，用户名或密码错误'
    return render_template('login.html')

# 登出
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# 敬请期待页面
@app.route('/coming-soon.html')
def coming_soon():
    return render_template('coming-soon.html')

# 下载页面
@app.route('/download.html')
def download():
    file = request.args.get('file')
    name = request.args.get('name', '文件')
    return render_template('download.html', file=file, name=name)

# 问卷页面
@app.route('/questionnaire.html')
def questionnaire():
    return render_template('questionnaire.html')

# 用户反馈页面
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        content = request.form['content']
        user_id = session['user_id']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO feedback (user_id, content) VALUES (%s, %s)", (user_id, content))
        mysql.connection.commit()
        cur.close()
        return render_template('feedback_success.html')

    return render_template('feedback.html')

# 定制化报告上传页
@app.route('/analysis')
def custom_report():
    return render_template('analysis.html')

@app.route('/upload_report', methods=['POST'])
def upload_report():
    if 'file' not in request.files:
        return '未找到文件', 400

    file = request.files['file']
    if file.filename == '':
        return '未选择文件', 400

    job_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{job_id}"
    upload_path = os.path.join(UPLOAD_FOLDER, folder_name)
    output_path = os.path.join(RESULT_FOLDER, folder_name)
    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    filename = secure_filename(file.filename)
    saved_path = os.path.join(upload_path, filename)
    file.save(saved_path)

    process = subprocess.Popen(
        ['python', '数据分析_Flask.py', saved_path, output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def generate():
        for line in iter(process.stdout.readline, ''):
            yield f"data:{line.strip()}\n\n"
        yield f"data:分析完成，文件夹名: {folder_name}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download_result/<folder>')
def download_result(folder):
    folder_path = os.path.join(RESULT_FOLDER, folder)
    zip_path = shutil.make_archive(folder_path, 'zip', folder_path)
    return send_file(zip_path, as_attachment=True, download_name=f"{folder}_结果.zip")

if __name__ == '__main__':
    app.run(debug=True)
