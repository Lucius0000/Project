# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:12:18 2024

@author: Lucius
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# 配置 MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql1234'
app.config['MYSQL_DB'] = 'yolo_app'
mysql = MySQL(app)

# 安全和用户会话管理
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# 加载 YOLO 模型
model = YOLO("yolov10s.pt")

# 全局变量
detected_objects = []  # 保存检测到的物体编号及其边界框
selected_object_id = None  # 用户选择的目标编号
tracking_enabled = True  # 控制摄像头启停状态

# 初始化OpenCV的目标跟踪器（使用MOSSE跟踪器为例）
tracker = cv2.legacy.TrackerMOSSE_create()

# 用户类
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    if user:
        return User(user[0], user[1])
    return None

# 路由
@app.route('/')
def index():
    return render_template('index01.html', current_user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            mysql.connection.commit()
            flash("注册成功，请登录", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash("用户名已存在", "danger")
        finally:
            cur.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        if user and bcrypt.check_password_hash(user[1], password):
            login_user(User(user[0], username))
            flash("登录成功", "success")
            return redirect(url_for('index'))
        else:
            flash("用户名或密码错误", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("已退出登录", "info")
    return redirect(url_for('login'))

# 全局变量，用于控制视频流状态
tracking_enabled = True

@app.route('/toggle_tracking', methods=['POST'])
@login_required
def toggle_tracking():
    global tracking_enabled
    tracking_enabled = not tracking_enabled  # 切换启停状态
    return {'status': 'on' if tracking_enabled else 'off'}

selected_bbox = None  # 用户选择的目标框 (x1, y1, x2, y2)

@app.route('/select_object', methods=['POST'])
@login_required
def select_object():
    """用户选择目标编号"""
    global selected_object_id
    data = request.json
    selected_object_id = int(data['object_id'])  # 接收用户选择的目标编号
    return {'message': f'Object {selected_object_id} selected successfully'}

@app.route('/get_detected_objects', methods=['GET'])
@login_required
def get_detected_objects():
    """返回检测到的目标列表"""
    global detected_objects
    return {'objects': detected_objects}

def generate_video_stream():
    """生成视频流并支持目标检测与跟踪"""
    global detected_objects, selected_object_id, tracking_enabled
    cap = cv2.VideoCapture(0)
    tracker = None
    tracking = False

    while True:
        if not tracking_enabled:
            # 暂停状态，发送空白帧
            blank_frame = cv2.imencode('.jpg', cv2.imread('static/blank.jpg'))[1].tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 检测目标
        results = model(frame)
        detected_objects = []
        for i, box in enumerate(results[0].boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # 获取类别 ID 和类别名称
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]  # 获取类别名称

            # 保存检测结果
            detected_objects.append({'id': i, 'bbox': (x1, y1, x2, y2), 'name': class_name})

            # 在视频帧上绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {i}, {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 初始化跟踪器
        if selected_object_id is not None and not tracking:
            selected_bbox = detected_objects[selected_object_id]['bbox']
            x1, y1, x2, y2 = selected_bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker = cv2.legacy.TrackerMOSSE_create()
            tracker.init(frame, bbox)
            tracking = True

        # 跟踪目标
        if tracking:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                tracking = False  # 跟踪失败，允许用户重新选择

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()



@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
