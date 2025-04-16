import cv2

# 创建视频捕获对象
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 使用 legacy 模块来创建 MOSSE 跟踪器
tracker = cv2.legacy.TrackerMOSSE_create()

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    cap.release()
    exit()

# 手动选择要跟踪的目标
bbox = cv2.selectROI("选择跟踪目标", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("选择跟踪目标")

# 初始化跟踪器
tracker.init(frame, bbox)

# 开始视频捕获和跟踪
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 更新跟踪器
    success, bbox = tracker.update(frame)

    # 绘制跟踪框
    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示当前帧
    cv2.imshow("Tracking", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
