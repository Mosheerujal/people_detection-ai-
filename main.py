import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import streamlink


# تحميل نموذج YOLO
model = YOLO('yolov8s.pt')

# وظيفة للتعامل مع أحداث الماوس
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# إعداد نافذة لعرض الفيديو
cv2.namedWindow('RGB_vedio_streem')
cv2.setMouseCallback('RGB_vedio_streem', RGB)

# رابط بث يوتيوب مباشر
youtube_url = 'https://www.youtube.com/watch?v=3LXQWU67Ufk'

# استخدام streamlink للحصول على رابط البث المباشر
streams = streamlink.streams(youtube_url)
stream_url = streams['best'].url

# فتح بث الفيديو باستخدام OpenCV
cap = cv2.VideoCapture(stream_url)

# قراءة ملف coco.txt وتحميل أسماء الفئات
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# متغير لتعقب عدد الإطارات
count = 0

# إنشاء كائن Tracker
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1020, 500))

    # إجراء التنبؤات باستخدام نموذج YOLO
    results = model.predict(frame)

    # تحويل النتائج إلى DataFrame
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes.xyxy  # الوصول إلى صناديق الحدود بتنسيق [x1, y1, x2, y2]
        confs = result.boxes.conf  # الوصول إلى درجات الثقة
        classes = result.boxes.cls  # الوصول إلى تسميات الفئات

        px = pd.DataFrame(boxes.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2'])
        px['conf'] = confs.cpu().numpy()
        px['class'] = classes.cpu().numpy().astype(int)

        for index, row in px.iterrows():
            x1 = int(row['x1'])
            y1 = int(row['y1'])
            x2 = int(row['x2'])
            y2 = int(row['y2'])
            d = int(row['class'])
            c = class_list[d]
            if 'person' in c:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # كود المفتاح ESC هو 27
        break

cap.release()
cv2.destroyAllWindows()
