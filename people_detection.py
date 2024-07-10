import sys
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import streamlink
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.stream_url = None
        self.model = YOLO('yolov8s.pt')

        # قراءة ملف coco.txt وتحميل أسماء الفئات
        with open("coco.txt", "r") as my_file:
            data = my_file.read()
        self.class_list = data.split("\n")

    def set_stream_url(self, url):
        self.stream_url = url

    def run(self):
        if self.stream_url:
            streams = streamlink.streams(self.stream_url)
            if 'best' in streams:
                stream_url = streams['best'].url
                cap = cv2.VideoCapture(stream_url)

                while self._run_flag:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (1020, 500))
                        results = self.model.predict(frame)

                        if len(results) > 0:
                            result = results[0]
                            boxes = result.boxes.xyxy
                            confs = result.boxes.conf
                            classes = result.boxes.cls

                            px = pd.DataFrame(boxes.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2'])
                            px['conf'] = confs.cpu().numpy()
                            px['class'] = classes.cpu().numpy().astype(int)

                            for index, row in px.iterrows():
                                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                                d = int(row['class'])
                                c = self.class_list[d]
                                if 'person' in c:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                    cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                        self.change_pixmap_signal.emit(frame)

                cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Live Stream Detector")
        self.disply_width = 1020
        self.display_height = 500
        
        # إنشاء العناصر
        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText("أدخل رابط البث المباشر هنا")
        self.start_button = QPushButton("تشغيل", self)
        self.stop_button = QPushButton("إيقاف", self)
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        
        # إعداد التخطيط
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.url_input)
        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.stop_button)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)
        
        # إنشاء ويدجت مركزي وتعيين التخطيط
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # ربط الأزرار بالوظائف
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        
        # إنشاء خيط الفيديو
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

    def start_video(self):
        url = self.url_input.text()
        if url:
            self.thread.set_stream_url(url)
            self.thread.start()

    def stop_video(self):
        self.thread.stop()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())