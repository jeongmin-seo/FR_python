import sys
import cv2
import insightface
import face_align

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from pymilvus import MilvusClient



class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        # --- DB ---
        self.client = MilvusClient("./test.db")

        # --- 얼굴인식 모델 ---
        self.det = insightface.model_zoo.retinaface.RetinaFace("./buffalo_l/det_10g.onnx")
        self.rec = insightface.model_zoo.arcface_onnx.ArcFaceONNX("./buffalo_l/w600k_r50.onnx")

        self.kps = None
        self.box = None
        self.align_img = None
        self.is_webcam = True

        self.upload_queue = []
        self.current_index = 0
        
        self.setWindowTitle("PyQt5 Webcam Viewer + Number Input")

        # --- UI 구성 ---
        self.image_label = QLabel()
        self.input_id_field = QLineEdit()
        self.input_id_field.setPlaceholderText("ID를 입력하세요")
        self.input_name_field = QLineEdit()
        self.input_name_field.setPlaceholderText("이름을 입력하세요")
        self.submit_button = QPushButton("저장")
        self.submit_button.clicked.connect(self.on_submit)
        self.upload_button = QPushButton("이미지 업로드")
        self.upload_button.clicked.connect(self.upload_image)
        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.on_cancel)

        # 레이아웃 구성
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_id_field)
        input_layout.addWidget(self.input_name_field)
        input_layout.addWidget(self.submit_button)
        input_layout.addWidget(self.cancel_button)
        input_layout.addWidget(self.upload_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

        # --- 비디오 스트림 ---
        self.cap = cv2.VideoCapture(1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):

        if not self.is_webcam:
            return

        ret, frame = self.cap.read()
        if ret:
            self.box = None
            self.kps = None
            self.align_img = None

            self.face_detect(frame)


    def upload_image(self):
        self.is_webcam = False

        file_paths, _  = QFileDialog.getOpenFileNames(self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg)")

        if not file_paths:  # upload 에서 취소 누른 경우
            print("파일 선택 취소")
            self.is_webcam = True
            return

        self.upload_queue = file_paths
        self.current_index = 0
        self.show_next_image()


    # 추후 list pop(0)로 변경하기
    def show_next_image(self):
        if self.current_index >= len(self.upload_queue):
            print("모든 이미지 처리가 끝났습니다.")
            self.upload_queue = []
            self.is_webcam = True
            return

        # 입력 필드 초기화
        self.input_id_field.clear()
        self.input_name_field.clear()

        file_path = self.upload_queue[self.current_index]
        image = cv2.imread(file_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {file_path}")
            self.current_index += 1
            self.show_next_image()
            return

        self.face_detect(image)


    def face_detect(self, image):

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (640, 640))

        boxes, kpss = self.det.detect(rgb_image, input_size=(640, 640))

        if boxes.size:
            r = boxes[0]
            self.box = boxes[0]
            self.kps = kpss[0]
            cv2.rectangle(rgb_image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)
            self.align_img = face_align.norm_crop(rgb_image, landmark=kpss[0], image_size=112)

        else:
            print("FAILED DETECT FACE")

        h, w, ch = rgb_image.shape
        img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    # def on_submit(self):
    #     face_id = self.input_id_field.text()
    #     user_name = self.input_name_field.text()
    #     try:
    #         face_id = int(face_id)
        
    #     except ValueError:
    #         print("유효한 숫자를 입력하세요.")

    #     if self.box is None:
    #             print("BOX 검출이 안됐습니다.")
            
    #     else:
    #         print("BOX 검출")
    #         face_id = int(face_id)
    #         print(f"입력된 숫자: {face_id}, 입력된 이름:{user_name}")

    #         vector = self.rec.get_feat(self.align_img).flatten()
            
    #         data = [
    #             {"id": face_id, "vector": vector, "name": user_name}
    #         ]

    #         print(data)

    #         res = self.client.insert(collection_name="demo_collection", data=data)

    def on_submit(self):
        if self.box is None or self.align_img is None:
            print("얼굴이 검출되지 않았습니다.")
            return

        face_id = self.input_id_field.text()
        user_name = self.input_name_field.text()
        try:
            face_id = int(face_id)
        except ValueError:
            print("유효한 숫자를 입력하세요.")
            return

        vector = self.rec.get_feat(self.align_img).flatten()
        data = [{"id": face_id, "vector": vector, "name": user_name}]
        print("Insert:", data)
        self.client.insert(collection_name="demo_collection", data=data)

        # 다음 이미지로
        self.current_index += 1
        self.show_next_image()


    def on_cancel(self):
        if not self.is_webcam:
            print("현재 이미지 건너뛰기")
            self.current_index += 1
            self.show_next_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())