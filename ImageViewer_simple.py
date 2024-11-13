import sys
import os
import json
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeView, QFileDialog, QSplitter, QSlider, QLineEdit, QFormLayout
from PyQt6.QtCore import Qt, QModelIndex
from PyQt6.QtGui import QPixmap, QImage, QStandardItemModel, QStandardItem, QKeyEvent
import cv2
import numpy as np
from ultralytics import YOLO  # YOLO 라이브러리 임포트


class ImageViewerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Folder and File Browser")
        self.setGeometry(100, 100, 1200, 800)

        # YOLO 모델 로드
        self.yolo_model = YOLO(r"E:\GitHub\swh_data\test\240730E150.pt")  # YOLO v11 모델 경로 설정

        # 이전 폴더 경로 저장 파일 경로
        self.settings_file = "settings.json"
        self.previous_folder_path = self.load_previous_folder()

        # Threshold 값 및 배율 초기화
        self.threshold_value = 30
        self.scale_factor = 1.0
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        # 레이아웃 설정
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # 스플리터로 좌우 구분
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 이미지 표시
        self.image_label = QLabel("Select an image file to display")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splitter.addWidget(self.image_label)

        # 오른쪽: 폴더 선택, 트리뷰 및 제어 설정
        right_layout = QVBoxLayout()
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        right_layout.addWidget(self.folder_button)

        # 트리뷰 설정
        self.tree_view = QTreeView()
        right_layout.addWidget(self.tree_view)

        # 파일 시스템 모델 대체용 트리 모델 설정
        self.file_system_model = QStandardItemModel()
        self.tree_view.setModel(self.file_system_model)

        # 트리뷰에서 선택 변경 시 이미지 갱신
        self.tree_view.selectionModel().selectionChanged.connect(self.update_image_from_selection)

        # Threshold 슬라이더 및 Confidence, IoU 설정 Edit Box 추가 (오른쪽 레이아웃에 배치)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(self.threshold_value)
        self.slider.valueChanged.connect(self.update_threshold)
        right_layout.addWidget(self.slider)

        form_layout = QFormLayout()
        self.conf_edit = QLineEdit(str(self.conf_threshold))
        self.iou_edit = QLineEdit(str(self.iou_threshold))
        form_layout.addRow("Confidence Threshold:", self.conf_edit)
        form_layout.addRow("IoU Threshold:", self.iou_edit)
        right_layout.addLayout(form_layout)

        # YOLO 라벨링 파일 내용 표시를 위한 QLabel 추가
        self.label_info = QLabel("YOLO Label Info:")
        self.label_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.label_info)

        # 오른쪽 위젯 설정
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        # 스플리터 비율 설정 (이미지 3: 트리뷰와 제어 1 비율)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def load_previous_folder(self):
        # 이전 폴더 경로를 저장한 파일에서 경로 로드
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as file:
                settings = json.load(file)
                return settings.get("previous_folder_path", "")
        return ""

    def save_previous_folder(self, folder_path):
        # 현재 폴더 경로를 파일에 저장
        with open(self.settings_file, 'w') as file:
            json.dump({"previous_folder_path": folder_path}, file)

    def select_folder(self):
        # 폴더 선택 대화상자
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", self.previous_folder_path or "")
        if folder_path:
            self.previous_folder_path = folder_path  # 선택한 폴더를 저장
            self.save_previous_folder(folder_path)  # 폴더 경로 저장
            self.populate_tree(folder_path)

    def populate_tree(self, folder_path):
        # 트리 모델 초기화
        self.file_system_model.clear()
        root_item = QStandardItem(folder_path)
        self.file_system_model.appendRow(root_item)

        # 재귀적으로 폴더와 파일 추가
        self.add_items(root_item, folder_path)

    def add_items(self, parent_item, folder_path):
        # 폴더 내부 파일 및 하위 폴더 추가
        for item_name in sorted(os.listdir(folder_path)):
            item_path = os.path.join(folder_path, item_name)
            item = QStandardItem(item_name)

            if os.path.isdir(item_path):
                parent_item.appendRow(item)
                self.add_items(item, item_path)
            elif item_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                parent_item.appendRow(item)

    def update_threshold(self):
        # 슬라이더로 설정된 Threshold 값을 업데이트
        self.threshold_value = self.slider.value()
        self.update_image_from_selection()

    def update_image_from_selection(self):
        # 선택된 항목을 가져와 이미지 업데이트
        index = self.tree_view.currentIndex()
        if index.isValid():
            self.display_selected_image(index)

    def display_selected_image(self, index: QModelIndex):
        # 선택한 파일 경로 가져오기
        file_path = self.get_full_path(index)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.show_image(file_path)
            self.display_label_info(file_path)

    def get_full_path(self, index: QModelIndex):
        # 선택한 아이템의 전체 경로 반환
        path = []
        while index.isValid():
            path.insert(0, index.data())
            index = index.parent()
        return os.path.join(*path)

    def show_image(self, file_path):
        # OpenCV로 이미지 로드 후 결함 부분만 표시
        image = cv2.imread(file_path)
        if image is not None:
            blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
            _, mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), self.threshold_value, 255, cv2.THRESH_BINARY)
            result_image = np.where(mask[:, :, None] == 255, image, blurred_image)

            # YOLO 모델을 사용한 객체 검출 수행
            self.conf_threshold = float(self.conf_edit.text())
            self.iou_threshold = float(self.iou_edit.text())

            self.yolo_model.conf = self.conf_threshold
            self.yolo_model.iou = self.iou_threshold

            results = self.yolo_model(result_image)

            # 검출된 결과 시각화
            annotated_image = results[0].plot()  # YOLO 결과를 이미지에 시각화

            # YOLO 라벨 파일의 바운딩 박스 추가
            label_file_path = os.path.splitext(file_path)[0] + ".txt"
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as file:
                    labels = file.readlines()
                for label in labels:
                    class_id, x_center, y_center, width, height = map(float, label.strip().split())
                    h, w, _ = annotated_image.shape
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_image, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 이미지 QImage로 변환하여 표시
            height, width, _ = annotated_image.shape
            bytes_per_line = width * 3
            q_image = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def display_label_info(self, file_path):
        # 라벨 파일의 내용을 표시
        label_file_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as file:
                label_data = file.read()
            self.label_info.setText(f"YOLO Label Info:\n{label_data}")
        else:
            self.label_info.setText("YOLO Label Info:\nNo label file found.")

    def keyPressEvent(self, event: QKeyEvent):
        # 방향키 위/아래 입력 감지
        if event.key() == Qt.Key.Key_Up:
            self.select_previous_image()
        elif event.key() == Qt.Key.Key_Down:
            self.select_next_image()

    def select_previous_image(self):
        # 이전 인덱스로 이동
        current_index = self.tree_view.currentIndex()
        previous_index = self.tree_view.indexAbove(current_index)
        if previous_index.isValid():
            self.tree_view.setCurrentIndex(previous_index)
            self.display_selected_image(previous_index)

    def select_next_image(self):
        # 다음 인덱스로 이동
        current_index = self.tree_view.currentIndex()
        next_index = self.tree_view.indexBelow(current_index)
        if next_index.isValid():
            self.tree_view.setCurrentIndex(next_index)
            self.display_selected_image(next_index)

    def wheelEvent(self, event):
        # 마우스 커서 위치 중심으로 확대/축소
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1  # 줌 인
        else:
            self.scale_factor /= 1.1  # 줌 아웃

        self.update_image_from_selection()


# 애플리케이션 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewerApp()
    window.show()
    sys.exit(app.exec())