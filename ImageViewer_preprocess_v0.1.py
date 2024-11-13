import sys
import os
import json
import logging
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeView, QFileDialog, QSplitter,
    QSlider, QLineEdit, QFormLayout, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QModelIndex, QPoint
from PyQt6.QtGui import QPixmap, QImage, QStandardItemModel, QStandardItem, QPainter, QColor, QPen, QFont
import cv2
import numpy as np
from ultralytics import YOLO

# 클래스 이름 매핑 정의
CLASS_NAMES = {
    0: "Door",
    1: "Fault",
    2: "Dent",
    3: "Uneven",
    4: "Crack",
    5: "Scratch",
    6: "SemiScartch",
    7: "Bent",
    8: "SoftUneven"
}

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageViewerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Detection Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # 이전 폴더 경로 저장 파일 경로
        self.settings_file = "settings.json"
        self.previous_folder_path = self.load_previous_folder()

        # Ground Truth 관련 변수
        self.gt_scale_factor = 1.0
        self.gt_show_labels = True
        self.gt_base_size = 300
        self.current_gt_image = None
        self.current_image_path = None

        # 배율 초기화 및 전처리 변수 초기화
        self.scale_factor = 1.0
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.clahe_clip_limit = 2.0
        self.gaussian_blur_kernel = 15
        self.canny_threshold1 = 100
        self.canny_threshold2 = 200
        self.use_canny = False
        self.bilateral_filter_diameter = 9
        self.bilateral_filter_sigma_color = 75
        self.bilateral_filter_sigma_space = 75
        self.use_bilateral_filter = False
        self.morph_kernel_size = 3
        self.use_morphological = False
        self.sharpening_amount = 1.0
        self.use_hist_equalization = False
        self.median_blur_kernel = 3
        self.use_median_blur = False
        self.adaptive_thresh_block_size = 11
        self.adaptive_thresh_c = 2
        self.use_adaptive_threshold = False

        # Ground Truth 관련 변수
        self.gt_scale_factor = 1.0
        self.gt_show_labels = True
        self.gt_base_size = 300
        self.current_gt_image = None
        self.current_image_path = None
        self.current_bbox_index = -1  # -1은 "All Boxes"를 의미
        self.center_align_enabled = True
        self.fit_to_bbox_enabled = False

        # YOLO 모델 로드
        try:
            self.yolo_model = YOLO(r"E:\GitHub\swh_data\test\240730E150.pt")
            logging.info("YOLO 모델 로드 성공")
        except Exception as e:
            logging.error(f"YOLO 모델 로드 실패: {e}")
            self.yolo_model = None

        self.initUI()

    def initUI(self):
        """UI 초기화"""
        # 메인 레이아웃
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
        self.setLayout(main_layout)

        # 3-way 스플리터 생성
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 이미지 표시 (4)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)  # 약간의 여백 추가
        self.image_label = QLabel("Select an image file to display")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumWidth(800)  # 최소 너비 설정
        left_layout.addWidget(self.image_label)
        splitter.addWidget(left_widget)

        # 중앙: Ground Truth 이미지 표시 (1)
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(5, 5, 5, 5)
        center_widget.setMinimumWidth(200)

        # Ground Truth 컨트롤 패널 - 첫 번째 줄
        gt_control_layout1 = QHBoxLayout()

        # 크기 조절 슬라이더
        self.gt_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.gt_size_slider.setRange(200, 800)
        self.gt_size_slider.setValue(300)
        self.gt_size_slider.valueChanged.connect(self.update_gt_size)

        # Ground Truth 이미지 크기 표시 레이블
        self.gt_size_label = QLabel("Size: 300x300")

        gt_control_layout1.addWidget(QLabel("GT Size:"))
        gt_control_layout1.addWidget(self.gt_size_slider)
        gt_control_layout1.addWidget(self.gt_size_label)

        # Ground Truth 컨트롤 패널 - 두 번째 줄
        gt_control_layout2 = QHBoxLayout()

        # 라벨링 표시 토글
        self.gt_label_toggle = QCheckBox("Labels")
        self.gt_label_toggle.setChecked(True)
        self.gt_label_toggle.stateChanged.connect(self.toggle_gt_labels)

        # 중앙 정렬 토글
        self.center_align_toggle = QCheckBox("Center")
        self.center_align_toggle.setChecked(True)
        self.center_align_toggle.stateChanged.connect(self.toggle_center_align)

        # 바운딩 박스에 맞춤
        self.fit_to_bbox = QCheckBox("Fit Box")
        self.fit_to_bbox.setChecked(False)
        self.fit_to_bbox.stateChanged.connect(self.toggle_fit_to_bbox)

        # 줌 컨트롤
        zoom_layout = QHBoxLayout()
        self.gt_zoom_out = QPushButton("-")
        self.gt_zoom_in = QPushButton("+")
        self.gt_zoom_out.clicked.connect(lambda: self.zoom_gt(0.9))
        self.gt_zoom_in.clicked.connect(lambda: self.zoom_gt(1.1))
        self.gt_zoom_out.setFixedSize(20, 20)
        self.gt_zoom_in.setFixedSize(20, 20)
        zoom_layout.addWidget(self.gt_zoom_out)
        zoom_layout.addWidget(self.gt_zoom_in)
        zoom_layout.setSpacing(2)

        gt_control_layout2.addWidget(self.gt_label_toggle)
        gt_control_layout2.addWidget(self.center_align_toggle)
        gt_control_layout2.addWidget(self.fit_to_bbox)
        gt_control_layout2.addLayout(zoom_layout)

        # Ground Truth 이미지 레이블
        self.gt_image_label = QLabel("Ground Truth Image")
        self.gt_image_label.setFixedSize(300, 300)
        self.gt_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gt_image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #cccccc;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }
            """)

        # 레이아웃 추가
        center_layout.addLayout(gt_control_layout1)
        center_layout.addLayout(gt_control_layout2)
        center_layout.addWidget(self.gt_image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        center_layout.addStretch()
        splitter.addWidget(center_widget)

        # 마우스 이벤트 활성화
        self.gt_image_label.setMouseTracking(True)

        # 오른쪽: 폴더 선택, 트리뷰 및 제어 설정
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 폴더 선택 버튼
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        right_layout.addWidget(self.folder_button)

        # 트리뷰 설정
        self.tree_view = QTreeView()
        self.file_model = QStandardItemModel()
        self.tree_view.setModel(self.file_model)
        self.tree_view.selectionModel().selectionChanged.connect(self.update_image_from_selection)
        right_layout.addWidget(self.tree_view)

        # 제어 패널 추가
        self.add_control_panel(right_layout)

        # YOLO 라벨링 파일 내용 표시를 위한 QLabel 추가
        self.label_info = QLabel("Detection Info:")
        self.label_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.label_info)

        splitter.addWidget(right_widget)

        # 스플리터 비율 설정 (4:1:1)
        splitter.setStretchFactor(0, 4)  # 왼쪽 패널
        splitter.setStretchFactor(1, 1)  # 중앙 패널
        splitter.setStretchFactor(2, 1)  # 오른쪽 패널

        # 스플리터 핸들 스타일 설정
        splitter.setHandleWidth(2)  # 핸들 너비 설정
        splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #cccccc;
                }
                QSplitter::handle:hover {
                    background-color: #999999;
                }
            """)

        # 초기 크기 설정
        total_width = self.width()
        left_width = int(total_width * 0.67)  # 4/6
        center_width = int(total_width * 0.165)  # 1/6
        right_width = int(total_width * 0.165)  # 1/6
        splitter.setSizes([left_width, center_width, right_width])

        main_layout.addWidget(splitter)

    def add_control_panel(self, layout):
        """제어 패널 추가"""
        # CLAHE Clip Limit 슬라이더
        self.clahe_label = QLabel("CLAHE Clip Limit:")
        layout.addWidget(self.clahe_label)
        self.clahe_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_slider.setRange(1, 10)
        self.clahe_slider.setValue(int(self.clahe_clip_limit))
        self.clahe_slider.valueChanged.connect(self.update_clahe_clip_limit)
        layout.addWidget(self.clahe_slider)

        # Gaussian Blur Kernel Size 슬라이더
        self.gaussian_blur_label = QLabel("Gaussian Blur Kernel Size:")
        layout.addWidget(self.gaussian_blur_label)
        self.gaussian_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_blur_slider.setRange(1, 31)
        self.gaussian_blur_slider.setValue(self.gaussian_blur_kernel)
        self.gaussian_blur_slider.setSingleStep(2)
        self.gaussian_blur_slider.valueChanged.connect(self.update_gaussian_blur_kernel)
        layout.addWidget(self.gaussian_blur_slider)

        # Canny Edge Detection 설정
        self.canny_label1 = QLabel("Canny Threshold 1:")
        layout.addWidget(self.canny_label1)
        self.canny_slider1 = QSlider(Qt.Orientation.Horizontal)
        self.canny_slider1.setRange(0, 500)
        self.canny_slider1.setValue(self.canny_threshold1)
        self.canny_slider1.valueChanged.connect(self.update_canny_threshold1)
        layout.addWidget(self.canny_slider1)

        self.canny_label2 = QLabel("Canny Threshold 2:")
        layout.addWidget(self.canny_label2)
        self.canny_slider2 = QSlider(Qt.Orientation.Horizontal)
        self.canny_slider2.setRange(0, 500)
        self.canny_slider2.setValue(self.canny_threshold2)
        self.canny_slider2.valueChanged.connect(self.update_canny_threshold2)
        layout.addWidget(self.canny_slider2)

        self.canny_checkbox = QCheckBox("Use Canny Edge Detection")
        self.canny_checkbox.setChecked(self.use_canny)
        self.canny_checkbox.stateChanged.connect(self.toggle_canny)
        layout.addWidget(self.canny_checkbox)

        # Bilateral Filter 설정
        self.bilateral_checkbox = QCheckBox("Use Bilateral Filter")
        self.bilateral_checkbox.setChecked(self.use_bilateral_filter)
        self.bilateral_checkbox.stateChanged.connect(self.toggle_bilateral_filter)
        layout.addWidget(self.bilateral_checkbox)

        self.bilateral_diameter_label = QLabel("Bilateral Filter Diameter:")
        layout.addWidget(self.bilateral_diameter_label)
        self.bilateral_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_diameter_slider.setRange(1, 15)
        self.bilateral_diameter_slider.setValue(self.bilateral_filter_diameter)
        self.bilateral_diameter_slider.valueChanged.connect(self.update_bilateral_filter_diameter)
        layout.addWidget(self.bilateral_diameter_slider)

        self.bilateral_sigma_color_label = QLabel("Bilateral Sigma Color:")
        layout.addWidget(self.bilateral_sigma_color_label)
        self.bilateral_sigma_color_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_sigma_color_slider.setRange(10, 150)
        self.bilateral_sigma_color_slider.setValue(self.bilateral_filter_sigma_color)
        self.bilateral_sigma_color_slider.valueChanged.connect(self.update_bilateral_sigma_color)
        layout.addWidget(self.bilateral_sigma_color_slider)

        self.bilateral_sigma_space_label = QLabel("Bilateral Sigma Space:")
        layout.addWidget(self.bilateral_sigma_space_label)
        self.bilateral_sigma_space_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_sigma_space_slider.setRange(10, 150)
        self.bilateral_sigma_space_slider.setValue(self.bilateral_filter_sigma_space)
        self.bilateral_sigma_space_slider.valueChanged.connect(self.update_bilateral_sigma_space)
        layout.addWidget(self.bilateral_sigma_space_slider)

        # Morphological Operations 설정
        self.morphological_checkbox = QCheckBox("Use Morphological Operations")
        self.morphological_checkbox.setChecked(self.use_morphological)
        self.morphological_checkbox.stateChanged.connect(self.toggle_morphological)
        layout.addWidget(self.morphological_checkbox)

        self.morph_kernel_label = QLabel("Morphological Kernel Size:")
        layout.addWidget(self.morph_kernel_label)
        self.morph_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_kernel_slider.setRange(1, 15)
        self.morph_kernel_slider.setValue(self.morph_kernel_size)
        self.morph_kernel_slider.valueChanged.connect(self.update_morph_kernel_size)
        layout.addWidget(self.morph_kernel_slider)

        # Histogram Equalization 체크박스
        self.hist_equalization_checkbox = QCheckBox("Use Histogram Equalization")
        self.hist_equalization_checkbox.setChecked(self.use_hist_equalization)
        self.hist_equalization_checkbox.stateChanged.connect(self.toggle_hist_equalization)
        layout.addWidget(self.hist_equalization_checkbox)

        # Median Blur 설정
        self.median_blur_checkbox = QCheckBox("Use Median Blur")
        self.median_blur_checkbox.setChecked(self.use_median_blur)
        self.median_blur_checkbox.stateChanged.connect(self.toggle_median_blur)
        layout.addWidget(self.median_blur_checkbox)

        self.median_blur_label = QLabel("Median Blur Kernel Size:")
        layout.addWidget(self.median_blur_label)
        self.median_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.median_blur_slider.setRange(1, 31)
        self.median_blur_slider.setValue(self.median_blur_kernel)
        self.median_blur_slider.setSingleStep(2)
        self.median_blur_slider.valueChanged.connect(self.update_median_blur_kernel)
        layout.addWidget(self.median_blur_slider)

        # YOLO 설정
        form_layout = QFormLayout()
        self.conf_edit = QLineEdit(str(self.conf_threshold))
        self.iou_edit = QLineEdit(str(self.iou_threshold))
        form_layout.addRow("Confidence Threshold:", self.conf_edit)
        form_layout.addRow("IoU Threshold:", self.iou_edit)
        layout.addLayout(form_layout)

    def add_control_panel(self, layout):
        """제어 패널 추가"""
        # CLAHE Clip Limit 슬라이더
        self.clahe_label = QLabel("CLAHE Clip Limit:")
        layout.addWidget(self.clahe_label)
        self.clahe_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_slider.setRange(1, 10)
        self.clahe_slider.setValue(int(self.clahe_clip_limit))
        self.clahe_slider.valueChanged.connect(self.update_clahe_clip_limit)
        layout.addWidget(self.clahe_slider)

        # Gaussian Blur Kernel Size 슬라이더
        self.gaussian_blur_label = QLabel("Gaussian Blur Kernel Size:")
        layout.addWidget(self.gaussian_blur_label)
        self.gaussian_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_blur_slider.setRange(1, 31)
        self.gaussian_blur_slider.setValue(self.gaussian_blur_kernel)
        self.gaussian_blur_slider.setSingleStep(2)
        self.gaussian_blur_slider.valueChanged.connect(self.update_gaussian_blur_kernel)
        layout.addWidget(self.gaussian_blur_slider)

        # Canny Edge Detection 설정
        self.canny_label1 = QLabel("Canny Threshold 1:")
        layout.addWidget(self.canny_label1)
        self.canny_slider1 = QSlider(Qt.Orientation.Horizontal)
        self.canny_slider1.setRange(0, 500)
        self.canny_slider1.setValue(self.canny_threshold1)
        self.canny_slider1.valueChanged.connect(self.update_canny_threshold1)
        layout.addWidget(self.canny_slider1)

        self.canny_label2 = QLabel("Canny Threshold 2:")
        layout.addWidget(self.canny_label2)
        self.canny_slider2 = QSlider(Qt.Orientation.Horizontal)
        self.canny_slider2.setRange(0, 500)
        self.canny_slider2.setValue(self.canny_threshold2)
        self.canny_slider2.valueChanged.connect(self.update_canny_threshold2)
        layout.addWidget(self.canny_slider2)

        self.canny_checkbox = QCheckBox("Use Canny Edge Detection")
        self.canny_checkbox.setChecked(self.use_canny)
        self.canny_checkbox.stateChanged.connect(self.toggle_canny)
        layout.addWidget(self.canny_checkbox)

        # Bilateral Filter 설정
        self.bilateral_checkbox = QCheckBox("Use Bilateral Filter")
        self.bilateral_checkbox.setChecked(self.use_bilateral_filter)
        self.bilateral_checkbox.stateChanged.connect(self.toggle_bilateral_filter)
        layout.addWidget(self.bilateral_checkbox)

        self.bilateral_diameter_label = QLabel("Bilateral Filter Diameter:")
        layout.addWidget(self.bilateral_diameter_label)
        self.bilateral_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_diameter_slider.setRange(1, 15)
        self.bilateral_diameter_slider.setValue(self.bilateral_filter_diameter)
        self.bilateral_diameter_slider.valueChanged.connect(self.update_bilateral_filter_diameter)
        layout.addWidget(self.bilateral_diameter_slider)

        self.bilateral_sigma_color_label = QLabel("Bilateral Sigma Color:")
        layout.addWidget(self.bilateral_sigma_color_label)
        self.bilateral_sigma_color_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_sigma_color_slider.setRange(10, 150)
        self.bilateral_sigma_color_slider.setValue(self.bilateral_filter_sigma_color)
        self.bilateral_sigma_color_slider.valueChanged.connect(self.update_bilateral_sigma_color)
        layout.addWidget(self.bilateral_sigma_color_slider)

        self.bilateral_sigma_space_label = QLabel("Bilateral Sigma Space:")
        layout.addWidget(self.bilateral_sigma_space_label)
        self.bilateral_sigma_space_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_sigma_space_slider.setRange(10, 150)
        self.bilateral_sigma_space_slider.setValue(self.bilateral_filter_sigma_space)
        self.bilateral_sigma_space_slider.valueChanged.connect(self.update_bilateral_sigma_space)
        layout.addWidget(self.bilateral_sigma_space_slider)

        # Morphological Operations 설정
        self.morphological_checkbox = QCheckBox("Use Morphological Operations")
        self.morphological_checkbox.setChecked(self.use_morphological)
        self.morphological_checkbox.stateChanged.connect(self.toggle_morphological)
        layout.addWidget(self.morphological_checkbox)

        self.morph_kernel_label = QLabel("Morphological Kernel Size:")
        layout.addWidget(self.morph_kernel_label)
        self.morph_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_kernel_slider.setRange(1, 15)
        self.morph_kernel_slider.setValue(self.morph_kernel_size)
        self.morph_kernel_slider.valueChanged.connect(self.update_morph_kernel_size)
        layout.addWidget(self.morph_kernel_slider)

        # Histogram Equalization 체크박스
        self.hist_equalization_checkbox = QCheckBox("Use Histogram Equalization")
        self.hist_equalization_checkbox.setChecked(self.use_hist_equalization)
        self.hist_equalization_checkbox.stateChanged.connect(self.toggle_hist_equalization)
        layout.addWidget(self.hist_equalization_checkbox)

        # Median Blur 설정
        self.median_blur_checkbox = QCheckBox("Use Median Blur")
        self.median_blur_checkbox.setChecked(self.use_median_blur)
        self.median_blur_checkbox.stateChanged.connect(self.toggle_median_blur)
        layout.addWidget(self.median_blur_checkbox)

        self.median_blur_label = QLabel("Median Blur Kernel Size:")
        layout.addWidget(self.median_blur_label)
        self.median_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.median_blur_slider.setRange(1, 31)
        self.median_blur_slider.setValue(self.median_blur_kernel)
        self.median_blur_slider.setSingleStep(2)
        self.median_blur_slider.valueChanged.connect(self.update_median_blur_kernel)
        layout.addWidget(self.median_blur_slider)

        # YOLO 설정
        form_layout = QFormLayout()
        self.conf_edit = QLineEdit(str(self.conf_threshold))
        self.iou_edit = QLineEdit(str(self.iou_threshold))
        form_layout.addRow("Confidence Threshold:", self.conf_edit)
        form_layout.addRow("IoU Threshold:", self.iou_edit)
        layout.addLayout(form_layout)

    # Ground Truth 관련 메서드들
    def update_gt_size(self, value):
        """Ground Truth 이미지 크기 업데이트"""
        try:
            self.gt_base_size = value
            self.gt_image_label.setFixedSize(value, value)
            self.gt_size_label.setText(f"Size: {value}x{value}")
            self.update_gt_display()
        except Exception as e:
            logging.error(f"GT 크기 업데이트 오류: {e}")

    def toggle_gt_labels(self, state):
        """Ground Truth 라벨 표시 토글"""
        try:
            self.gt_show_labels = bool(state)
            self.update_gt_display()
        except Exception as e:
            logging.error(f"GT 라벨 토글 오류: {e}")

    def zoom_gt(self, factor):
        """Ground Truth 이미지 확대/축소"""
        try:
            self.gt_scale_factor *= factor
            # 최소 0.1배, 최대 5배로 제한
            self.gt_scale_factor = max(0.1, min(self.gt_scale_factor, 5.0))

            # 줌 상태 표시를 위한 레이블 업데이트
            zoom_percent = int(self.gt_scale_factor * 100)
            self.gt_size_label.setText(f"{self.gt_base_size}x{self.gt_base_size} ({zoom_percent}%)")

            self.update_gt_display()
        except Exception as e:
            logging.error(f"GT 줌 오류: {e}")

    def update_bbox_selector(self):
        """바운딩 박스 선택기 업데이트"""
        try:
            self.bbox_selector.clear()
            self.bbox_selector.addItem("All Boxes")

            if self.current_image_path:
                label_file = os.path.splitext(self.current_image_path)[0] + '.txt'
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        for i, line in enumerate(f.readlines(), 1):
                            class_id = int(float(line.strip().split()[0]))
                            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                            self.bbox_selector.addItem(f"Box {i}: {class_name}")
        except Exception as e:
            logging.error(f"바운딩 박스 선택기 업데이트 오류: {e}")

    def on_bbox_selection_changed(self, index):
        """바운딩 박스 선택 변경 처리"""
        self.current_bbox_index = index - 1  # -1은 "All Boxes"
        self.update_gt_display()

    def toggle_center_align(self, state):
        """중앙 정렬 토글 처리"""
        self.center_align_enabled = bool(state)
        self.update_gt_display()

    def toggle_fit_to_bbox(self, state):
        """바운딩 박스에 맞추기 토글 처리"""
        self.fit_to_bbox_enabled = bool(state)
        self.update_gt_display()

    def get_bbox_info(self, lines, index=None):
        """바운딩 박스 정보 가져오기"""
        try:
            img_height, img_width = self.current_gt_image.shape[:2]
            boxes = []

            for i, line in enumerate(lines):
                if index is not None and i != index:
                    continue

                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # 상대 좌표를 픽셀 좌표로 변환
                center_x = int(x_center * img_width)
                center_y = int(y_center * img_height)
                box_w = int(width * img_width)
                box_h = int(height * img_height)
                x1 = int(center_x - box_w / 2)
                y1 = int(center_y - box_h / 2)
                x2 = x1 + box_w
                y2 = y1 + box_h

                boxes.append({
                    'class_id': int(class_id),
                    'center': (center_x, center_y),
                    'size': (box_w, box_h),
                    'coords': (x1, y1, x2, y2)
                })

                if index is not None:
                    break

            return boxes
        except Exception as e:
            logging.error(f"바운딩 박스 정보 가져오기 오류: {e}")
            return []

    def update_gt_display(self):
        """Ground Truth 이미지 표시 업데이트"""
        if self.current_gt_image is None:
            return

        try:
            # 원본 이미지 복사
            gt_image = self.current_gt_image.copy()
            img_height, img_width = gt_image.shape[:2]

            # 라벨 파일에서 바운딩 박스 정보 읽기
            label_file = os.path.splitext(self.current_image_path)[0] + '.txt'
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    boxes = self.get_bbox_info(lines, self.current_bbox_index if self.current_bbox_index >= 0 else None)

                    if boxes:
                        # 중앙 정렬이 활성화된 경우
                        if self.center_align_enabled:
                            center_x, center_y = boxes[0]['center']

                            # 이동량 계산
                            offset_x = (self.gt_base_size // 2) - center_x
                            offset_y = (self.gt_base_size // 2) - center_y

                            # 이미지 이동
                            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                            gt_image = cv2.warpAffine(gt_image, M, (self.gt_base_size, self.gt_base_size))

                            # 박스 좌표 조정
                            for box in boxes:
                                x1, y1, x2, y2 = box['coords']
                                x1 += offset_x
                                y1 += offset_y
                                x2 += offset_x
                                y2 += offset_y
                                box['coords'] = (x1, y1, x2, y2)

                        # 바운딩 박스에 맞추기가 활성화된 경우
                        if self.fit_to_bbox_enabled and boxes:
                            box_w, box_h = boxes[0]['size']

                            # 패딩 추가 (박스 크기의 20%)
                            padding = 0.2
                            crop_w = int(box_w * (1 + padding))
                            crop_h = int(box_h * (1 + padding))

                            # 크롭 영역 계산
                            center_x = self.gt_base_size // 2
                            center_y = self.gt_base_size // 2
                            x1 = max(0, center_x - crop_w // 2)
                            y1 = max(0, center_y - crop_h // 2)
                            x2 = min(self.gt_base_size, x1 + crop_w)
                            y2 = min(self.gt_base_size, y1 + crop_h)

                            # 이미지 크롭
                            gt_image = gt_image[y1:y2, x1:x2]

                            # 박스 좌표 조정
                            for box in boxes:
                                bx1, by1, bx2, by2 = box['coords']
                                bx1 -= x1
                                by1 -= y1
                                bx2 -= x1
                                by2 -= y1
                                box['coords'] = (bx1, by1, bx2, by2)

                        # 라벨 표시가 활성화된 경우 박스 그리기
                        if self.gt_show_labels:
                            for box in boxes:
                                x1, y1, x2, y2 = box['coords']
                                class_id = box['class_id']

                                # 박스 그리기
                                cv2.rectangle(gt_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # 클래스 이름 표시
                                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                                label = f"{class_name}"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                cv2.rectangle(gt_image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
                                cv2.putText(gt_image, label, (x1, y1 - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 이미지 크기 조정을 위한 크기 계산
            scaled_size = int(self.gt_base_size * self.gt_scale_factor)

            # 이미지 크기 조정 및 표시
            gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            gt_qimage = QImage(gt_rgb.data, gt_image.shape[1], gt_image.shape[0],
                               gt_image.shape[1] * 3, QImage.Format.Format_RGB888)

            gt_pixmap = QPixmap.fromImage(gt_qimage)
            scaled_gt_pixmap = gt_pixmap.scaled(
                scaled_size, scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Ground Truth 이미지 레이블 크기 업데이트
            self.gt_image_label.setFixedSize(self.gt_base_size, self.gt_base_size)
            self.gt_image_label.setPixmap(scaled_gt_pixmap)

        except Exception as e:
            logging.error(f"GT 디스플레이 업데이트 오류: {e}")
    def draw_gt_labels(self, image):
        """Ground Truth 라벨 그리기"""
        try:
            # 라벨 파일에서 정보 읽기
            label_file = os.path.splitext(self.current_image_path)[0] + '.txt'
            if not os.path.exists(label_file):
                return

            img_height, img_width = image.shape[:2]

            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())

                        # 상대 좌표를 픽셀 좌표로 변환
                        x1 = int((x_center - width / 2) * img_width)
                        y1 = int((y_center - height / 2) * img_height)
                        x2 = int((x_center + width / 2) * img_width)
                        y2 = int((y_center + height / 2) * img_height)

                        # 박스 그리기
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 클래스 이름 표시
                        class_name = CLASS_NAMES.get(int(class_id), f"Class {class_id}")
                        label = f"{class_name}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(image, label, (x1, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    except Exception as e:
                        logging.error(f"GT 라벨 그리기 오류: {e}")
                        continue

        except Exception as e:
            logging.error(f"GT 라벨 처리 오류: {e}")

    # 컨트롤 패널 업데이트 메서드들
    def update_clahe_clip_limit(self, value):
        """CLAHE Clip Limit 업데이트"""
        self.clahe_clip_limit = float(value)
        self.update_image_from_selection()

    def update_gaussian_blur_kernel(self, value):
        """Gaussian Blur Kernel Size 업데이트"""
        self.gaussian_blur_kernel = int(value)
        if self.gaussian_blur_kernel % 2 == 0:
            self.gaussian_blur_kernel += 1
        self.update_image_from_selection()

    def update_canny_threshold1(self, value):
        """Canny Threshold 1 업데이트"""
        self.canny_threshold1 = int(value)
        self.update_image_from_selection()

    def update_canny_threshold2(self, value):
        """Canny Threshold 2 업데이트"""
        self.canny_threshold2 = int(value)
        self.update_image_from_selection()

    def toggle_canny(self, state):
        """Canny Edge Detection 토글"""
        self.use_canny = bool(state)
        self.update_image_from_selection()

    def update_bilateral_filter_diameter(self, value):
        """Bilateral Filter Diameter 업데이트"""
        self.bilateral_filter_diameter = int(value)
        self.update_image_from_selection()

    def update_bilateral_sigma_color(self, value):
        """Bilateral Sigma Color 업데이트"""
        self.bilateral_filter_sigma_color = int(value)
        self.update_image_from_selection()

    def update_bilateral_sigma_space(self, value):
        """Bilateral Sigma Space 업데이트"""
        self.bilateral_filter_sigma_space = int(value)
        self.update_image_from_selection()

    def toggle_bilateral_filter(self, state):
        """Bilateral Filter 토글"""
        self.use_bilateral_filter = bool(state)
        self.update_image_from_selection()

    def toggle_morphological(self, state):
        """Morphological Operations 토글"""
        self.use_morphological = bool(state)
        self.update_image_from_selection()

    def update_morph_kernel_size(self, value):
        """Morphological Kernel Size 업데이트"""
        self.morph_kernel_size = int(value)
        self.update_image_from_selection()

    def toggle_hist_equalization(self, state):
        """Histogram Equalization 토글"""
        self.use_hist_equalization = bool(state)
        self.update_image_from_selection()

    def toggle_median_blur(self, state):
        """Median Blur 토글"""
        self.use_median_blur = bool(state)
        self.update_image_from_selection()

    def update_median_blur_kernel(self, value):
        """Median Blur Kernel Size 업데이트"""
        self.median_blur_kernel = int(value)
        if self.median_blur_kernel % 2 == 0:
            self.median_blur_kernel += 1
        self.update_image_from_selection()

    def process_and_display_image(self, image_path):
        """이미지 처리 및 표시"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"이미지 로드 실패: {image_path}")
                return

            # Ground Truth 이미지 저장
            self.current_gt_image = image.copy()
            self.current_image_path = image_path

            # 전처리 적용
            processed = self.apply_preprocessing(image.copy())

            # YOLO 검출
            if self.yolo_model:
                try:
                    # YOLO 라벨 파일 읽기
                    label_file = os.path.splitext(image_path)[0] + '.txt'
                    boxes = []
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            img_height, img_width = processed.shape[:2]
                            for line in lines:
                                try:
                                    class_id, x_center, y_center, width, height = map(float, line.strip().split())

                                    # 상대 좌표를 픽셀 좌표로 변환
                                    x1 = int((x_center - width / 2) * img_width)
                                    y1 = int((y_center - height / 2) * img_height)
                                    x2 = int((x_center + width / 2) * img_width)
                                    y2 = int((y_center + height / 2) * img_height)

                                    boxes.append({
                                        'class_id': int(class_id),
                                        'coords': (x1, y1, x2, y2)
                                    })
                                except Exception as e:
                                    logging.error(f"라벨 라인 파싱 오류: {e}")
                                    continue

                    # YOLO 검출 수행
                    results = self.yolo_model(processed)

                    # 검출 결과 그리기
                    self.draw_detections(processed, results[0], boxes)
                    # 검출 정보 업데이트
                    self.update_detection_info(results[0], boxes, image_path)

                except Exception as e:
                    logging.error(f"YOLO 처리 오류: {e}")

            # 이미지 표시
            self.display_image(processed)

            # Ground Truth 디스플레이 업데이트
            self.update_gt_display()

        except Exception as e:
            logging.error(f"이미지 처리 중 오류: {e}")

    def update_detection_info(self, results, gt_boxes, image_path):
        """검출 정보 및 평가 메트릭 업데이트"""
        try:
            # 예측 박스 정보 구성
            pred_boxes = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pred_boxes.append({
                    'class_id': cls,
                    'confidence': conf,
                    'coords': (x1, y1, x2, y2)
                })

            # 정보 텍스트 구성
            info_text = f"File: {os.path.basename(image_path)}\n\n"

            # 예측 결과 정보
            info_text += "Predictions:\n"
            for box in pred_boxes:
                cls = box['class_id']
                x1, y1, x2, y2 = box['coords']
                conf = box['confidence']
                class_name = CLASS_NAMES.get(cls, f"Class {cls}")
                info_text += f"- {class_name} (conf: {conf:.3f}): ({x1}, {y1}) to ({x2}, {y2})\n"

            # Ground Truth 정보
            info_text += "\nGround Truth:\n"
            for box in gt_boxes:
                cls = box['class_id']
                x1, y1, x2, y2 = box['coords']
                class_name = CLASS_NAMES.get(cls, f"Class {cls}")
                info_text += f"- {class_name}: ({x1}, {y1}) to ({x2}, {y2})\n"

            # 매칭 분석
            info_text += "\nMatching Analysis:\n"
            for pred_box in pred_boxes:
                pred_cls = pred_box['class_id']
                pred_coords = pred_box['coords']

                # 가장 높은 IoU를 가진 GT 박스 찾기
                best_iou = 0
                best_gt = None
                for gt_box in gt_boxes:
                    if gt_box['class_id'] == pred_cls:
                        iou = self.calculate_iou(pred_coords, gt_box['coords'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt_box

                pred_class_name = CLASS_NAMES.get(pred_cls, f"Class {pred_cls}")
                if best_gt is not None:
                    info_text += f"Prediction {pred_class_name} matches GT with IoU: {best_iou:.3f}\n"
                else:
                    info_text += f"Prediction {pred_class_name} has no matching GT\n"

            self.label_info.setText(info_text)

        except Exception as e:
            logging.error(f"검출 정보 업데이트 오류: {e}")

    def calculate_iou(self, box1, box2):
        """두 박스 간의 IoU(Intersection over Union) 계산"""
        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            # 교집합 영역 계산
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            if x2_i < x1_i or y2_i < y1_i:
                return 0.0

            intersection = (x2_i - x1_i) * (y2_i - y1_i)

            # 각 박스의 면적 계산
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            # IoU 계산
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0

            return iou

        except Exception as e:
            logging.error(f"IoU 계산 중 오류: {e}")
            return 0.0

    def apply_preprocessing(self, image):
        """이미지 전처리 적용"""
        try:
            # 그레이스케일 변환
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 1. 노이즈 제거 단계
            if self.use_median_blur:
                kernel_size = self.median_blur_kernel
                if kernel_size % 2 == 0:
                    kernel_size += 1
                processed = cv2.medianBlur(processed, kernel_size)

            if self.use_bilateral_filter:
                processed = cv2.bilateralFilter(
                    processed,
                    self.bilateral_filter_diameter,
                    self.bilateral_filter_sigma_color,
                    self.bilateral_filter_sigma_space
                )

            kernel_size = self.gaussian_blur_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

            # 2. 대비 향상 단계
            if self.use_hist_equalization:
                processed = cv2.equalizeHist(processed)

            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
            processed = clahe.apply(processed)

            # 3. 형태학적 처리 단계
            if self.use_morphological:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (self.morph_kernel_size, self.morph_kernel_size)
                )
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

            # 4. 엣지 검출 및 이진화 단계
            if self.use_adaptive_threshold:
                processed = cv2.adaptiveThreshold(
                    processed,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    self.adaptive_thresh_block_size,
                    self.adaptive_thresh_c
                )

            if self.use_canny:
                processed = cv2.Canny(processed, self.canny_threshold1, self.canny_threshold2)

            # 컬러로 변환
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            return processed

        except Exception as e:
            logging.error(f"전처리 중 오류: {e}")
            return image

    def draw_detections(self, image, results, label_boxes):
        """검출 결과와 라벨 정보 시각화"""
        try:
            # Ground Truth 박스 그리기 (녹색)
            for box in label_boxes:
                x1, y1, x2, y2 = box['coords']
                cls = box['class_id']
                class_name = CLASS_NAMES.get(cls, f"Class {cls}")

                # Ground Truth 박스 (녹색)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Ground Truth 라벨 텍스트
                gt_label = f"GT: {class_name}"
                (tw, th), _ = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(image, gt_label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # YOLO 검출 결과 박스 그리기 (빨간색)
            for box in results.boxes:
                try:
                    # 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 클래스 및 신뢰도
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 클래스 이름
                    class_name = CLASS_NAMES.get(cls, f"Class {cls}")

                    # 검출 박스 (빨간색)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # 라벨 텍스트
                    pred_label = f"Pred: {class_name} {conf:.2f}"

                    # y1-th-4 대신 y1-th-24를 사용하여 GT 라벨과 겹치지 않도록 함
                    (tw, th), _ = cv2.getTextSize(pred_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(image, (x1, y1 - th - 24), (x1 + tw, y1 - 20), (0, 0, 255), -1)
                    cv2.putText(image, pred_label, (x1, y1 - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                except Exception as e:
                    logging.error(f"박스 그리기 오류: {e}")
                    continue

        except Exception as e:
            logging.error(f"검출 결과 시각화 오류: {e}")

    def display_image(self, image):
        """이미지를 Qt 라벨에 표시"""
        try:
            height, width = image.shape[:2]

            # 스케일링된 크기 계산
            scaled_width = int(width * self.scale_factor)
            scaled_height = int(height * self.scale_factor)

            # 이미지 리사이즈
            if self.scale_factor != 1.0:
                scaled_image = cv2.resize(image, (scaled_width, scaled_height))
            else:
                scaled_image = image

            # OpenCV BGR -> RGB 변환
            image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

            # QImage 생성
            bytes_per_line = 3 * scaled_width
            q_image = QImage(image_rgb.data, scaled_width, scaled_height,
                             bytes_per_line, QImage.Format.Format_RGB888)

            # QLabel에 표시 (라벨 크기에 맞게 조정)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            logging.error(f"이미지 표시 오류: {e}")

    def wheelEvent(self, event):
        """마우스 휠 이벤트 처리 (줌 인/아웃)"""
        try:
            # 휠 델타값 얻기
            delta = event.angleDelta().y()

            # 줌 스케일 조정
            if delta > 0:  # 휠 위로 (줌 인)
                self.scale_factor *= 1.1
            else:  # 휠 아래로 (줌 아웃)
                self.scale_factor /= 1.1

            # 최소/최대 줌 레벨 제한
            self.scale_factor = max(0.1, min(self.scale_factor, 5.0))

            # 현재 이미지 업데이트
            self.update_image_from_selection()

        except Exception as e:
            logging.error(f"줌 처리 중 오류: {e}")

    def load_previous_folder(self):
        """이전 폴더 경로 로드"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as file:
                    settings = json.load(file)
                    return settings.get("previous_folder_path", "")
            except Exception as e:
                logging.error(f"설정 파일 로드 실패: {e}")
        return ""

    def save_previous_folder(self, folder_path):
        """현재 폴더 경로 저장"""
        try:
            with open(self.settings_file, 'w') as file:
                json.dump({"previous_folder_path": folder_path}, file)
        except Exception as e:
            logging.error(f"설정 파일 저장 실패: {e}")

    def select_folder(self):
        """폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.previous_folder_path or "")
        if folder:
            try:
                self.previous_folder_path = folder
                self.save_previous_folder(folder)
                self.file_model.clear()
                root_item = QStandardItem(os.path.basename(folder))
                root_item.setData(folder)
                self.file_model.appendRow(root_item)
                self.add_items(root_item, folder)
            except Exception as e:
                logging.error(f"폴더 선택 중 오류: {e}")

    def add_items(self, parent_item, path):
        """트리에 아이템 추가"""
        try:
            items = []
            for name in os.listdir(path):
                item_path = os.path.join(path, name)
                item = QStandardItem(name)
                item.setData(item_path)

                if os.path.isdir(item_path):
                    self.add_items(item, item_path)
                    items.append(item)
                elif name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    items.append(item)

            items.sort(key=lambda x: x.text().lower())
            for item in items:
                parent_item.appendRow(item)
        except Exception as e:
            logging.error(f"아이템 추가 중 오류: {e}")

    def update_image_from_selection(self):
        """선택된 이미지 업데이트"""
        try:
            index = self.tree_view.currentIndex()
            if index.isValid():
                path = self.get_full_path(index)
                if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.process_and_display_image(path)
        except Exception as e:
            logging.error(f"이미지 업데이트 중 오류: {e}")

    def get_full_path(self, index):
        """트리 아이템의 전체 경로 반환"""
        try:
            if not index.isValid():
                return ""

            item = self.file_model.itemFromIndex(index)
            if item is None:
                return ""

            return item.data() or ""
        except Exception as e:
            logging.error(f"경로 가져오기 오류: {e}")
            return ""


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewerApp()
    viewer.show()
    sys.exit(app.exec())