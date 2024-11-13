import sys
import cv2
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

class FilterApplicationTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Filter Application Tool")
        self.setGeometry(100, 100, 1200, 1000)

        # 필터 설정 변수
        self.original_image = None
        self.filters = {
            "Original": self.show_original,
            "Gaussian Blur": self.apply_gaussian_blur,
            "Median Blur": self.apply_median_blur,
            "Laplacian": self.apply_laplacian,
            "Sobel X": self.apply_sobel_x,
            "Sobel Y": self.apply_sobel_y,
            "Scharr X": self.apply_scharr_x,
            "Scharr Y": self.apply_scharr_y,
            "Prewitt": self.apply_prewitt,
            "Canny Edge": self.apply_canny,
        }

        self.initUI()

    def initUI(self):
        """UI 초기화 및 레이아웃 설정"""
        main_layout = QVBoxLayout(self)

        # 원본 이미지 표시 라벨
        self.original_image_label = QLabel("Load an image to start.")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.original_image_label)

        # 필터 미리보기 레이아웃 (5x2 그리드)
        self.preview_layout = QGridLayout()
        main_layout.addLayout(self.preview_layout)

        # Load Image 버튼을 미리보기 레이아웃 아래에 추가
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_image(self):
        """이미지 로드 기능"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            # OpenCV로 이미지 로드 시도
            self.original_image = self.load_image_with_pil(file_path)
            if self.original_image is None:
                print("Error: Could not load the image. Please check the file path or file format.")
                return
            self.display_original_image(self.original_image)
            self.apply_all_filters()

    def load_image_with_pil(self, file_path):
        """PIL을 사용하여 이미지를 로드한 후 OpenCV 형식으로 변환"""
        try:
            pil_image = Image.open(file_path).convert("RGB")  # RGB로 변환
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # OpenCV 형식으로 변환
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def display_original_image(self, image):
        """원본 이미지를 QLabel에 표시"""
        pixmap = self.convert_cv_qt(image, width=600, height=400)
        self.original_image_label.setPixmap(pixmap)

    def apply_all_filters(self):
        """모든 필터를 적용하여 미리보기 이미지 생성"""
        if self.original_image is None:
            return

        # 기존 미리보기 이미지 및 라벨 삭제
        for i in reversed(range(self.preview_layout.count())):
            self.preview_layout.itemAt(i).widget().setParent(None)

        # 각 필터를 적용하여 미리보기 이미지 및 필터명 표시
        for idx, (name, func) in enumerate(self.filters.items()):
            row, col = divmod(idx, 5)  # 5개씩 한 행에 배치
            filtered_image = func(self.original_image)
            image_label = QLabel()
            image_label.setPixmap(self.convert_cv_qt(filtered_image, width=200, height=200))  # 이미지 크기 조정
            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # 그리드에 이미지와 필터 이름 배치
            self.preview_layout.addWidget(image_label, row * 2, col)
            self.preview_layout.addWidget(name_label, row * 2 + 1, col)

    def convert_cv_qt(self, cv_img, width, height):
        """OpenCV 이미지를 QPixmap 형식으로 변환하여 QLabel에 표시"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
        h, w, ch = resized_image.shape
        bytes_per_line = ch * w
        q_image = QImage(resized_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

    # 필터 함수 정의
    def show_original(self, image):
        return image

    def apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def apply_median_blur(self, image):
        return cv2.medianBlur(image, 5)

    def apply_laplacian(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    def apply_sobel_x(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        return cv2.convertScaleAbs(sobelx)

    def apply_sobel_y(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(sobely)

    def apply_scharr_x(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        return cv2.convertScaleAbs(scharrx)

    def apply_scharr_y(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(scharry)

    def apply_prewitt(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        processed_x = cv2.filter2D(gray, cv2.CV_32F, kernelx)
        processed_y = cv2.filter2D(gray, cv2.CV_32F, kernely)
        processed = cv2.magnitude(processed_x, processed_y)
        return cv2.convertScaleAbs(processed)

    def apply_canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tool = FilterApplicationTool()
    tool.show()
    sys.exit(app.exec())
