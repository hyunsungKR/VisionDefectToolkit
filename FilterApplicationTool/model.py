# model.py

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
import os

class FilterApplicationModel:
    def __init__(self):
        self.original_images = []  # List of original OpenCV images
        self.filtered_images = []  # List of tuples: (original_index, filter_name, filtered_image)
        self.main_image = None     # Currently selected OpenCV image
        self.main_filter = "Original"
        self.filter_intensity = 1.0
        self.filters = {
            "Original": self.show_original,
            "Bandpass Filter": self.apply_bandpass_filter,
            "Gabor Filter": self.apply_gabor_filter,
            "Laplacian": self.apply_laplacian,
            "Sobel X": self.apply_sobel_x,
            "Sobel Y": self.apply_sobel_y,
            "Scharr X": self.apply_scharr_x,
            "Scharr Y": self.apply_scharr_y,
            "Prewitt": self.apply_prewitt,
            "Canny Edge": self.apply_canny,
        }
        self.filter_intensities = {name: 1.0 for name in self.filters}

    def load_images(self, from_directory=False):
        """Load images either individually or from a directory."""
        if from_directory:
            directory = QFileDialog.getExistingDirectory(None, "Select Directory")
            if not directory:
                return []

            file_paths = []
            for root, _, files in os.walk(directory):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_paths.append(os.path.join(root, filename))
        else:
            file_paths, _ = QFileDialog.getOpenFileNames(None, "Load Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
            if not file_paths:
                return []

        loaded_images = []
        for path in file_paths:
            img = self.load_image_with_pil(path)
            if img is not None:
                loaded_images.append(img)
            if len(loaded_images) >= 1:  # Limit to 1 original image for 10 filters
                break

        if not loaded_images:
            return []

        self.original_images = loaded_images
        self.filtered_images = []
        for idx, img in enumerate(self.original_images):
            for filter_name in self.filters.keys():
                filtered = self.apply_filter(filter_name, img, self.filter_intensity)
                self.filtered_images.append((idx, filter_name, filtered))
                if len(self.filtered_images) >= 10:
                    break
            if len(self.filtered_images) >= 10:
                break

        return [(self.convert_cv_qt(img), filter_name) for (_, filter_name, img) in self.filtered_images]

    def load_image_with_pil(self, file_path):
        """Load image using PIL and convert to OpenCV format."""
        try:
            pil_image = Image.open(file_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def set_main_image_by_filter(self, filter_name):
        """Set the main image based on the filter name."""
        for orig_idx, fname, img in self.filtered_images:
            if fname == filter_name:
                self.main_image = self.original_images[orig_idx]
                self.main_filter = fname
                self.filter_intensity = self.filter_intensities[fname]
                break

    def update_filter_intensity(self, intensity):
        """Update the intensity for the current main filter and reapply it."""
        self.filter_intensity = intensity
        self.filter_intensities[self.main_filter] = intensity

        # Apply current filter with new intensity
        filtered = self.apply_filter(self.main_filter, self.main_image, self.filter_intensity)
        if filtered is not None:
            # Update the filtered_images list
            for idx, (orig_idx, fname, _) in enumerate(self.filtered_images):
                if orig_idx == self.original_images.index(self.main_image) and fname == self.main_filter:
                    self.filtered_images[idx] = (orig_idx, fname, filtered)
                    break
            return self.convert_cv_qt(filtered)
        return None

    def apply_main_filter(self):
        """Apply the main filter to the main image and return QPixmap."""
        if self.main_image is not None and self.main_filter in self.filters:
            filtered_image = self.apply_filter(self.main_filter, self.main_image, self.filter_intensity)
            return self.convert_cv_qt(filtered_image)
        return None

    def apply_filter(self, filter_name, image, intensity=1.0):
        """Apply a specific filter to the image."""
        filter_func = self.filters.get(filter_name, self.show_original)
        filtered = filter_func(image, intensity)
        return filtered

    def convert_cv_qt(self, cv_img):
        """Convert OpenCV image to QPixmap."""
        if cv_img.ndim == 2:
            # Grayscale image
            q_image = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], QImage.Format.Format_Grayscale8)
        else:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

    # Filter Definitions
    def show_original(self, image, intensity=1.0):
        return image

    def apply_bandpass_filter(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        radius_outer, radius_inner = int(60 * intensity), int(10 * intensity)

        # Ensure radii are odd and <=31
        radius_outer = max(1, min(radius_outer, 31))
        if radius_outer % 2 == 0:
            radius_outer += 1
        radius_inner = max(1, min(radius_inner, 31))
        if radius_inner % 2 == 0:
            radius_inner += 1

        cv2.circle(mask, (ccol, crow), radius_outer, (1, 1), -1)
        cv2.circle(mask, (ccol, crow), radius_inner, (0, 0), -1)
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def apply_gabor_filter(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = int(21 * intensity)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 31)  # Limit ksize to 31
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), 8.0, np.pi/4, 10.0 * intensity, 0.5, 0)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
        return cv2.convertScaleAbs(filtered)

    def apply_laplacian(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = int(1 + 2 * intensity)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 31)  # Limit ksize to 31
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        return cv2.convertScaleAbs(laplacian)

    def apply_sobel_x(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = int(1 + 2 * intensity)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 31)  # Limit ksize to 31
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        return cv2.convertScaleAbs(sobelx)

    def apply_sobel_y(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = int(1 + 2 * intensity)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 31)  # Limit ksize to 31
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        return cv2.convertScaleAbs(sobely)

    def apply_scharr_x(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scharr uses a fixed kernel size of 3
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        return cv2.convertScaleAbs(scharrx * intensity)

    def apply_scharr_y(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(scharry * intensity)

    def apply_prewitt(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) * intensity
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) * intensity
        processed_x = cv2.filter2D(gray, cv2.CV_32F, kernelx)
        processed_y = cv2.filter2D(gray, cv2.CV_32F, kernely)
        processed = cv2.magnitude(processed_x, processed_y)
        return cv2.convertScaleAbs(processed)

    def apply_canny(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold1 = int(50 * intensity)
        threshold2 = int(150 * intensity)
        canny = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
