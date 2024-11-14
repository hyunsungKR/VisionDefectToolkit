# view.py

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea, QSlider, QFileDialog, QMessageBox, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QMouseEvent

class FilterPreviewLabel(QLabel):
    """Custom QLabel to display filter preview and handle click events."""
    clicked = pyqtSignal(str)  # Emits the filter name when clicked

    def __init__(self, pixmap, filter_name):
        super().__init__()
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filter_name = filter_name
        self.setScaledContents(True)  # Allow pixmap to scale with QLabel size

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.filter_name)

class FilterApplicationView(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Filter Application Tool")
        self.resize(1200, 800)
        self.initUI()

    def initUI(self):
        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

        # Left filter preview area
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.image_grid_layout = QGridLayout(self.scroll_area_widget)
        self.image_grid_layout.setSpacing(5)  # Reduced spacing
        self.image_grid_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        self.scroll_area_widget.setLayout(self.image_grid_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.main_layout.addWidget(self.scroll_area, 3)  # Ratio: left 3, right 2

        # List to keep track of filter preview widgets
        self.image_widgets = []  # List of tuples: (FilterPreviewLabel, QLabel)

        # Right control area
        self.control_layout = QVBoxLayout()

        # Main image display
        self.main_image_display = QLabel("Load an image to start.")
        self.main_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_image_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_image_display.setStyleSheet("border: 1px solid black;")
        # Ensure main_image_display does not change size dynamically
        self.main_image_display.setMinimumSize(400, 400)
        self.main_image_display.setMaximumSize(1000, 1000)
        self.control_layout.addWidget(self.main_image_display)

        # Slider for filter intensity
        self.slider_label = QLabel("Filter Intensity:")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(10)
        self.slider.setValue(5)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(lambda value: self.controller.update_filter_intensity(value / 5.0))
        self.control_layout.addWidget(self.slider_label)
        self.control_layout.addWidget(self.slider)

        # File load buttons
        load_buttons_layout = QHBoxLayout()

        load_file_button = QPushButton("Load Image")
        load_file_button.clicked.connect(lambda: self.controller.load_image(from_directory=False))
        load_buttons_layout.addWidget(load_file_button)

        load_dir_button = QPushButton("Load Directory")
        load_dir_button.clicked.connect(lambda: self.controller.load_image(from_directory=True))
        load_buttons_layout.addWidget(load_dir_button)

        self.control_layout.addLayout(load_buttons_layout)

        # Add control layout to main layout
        self.main_layout.addLayout(self.control_layout, 2)  # Ratio: left 3, right 2

        # Handle window resize
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        """Handle window resize event to adjust image sizes."""
        self.update_image_sizes()
        event.accept()

    def update_image_sizes(self):
        """Adjust the size of filter preview images based on the current window size."""
        if not self.image_widgets:
            return

        columns = 5
        rows = 2
        max_filters = columns * rows

        # Calculate available width and height in the scroll area
        available_width = self.scroll_area_widget.width()
        available_height = self.scroll_area_widget.height()

        # Calculate image sizes based on available width
        spacing = self.image_grid_layout.spacing()
        margins = self.image_grid_layout.contentsMargins()
        total_spacing_width = spacing * (columns - 1) + margins.left() + margins.right()
        image_width = (available_width - total_spacing_width) // columns

        # Ensure a 4:3 aspect ratio
        image_height = int(image_width * 0.75)

        # Set minimum size
        min_width, min_height = 80, 60
        image_width = max(image_width, min_width)
        image_height = max(image_height, min_height)

        for image_label, _ in self.image_widgets:
            image_label.setFixedSize(QSize(image_width, image_height))

    def update_image_list(self, images_with_filters):
        """Update the filter preview list on the left."""
        # Clear existing widgets
        for image_label, filter_label in self.image_widgets:
            image_label.deleteLater()
            filter_label.deleteLater()
        self.image_widgets.clear()

        # Add new widgets in a 5x2 grid
        columns = 5
        max_filters = 10  # 5 columns x 2 rows

        for idx, (pixmap, filter_name) in enumerate(images_with_filters[:max_filters]):
            # Create filter preview label
            filter_preview = FilterPreviewLabel(pixmap, filter_name)
            filter_preview.clicked.connect(lambda fname=filter_name: self.controller.set_main_image_by_filter(fname))

            # Create filter name label
            filter_label = QLabel(filter_name)
            filter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Determine grid position
            row = (idx // columns) * 2
            col = idx % columns
            self.image_grid_layout.addWidget(filter_preview, row, col)
            self.image_grid_layout.addWidget(filter_label, row + 1, col)

            # Add to the list
            self.image_widgets.append((filter_preview, filter_label))

        self.update_image_sizes()

    def update_main_image(self, pixmap):
        """Update the main image display on the right."""
        # Scale pixmap to label size, maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.main_image_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.main_image_display.setPixmap(scaled_pixmap)

    def set_slider_value(self, value):
        """Set the slider's value without emitting signals."""
        if self.slider is not None:
            self.slider.blockSignals(True)
            self.slider.setValue(int(value * 5))
            self.slider.blockSignals(False)

    def show_error_message(self, message):
        """Display an error message to the user."""
        QMessageBox.critical(self, "Error", message)
