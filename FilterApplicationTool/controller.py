# controller.py

from view import FilterApplicationView
from model import FilterApplicationModel
from PyQt6.QtCore import Qt

class FilterController:
    def __init__(self):
        self.model = FilterApplicationModel()
        self.view = FilterApplicationView(self)
        self.view.show()

    def load_image(self, from_directory=False):
        """Load image(s) from file or directory and update the view."""
        images_with_filters = self.model.load_images(from_directory=from_directory)
        if images_with_filters:
            self.view.update_image_list(images_with_filters)
            # Set the first filter as the default main image
            if images_with_filters:
                first_filter_name = images_with_filters[0][1]
                self.set_main_image_by_filter(first_filter_name)
        else:
            self.view.show_error_message("No images were loaded. Please check the selected path.")

    def update_filter_intensity(self, intensity):
        """Update filter intensity and apply it to the main image."""
        filtered_pixmap = self.model.update_filter_intensity(intensity)
        if filtered_pixmap:
            self.view.update_main_image(filtered_pixmap)

    def set_main_image_by_filter(self, filter_name):
        """Set the main image based on the selected filter."""
        self.model.set_main_image_by_filter(filter_name)
        filtered_pixmap = self.model.apply_main_filter()
        if filtered_pixmap:
            self.view.update_main_image(filtered_pixmap)
            current_intensity = self.model.filter_intensity
            self.view.set_slider_value(current_intensity)
