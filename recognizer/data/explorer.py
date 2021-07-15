import sys
from typing import *

import PIL
from PySide6 import QtGui
from PySide6.QtCore import Slot, Qt, QEvent
from PySide6.QtGui import QPixmap, QPalette, QColor
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton,
                               QVBoxLayout, QWidget, QHBoxLayout, QSlider, QSpinBox, QColorDialog, QLineEdit)

from recognizer.data import character_sets
from recognizer.data.training_dataset import RecognizerTrainingDataset


class IntParameter(QWidget):
    def __init__(self, name: str, default, minimum, maximum):
        QWidget.__init__(self)
        self.value = default

        self.layout = QHBoxLayout()

        self.label = QLabel(name)
        self.layout.addWidget(self.label)

        self.slider = QSlider()
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setValue(self.value)
        self.slider.valueChanged.connect(self.on_value_changed)
        self.layout.addWidget(self.slider)

        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(0)
        self.spin_box.setMaximum(2147483647)
        self.spin_box.setValue(self.value)
        self.spin_box.valueChanged.connect(self.on_value_changed)
        self.layout.addWidget(self.spin_box)

        self.setLayout(self.layout)
        self.value_changed = None

    @Slot()
    def on_value_changed(self, value):
        self.value = value
        self.slider.setValue(value)
        self.spin_box.setValue(value)
        self.value_changed()


class ColorParameter(QWidget):
    def __init__(self, name: str, default: List[int]):
        QWidget.__init__(self)
        self.value = default

        self.layout = QHBoxLayout()

        self.label = QLabel(name)
        self.layout.addWidget(self.label)

        # self.line_edit = QLineEdit()
        # self.line_edit.setText(str(self.value))
        # self.line_edit.textChanged.connect(self.on_text_changed)
        # self.layout.addWidget(self.line_edit)

        self.button_pixmap = QPixmap(32, 32)
        self.button_pixmap.fill(QColor.fromRgb(self.value[0], self.value[1], self.value[2], 255))
        self.button = QPushButton()
        self.button.pressed.connect(self.on_button_pressed)
        self.button.setIcon(self.button_pixmap)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)
        self.value_changed = None

    # @Slot()
    # def on_text_changed(self, value):
    #     print("TODO")
    #     # self.value = value
    #     self.value_changed()

    @Slot()
    def on_color_selected(self, color):
        self.value = color.getRgb()[:3]
        # self.line_edit.setText(color.name())
        self.button_pixmap = QPixmap(32, 32)
        self.button_pixmap.fill(QColor.fromRgb(self.value[0], self.value[1], self.value[2], 255))
        self.button.setIcon(self.button_pixmap)
        self.value_changed()

    @Slot()
    def on_button_pressed(self):
        self.color_picker = QColorDialog()
        self.color_picker.setCurrentColor(QColor.fromRgb(self.value[0], self.value[1], self.value[2], 255))
        self.color_picker.colorSelected.connect(self.on_color_selected)
        self.color_picker.show()


class DataExplorer(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)

        self.dataset = RecognizerTrainingDataset("data", character_sets.frequent_kanji_plus)

        self.setWindowTitle("Data Explorer")

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.parameters_layout = QVBoxLayout()
        self.layout.addLayout(self.parameters_layout)

        self.parameters = {
            "stage": IntParameter("Stage", 0, 0, 10),

            "width": IntParameter("width", 128, 32, 1024),
            "height": IntParameter("height", 128, 32, 1024),
            "background_color": ColorParameter("background_color", QColor("white").getRgb()[:3]),
            "font_size": IntParameter("font_size", 10, 0, 100),
            "font_color": ColorParameter("font_color", QColor("black").getRgb()[:3])
        }
        for parameter in self.parameters.values():
            parameter.value_changed = self.regenerate
            self.parameters_layout.addWidget(parameter)

        self.regenerate_button = QPushButton("Regenerate")
        self.parameters_layout.addWidget(self.regenerate_button)

        self.sample_layout = QVBoxLayout()
        self.layout.addLayout(self.sample_layout)

        self.sample_pixmap: Optional[QPixmap] = None
        self.sample_label = QLabel()
        self.sample_label.setAlignment(Qt.AlignCenter)
        self.sample_layout.addWidget(self.sample_label)

        self.region_score_pixmap: Optional[QPixmap] = None
        self.region_score_label = QLabel()
        self.region_score_label.setAlignment(Qt.AlignCenter)
        self.sample_layout.addWidget(self.region_score_label)

        self.character_label = QLabel()
        self.character_label.setAlignment(Qt.AlignCenter)
        self.sample_layout.addWidget(self.character_label)

        self.old_sample_layout = QVBoxLayout()
        self.layout.addLayout(self.old_sample_layout)

        self.old_sample_pixmap: Optional[QPixmap] = None
        self.old_sample_label = QLabel()
        self.old_sample_label.setAlignment(Qt.AlignCenter)
        self.old_sample_layout.addWidget(self.old_sample_label)

        self.old_region_score_pixmap: Optional[QPixmap] = None
        self.old_region_score_label = QLabel()
        self.old_region_score_label.setAlignment(Qt.AlignCenter)
        self.old_sample_layout.addWidget(self.old_region_score_label)

        self.old_character_label = QLabel()
        self.old_character_label.setAlignment(Qt.AlignCenter)
        self.old_sample_layout.addWidget(self.old_character_label)

        self.regenerate_button.clicked.connect(self.regenerate)

        self.regenerate()

    @Slot()
    def regenerate(self) -> None:
        self.dataset.stage = self.parameters["stage"].value
        old_sample, old_label, old_region_score = self.dataset.generate()
        self.old_sample_label.setPixmap(self.to_pixmap(old_sample))
        self.old_region_score_label.setPixmap(self.to_pixmap(old_region_score.convert("RGB")))
        self.old_character_label.setText(self.dataset.characters[old_label])

        # sample, label, region_score = self.dataset.new_generate_stage_X(
        #     self.parameters["width"].value,
        #     self.parameters["height"].value,
        #     self.parameters["background_color"].value,
        #     self.parameters["font_size"].value,
        #     self.parameters["font_color"].value,
        # )
        sample, label, region_score = self.dataset.new_generate(self.parameters["stage"].value)
        self.sample_label.setPixmap(self.to_pixmap(sample))
        self.region_score_label.setPixmap(self.to_pixmap(region_score.convert("RGB")))
        self.character_label.setText(self.dataset.characters[label])

    @staticmethod
    def to_pixmap(image: PIL.Image):
        data = image.tobytes("raw", "RGB")
        qim = QtGui.QImage(data, image.size[0], image.size[1], QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(qim)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.type() != QEvent.KeyPress:
            return
        if event.key() != Qt.Key_R:
            return
        self.regenerate()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = DataExplorer()
    widget.show()

    sys.exit(app.exec_())
