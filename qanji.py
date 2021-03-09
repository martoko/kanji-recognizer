import io
import math
import sys
from typing import *

import PIL
import numpy as np
import torch
from PIL import Image
from PIL.Image import NEAREST
from PySide6 import QtGui
from PySide6.QtCore import Slot, Qt, QEvent, QRect, QByteArray, QBuffer, QIODevice, QPoint
from PySide6.QtGui import QGuiApplication, QCursor, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton,
                               QVBoxLayout, QWidget, QLineEdit)
from torchvision.transforms import transforms

# from box_model import KanjiBoxer
from recognizer.data import character_sets
from recognizer.model import KanjiRecognizer


class Qanji(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)

        # This hangs, show nice loading bar
        # self.ocr_reader = easyocr.Reader(['ja'], gpu=False)
        # self.boxer = KanjiBoxer(input_dimensions=32)
        # self.boxer.load_state_dict(torch.load('./box_saved_model.pt'))

        self.characters = character_sets.frequent_kanji_plus
        self.recog = KanjiRecognizer.load_from_checkpoint('/home/martoko/Code/kanji-recognizer/rare-durian-72.ckpt')
        self.recog.eval()

        self.setWindowTitle("Qanji")

        self.pixmap: Optional[QPixmap] = None

        self.button = QPushButton("Click me!")
        self.screenshot_label_org = QLabel()
        self.screenshot_label_org.setAlignment(Qt.AlignCenter)
        self.screenshot_label2 = QLabel()
        self.screenshot_label2.setAlignment(Qt.AlignCenter)
        self.screenshot_label3 = QLabel()
        self.screenshot_label3.setAlignment(Qt.AlignCenter)
        self.screenshot_label_final = QLabel()
        self.screenshot_label_final.setAlignment(Qt.AlignCenter)
        self.text = QLineEdit("While focus is on this window, press shift to perform OCR")
        self.text.setFont(QFont("sans serif", 32))
        self.text.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.screenshot_label_org)
        self.layout.addWidget(self.screenshot_label2)
        self.layout.addWidget(self.screenshot_label3)
        self.layout.addWidget(self.screenshot_label_final)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        # Connecting the signal
        # self.button.clicked.connect(self.new_screenshot)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.type() != QEvent.KeyPress:
            return
        if event.key() != Qt.Key_Shift:
            return
        self.shoot_screen()

    @Slot()
    def shoot_screen(self) -> None:
        self.pixmap = self.clip_around(QCursor.pos(), 128)
        self.screenshot_label_org.setPixmap(self.pixmap)
        if self.pixmap is None:
            return
        pilimg = self.pixmap_to_pil(self.pixmap)

        # load data
        image = pilimg
        image = np.array(image)

        tensors = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            transforms.ToTensor()(
                # np_array
                pilimg
            )
        )

        outputs = self.recog(tensors.reshape(-1, 3, 128, 128))
        best_confidence, best_indices = torch.sort(outputs, 1, descending=True)
        # print(best_confidence)
        # print(best_confidence[0][0])
        # print(best_indices[0][0])
        # print(self.characters[best_indices[0][0]])
        _, predicted = torch.max(outputs, 1)
        # ocr = self.characters[predicted]
        # print(ocr)
        ocr2 = [self.characters[best_indices[0][i]] for i in range(5)]
        print(ocr2)
        self.text.setText("　".join(ocr2))

        im = pilimg.convert("RGB")
        data = im.tobytes("raw", "RGB")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qim)
        # scaled_pixmap = scaled  # .scaled(
        #     self.screenshot_label_final.size(),
        #     Qt.KeepAspectRatio,
        #     Qt.SmoothTransformation
        # )
        self.screenshot_label_final.setPixmap(scaled_pixmap)

    @staticmethod
    def clip_around(point: QPoint, size: int) -> Optional[QPixmap]:
        screen = QGuiApplication.screenAt(point)
        screen_geometry = screen.geometry()
        clip_geometry = QRect(
            point.x() - size / 2, point.y() - size / 2,
            size, size
        )

        if clip_geometry.left() < screen_geometry.left():
            clip_geometry.moveLeft(screen_geometry.left())

        if clip_geometry.right() > screen_geometry.right():
            clip_geometry.moveRight(screen_geometry.right())

        if clip_geometry.top() < screen_geometry.top():
            clip_geometry.moveTop(screen_geometry.top())

        if clip_geometry.bottom() > screen_geometry.bottom():
            clip_geometry.moveBottom(screen_geometry.bottom())

        if not screen_geometry.contains(clip_geometry):
            print("Clip size is larger than screen size")
            return None

        clip_geometry.moveTopLeft(clip_geometry.topLeft() - screen_geometry.topLeft())

        return screen.grabWindow(
            0,
            x=clip_geometry.x(),
            y=clip_geometry.y(),
            w=clip_geometry.width(),
            h=clip_geometry.height()
        )

    @staticmethod
    def pixmap_to_bytes(pixmap: QPixmap) -> bytes:
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        return byte_array.data()

    @staticmethod
    def pixmap_to_pil(pixmap: QPixmap) -> PIL.Image.Image:
        return Image.open(io.BytesIO(Qanji.pixmap_to_bytes(pixmap)))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = Qanji()
    widget.show()

    sys.exit(app.exec_())
