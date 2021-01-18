import io
import math
import sys
from typing import *

import cv2
import torch
from PIL import Image, ImageDraw
from PIL.Image import NEAREST
from PySide6 import QtGui
import numpy as np
import PIL

from PySide6.QtGui import QGuiApplication, QCursor, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton,
                               QVBoxLayout, QWidget, QLineEdit)
from PySide6.QtCore import Slot, Qt, QEvent, QRect, QByteArray, QBuffer, QIODevice, QPoint
from matplotlib import pyplot
from torchvision.transforms import transforms

# from box_model import KanjiBoxer
import kanji
from craft import CRAFT, copyStateDict, test_net
from recognizer.model import KanjiRecognizer


class Qanji(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)

        # This hangs, show nice loading bar
        # self.ocr_reader = easyocr.Reader(['ja'], gpu=False)
        # self.boxer = KanjiBoxer(input_dimensions=32)
        # self.boxer.load_state_dict(torch.load('./box_saved_model.pt'))

        self.characters = kanji.frequent_kanji
        self.recog = KanjiRecognizer(output_dimensions=len(self.characters))
        self.recog.load_state_dict(torch.load('/home/martoko/saved_model.pt'))
        self.recog.eval()

        self.craft = CRAFT()
        self.craft.load_state_dict(
            copyStateDict(torch.load("/home/martoko/Code/CRAFT-pytorch/weights/craft_mlt_25k.pth", map_location='cpu')))
        self.craft.eval()

        self.setWindowTitle("Qanji")

        self.pixmap: Optional[QPixmap] = None

        self.button = QPushButton("Click me!")
        self.screenshot_label = QLabel()
        self.screenshot_label.setAlignment(Qt.AlignCenter)
        self.text = QLineEdit("While focus is on this window, press shift to perform OCR")
        self.text.setFont(QFont("sans serif", 32))
        self.text.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.screenshot_label)
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
        self.pixmap = self.clip_around(QCursor.pos(), 32)
        if self.pixmap is None:
            return

        # ocr = self.ocr_reader.readtext(self.pixmap_to_bytes(self.pixmap))
        # self.text.setText("\n".join([value for _, value, _ in ocr]))

        scaled = self.pixmap
        img = self.pixmap.toImage()
        # img = Image.frombytes("RGB", [self.pixmap.width, self.pixmap.height], self.pixmap)
        # img
        np_array = np.empty((3, 32, 32), dtype=np.uint8)
        for x in range(0, 32):
            for y in range(0, 32):
                c = img.pixelColor(x, y)
                np_array[0, y, x] = c.getRgb()[0]
                np_array[1, y, x] = c.getRgb()[1]
                np_array[2, y, x] = c.getRgb()[2]
        tensors = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            transforms.ToTensor()(
                # np_array
                self.pixmap_to_pil(scaled)
            )
        )
        pilimg = self.pixmap_to_pil(scaled)

        # load data
        image = pilimg
        image = np.array(image)

        bboxes, polys, score_text = test_net(self.craft, image, 0.7, 0.4, 0.4, False, False, None)
        print(bboxes)

        if len(bboxes) > 0:
            best_distance = 64
            best = None
            for box in bboxes:
                centroid = [
                    (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4,
                    (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
                ]
                distance = math.sqrt(
                    math.pow(centroid[0] - 32, 2) +
                    math.pow(centroid[1] - 32, 2)
                )
                if distance < best_distance:
                    best_distance = distance
                    best = box

            box = best
            print(len(bboxes))
            box = [
                box[0][0],  # left
                box[0][1],  # top
                box[2][0],  # right
                box[2][1],  # bottom
            ]
            pilimg = pilimg.resize((32, 32), box=list(box), resample=NEAREST)

        # box = self.boxer(tensors.reshape(-1, 3, 32, 32))
        # box = (box.detach().numpy() * 32)[0]
        # box[0] -= 1
        # box[1] -= 1
        # box[2] += 1
        # box[3] += 1
        # d = ImageDraw.Draw(pilimg)
        # d.rectangle(box, outline='red')
        # pyplot.imshow(pilimg)
        # pyplot.show()

        # pilimg = self.pixmap_to_pil(scaled)
        # pilimg = pilimg.resize((32, 32), box=list(box), resample=NEAREST)
        # pyplot.imshow(pilimg)
        # pyplot.show()
        tensors = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            transforms.ToTensor()(
                # np_array
                pilimg
            )
        )

        outputs = self.recog(tensors.reshape(-1, 3, 32, 32))
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
        #     self.screenshot_label.size(),
        #     Qt.KeepAspectRatio,
        #     Qt.SmoothTransformation
        # )
        self.screenshot_label.setPixmap(scaled_pixmap)

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
