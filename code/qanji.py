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
import kanji
from craft import CRAFT, copyStateDict, test_net
import watershed
from recognizer.model import KanjiRecognizer


class Qanji(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)

        # This hangs, show nice loading bar
        # self.ocr_reader = easyocr.Reader(['ja'], gpu=False)
        # self.boxer = KanjiBoxer(input_dimensions=32)
        # self.boxer.load_state_dict(torch.load('./box_saved_model.pt'))

        self.characters = kanji.frequent_kanji_plus
        self.recog = torch.jit.load('/home/martoko/Code/kanji-recognizer/model.pt')
        self.recog.eval()

        self.craft = CRAFT()
        self.craft.load_state_dict(
            copyStateDict(torch.load("/home/martoko/Code/CRAFT-pytorch/weights/craft_mlt_25k.pth", map_location='cpu')))
        self.craft.eval()

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

        # ocr = self.ocr_reader.readtext(self.pixmap_to_bytes(self.pixmap))
        # self.text.setText("\n".join([value for _, value, _ in ocr]))

        # img = self.pixmap.toImage()
        # # img = Image.frombytes("RGB", [self.pixmap.width, self.pixmap.height], self.pixmap)
        # # img
        # np_array = np.empty((3, 64, 64), dtype=np.uint8)
        # for x in range(0, 64):
        #     for y in range(0, 64):
        #         c = img.pixelColor(x, y)
        #         np_array[0, y, x] = c.getRgb()[0]
        #         np_array[1, y, x] = c.getRgb()[1]
        #         np_array[2, y, x] = c.getRgb()[2]
        pilimg = self.pixmap_to_pil(self.pixmap)

        # load data
        image = pilimg
        image = np.array(image)

        bboxes, polys, score_text, craftimg = test_net(self.craft, image, 0.7, 0.4, 0.4, False, False, None)
        print(bboxes)

        craftimg = (craftimg * 255).astype(np.uint8)
        height, width = craftimg.shape
        bytesPerLine = width
        qim = QtGui.QImage(craftimg.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        scaled_pixmap = QPixmap.fromImage(qim)
        self.screenshot_label2.setPixmap(scaled_pixmap)

        watershedded = watershed.box_me(craftimg)
        print(watershedded)

        if len(watershedded) > 0:
            best_distance = 128
            best = None
            for box in watershedded:
                centroid = [
                    (box[0] + box[2]) / 2,
                    (box[1] + box[3]) / 2
                ]
                distance = math.sqrt(
                    math.pow(centroid[0] - 64 / 2, 2) +
                    math.pow(centroid[1] - 64 / 2, 2)
                )
                if distance < best_distance:
                    best_distance = distance
                    best = box

            watershed_box = best
            box = [
                watershed_box[0] / 64 * 128,
                watershed_box[1] / 64 * 128,
                watershed_box[2] / 64 * 128,
                watershed_box[3] / 64 * 128,
            ]
            print(box)
            pilimg = pilimg.resize((32, 32), box=list(box), resample=NEAREST)

            resized_craftimg = craftimg[int(watershed_box[1]):int(watershed_box[3]),
                               int(watershed_box[0]):int(watershed_box[2])]
            height, width = resized_craftimg.shape
            bytesPerLine = width
            qim = QtGui.QImage(np.ascontiguousarray(resized_craftimg).data, width, height, bytesPerLine,
                               QtGui.QImage.Format_Grayscale8)
            scaled_pixmap = QPixmap.fromImage(qim)
            self.screenshot_label3.setPixmap(scaled_pixmap)

        # if len(bboxes) > 0:
        #     best_distance = 64
        #     best = None
        #     for box in bboxes:
        #         centroid = [
        #             (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4,
        #             (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
        #         ]
        #         distance = math.sqrt(
        #             math.pow(centroid[0] - 32, 2) +
        #             math.pow(centroid[1] - 32, 2)
        #         )
        #         if distance < best_distance:
        #             best_distance = distance
        #             best = box
        #
        #     box = best
        #     print(len(bboxes))
        #     box = [
        #         box[0][0],  # left
        #         box[0][1],  # top
        #         box[2][0],  # right
        #         box[2][1],  # bottom
        #     ]
        #
        #     print(box)
        #     print(craftimg.shape)
        #     print(craftimg)
        #     resized_craftimg = craftimg[int(box[0] / 64 * 32):int(box[2] / 64 * 32),
        #                        int(box[1] / 64 * 32):int(box[3] / 64 * 32)]
        #     pilimg = pilimg.resize((32, 32), box=list(box), resample=NEAREST)
        #     print(resized_craftimg.shape)
        #     print(resized_craftimg)
        #     watershedded = watershed.box_me(resized_craftimg)
        #     print(watershedded)
        #
        #     if len(watershedded) > 0:
        #         h = resized_craftimg.shape[0]
        #         w = resized_craftimg.shape[0]
        #         box = [
        #             watershedded[0][0] / w * 32,
        #             watershedded[0][1] / h * 32,
        #             watershedded[0][2] / w * 32,
        #             watershedded[0][3] / h * 32,
        #         ]
        #         pilimg = pilimg.resize((32, 32), box=list(box), resample=NEAREST)
        #
        #     height, width = resized_craftimg.shape
        #     bytesPerLine = width
        #     qim = QtGui.QImage(np.ascontiguousarray(resized_craftimg).data, width, height, bytesPerLine,
        #                        QtGui.QImage.Format_Grayscale8)
        #     scaled_pixmap = QPixmap.fromImage(qim)
        #     self.screenshot_label3.setPixmap(scaled_pixmap)

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
