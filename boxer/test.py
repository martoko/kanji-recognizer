import argparse
import glob
import os

import cv2 as cv

import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch

import torchvision.transforms

import pytorch_lightning as plt
from pytorch_lightning.loggers import WandbLogger

from boxer.model import KanjiBoxer
from recognizer.data.data_module import RecognizerDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model finding the location of kanji.")
    parser.add_argument("-m", "--model-path", type=str,
                        help="path to a checkpoint")
    parser.add_argument("--data-folder", type=str, default="data",
                        help="path to a folder containing train/val/test data (default: data)")
    args = parser.parse_args()

    model = KanjiBoxer.load_from_checkpoint(args.model_path)

    paths = glob.glob("/home/martoko/Code/kanji-recognizer/data/free-kanji/*/1.png")
    print(paths)
    for p in paths:
        outpath = os.path.splitext(p)[0] + "_gen" + os.path.splitext(p)[1]
        outpath2 = os.path.splitext(p)[0] + "_gen2" + os.path.splitext(p)[1]
        outpath3 = os.path.splitext(p)[0] + "_gen3" + os.path.splitext(p)[1]
        image = PIL.Image.open(p).convert('RGB')
        # image = PIL.ImageChops.offset(image, xoffset=-10, yoffset=-10)
        i = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        print(i.shape)
        o, _ = model(i)
        o = o.squeeze(0).repeat(3, 1, 1).clamp(0, 1)
        print(o.shape)
        po = torchvision.transforms.ToPILImage()(o).resize((128, 128))
        po.save(outpath)

        cvo = np.array(po)[:, :, :1].copy()

        kernel = np.ones((5, 5), np.uint8)
        # erosion = cv.erode(cvo, kernel, iterations=1)
        opened = cv.morphologyEx(cvo, cv.MORPH_OPEN, kernel, iterations=1)

        # erode/dilate/opening/closing.
        cv.imwrite(outpath2, opened)

        ret, thresh = cv.threshold(opened, 64, 255, 0)
        print(thresh.dtype)
        print(thresh.shape)
        contours, hierarchy = cv.findContours(thresh, 1, 2)
        cnt = contours[0]
        x, y, w, h = cv.boundingRect(cnt)
        expand = 2
        x -= expand
        y -= expand
        w += expand * 2
        h += expand * 2
        cvi = cv.imread(p)
        lbelled = cv.rectangle(cvi, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.imwrite(outpath3, lbelled)
