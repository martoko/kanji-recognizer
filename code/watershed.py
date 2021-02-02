import math

import cv2
import numpy as np

#
# i = 0
# def s(im, n=None):
#     global i
#     i += 1
#     out_path = f"/home/martoko/water/o_{i}_{n}.png"
#     cv2.imwrite(out_path, im)
#     return im
#
#
# in_path = "/home/martoko/water/i.jpg"
# in_org_path = "/home/martoko/water/i_org.jpg"
# im_org = cv2.imread(in_org_path)
# im = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
#
# markers = s(cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1], 'SMALL')
# markers = s(cv2.connectedComponents(markers)[1], "connect")
#
# im = cv2.bitwise_not(im)
#
# cimg = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
# markers = cv2.watershed(cimg, markers)
# s(cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET), 'final_markers')
# cimg[markers == -1] = [0, 0, 255]
# s(cimg, 'final')
#
# exit(0)
#
# t = s(cv2.threshold(im, 20, 255, cv2.THRESH_BINARY)[1], 'thresh')
#
# thresh = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)[1]
# s(thresh, 'r')
#
# kernel = np.ones((3, 3), np.uint8)
#
# sure_bg = cv2.dilate(thresh, kernel, iterations=3)
# s(sure_bg, 'sure_bg')
#
# dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
# s(dist, 'dist')
#
# sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)[1]
# s(sure_fg, 'sure_fg')
#
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
# s(unknown, 'unk')
#
# # Marker labelling
# markers = cv2.connectedComponents(sure_fg)[1]
# markers = markers + 1
# markers[unknown == 255] = 0
# s(cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET), 'markers')
#
# cimg = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
# markers = cv2.watershed(cimg, markers)
# s(cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET), 'final_markers')
# cimg[markers == -1] = [0, 0, 255]
# s(cimg, 'final')
#
# exit(0)
#
# ## ORG


i = 0


def s(im, n=None):
    global i
    i += 1
    i = 0
    out_path = f"/home/martoko/water/o_{i}_{n}.png"
    cv2.imwrite(out_path, im)
    return im


def d(im, n=None):
    cv2.imshow(str(n), im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def box_me(im):
    t = s(cv2.threshold(im, 20, 255, cv2.THRESH_BINARY)[1], 'thresh')

    thresh = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)[1]
    s(thresh, 'r')

    kernel = np.ones((3, 3), np.uint8)

    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    sure_bg = t
    s(sure_bg, 'sure_bg')

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    s(dist, 'dist')

    sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)[1]
    sure_fg = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1]
    s(sure_fg, 'sure_fg')

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    s(unknown, 'unk')

    # Marker labelling
    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1
    markers[unknown == 255] = 0
    s(cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET), 'markers')

    cimg = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(cimg, markers)
    s(cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET), 'final_markers')
    markered = np.copy(cimg)
    markered[markers == -1] = [0, 0, 255]
    s(markered, 'final')

    label_count = markers[...].max()
    labels = {}
    width = markers.shape[1]
    height = markers.shape[0]
    for y in range(height):
        for x in range(width):
            label = markers[y][x]
            if label not in labels:
                labels[label] = [
                    math.inf,  # min x
                    math.inf,  # min y
                    0,  # max x
                    0  # max y
                ]
            else:
                min_x, min_y, max_x, max_y = labels[label]
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
                labels[label] = [min_x, min_y, max_x, max_y]

    for label in labels:
        # print(label)
        min_x, min_y, max_x, max_y = labels[label]
        # border around sheds are 1 px
        min_x -= 1  # TODO: Bounds
        max_x += 1  # TODO: Bounds
        min_y -= 1  # TODO: Bounds
        max_y += 1  # TODO: Bounds
        labels[label] = [min_x, min_y, max_x, max_y]

    rect = cimg
    for label in labels:
        if label == -1 or label == 1: continue
        min_x, min_y, max_x, max_y = labels[label]
        rect = cv2.rectangle(cimg, (min_x, min_y), (max_x, max_y), (0, 0, 255))
    s(rect, "boxed")
    return [labels[label] for label in labels if label != -1 and label != 1]


if __name__ == '__main__':
    in_path = "/home/martoko/water/i.jpg"
    in_org_path = "/home/martoko/water/i_org.jpg"
    im_org = cv2.imread(in_org_path)
    im = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    print(box_me(im))
