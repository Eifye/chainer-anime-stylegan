import os

import numpy as np
import cv2

def imread(path, flags = cv2.IMREAD_ANYCOLOR):
    raw_data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw_data, flags)

    return img

def imwrite(path, img, params=None):
    ext = os.path.splitext(path)[1]

    _, raw_data = cv2.imencode(ext, img, params)
    with open(path, mode='w+b') as fp:
        raw_data.tofile(fp)
    
def dir_read(path, ext_filter=['.png', '.bmp', '.jpg']):

    dst_imdata = []
    dst_fnames = []

    files = os.listdir(path)

    for file in files:
        ext = os.path.splitext(file)[1].lower()

        if not ext in ext_filter:
            continue

        fpath = os.path.join(path, file)
        img = imread(fpath)

        dst_imdata.append(img)
        dst_fnames.append(fpath)

    return dst_fnames, dst_imdata

