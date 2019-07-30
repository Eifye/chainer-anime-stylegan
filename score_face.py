import os
import sys
import argparse
from itertools import chain

import numpy as np
import PIL
import cv2
import chainer

from externals.chainer_stylegan.src.stylegan import net

import lib.util

def convert_images2batch(x):
    x = x.astype(np.float32)/127.5 - 1
    x = x.transpose((0, 3, 1, 2))
    x = x[:,::-1]
    return x

def detect_face(images, cascade_file):
    cascade = cv2.CascadeClassifier(cascade_file)

    resized_dst = []
    org_dst = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30,30))

        resized_faces = []
        org_faces = []

        for rect in rects:
            roi = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            org_faces.append(roi)
            tmp = PIL.Image.fromarray(roi)
            tmp = tmp.resize((512, 512), PIL.Image.ANTIALIAS)
            roi = np.array(tmp)
            resized_faces.append(roi)

        org_dst.append(org_faces)
        resized_dst.append(resized_faces)

    return resized_dst, org_dst

def score_images(dis, images, stage=14):
    
    scores = []

    for ii in range(len(images)):

        batch = dis.xp.array(images[ii])[np.newaxis]
        batch = convert_images2batch(batch)
        score = dis(batch, stage)
        score.to_cpu()
        score = score.data

        scores.append(score)

    scores = np.array(scores)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/face_images')
    parser.add_argument('--model_path', type=str, default='data/chainer_model')
    parser.add_argument('--output_path', type=str, default='data/score_results')
    parser.add_argument('--detector_path', type=str, default='data/lbpcascade_animeface.xml')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    dis = net.Discriminator(512, True)
    ipath_dis = os.path.join(args.model_path, 'dis.npz')
    chainer.serializers.load_npz(ipath_dis, dis)

    if 0 <= args.gpu:
        chainer.cuda.get_device(args.gpu).use()
        dis.to_gpu()

    fnames, images = lib.util.dir_read(args.input_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.crop:
        _fnames = []
        faces_512, faces_org = detect_face(images, args.detector_path)
        for ii, face in enumerate(faces_512):
            for jj in range(len(face)):
                fname = os.path.splitext(os.path.basename(fnames[ii]))[0]
                _fname = '{0}_{1:03d}.png'.format(fname, jj)
                _fnames.append(_fname)
                _fname = os.path.join(args.output_path, _fname)
                lib.util.imwrite(_fname, face[jj])
                _fname = '{0}_{1:03d}_org.png'.format(fname, jj)
                _fname = os.path.join(args.output_path, _fname)
                lib.util.imwrite(_fname, faces_org[ii][jj])

        faces = list(chain.from_iterable(faces_512))
    else:
        faces = images
        _fnames = fnames

    scores = score_images(dis, faces)

    dpath = os.path.join(args.output_path, 'scores.txt')

    with open(dpath, 'w') as fp:
        for fname, score in zip(_fnames, scores):
            fp.write('{},{}\n'.format(fname, score))    

    pass

if __name__ == '__main__':
    main()