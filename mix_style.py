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

def load_latent(path, stage):
    data = np.load(path, allow_pickle=True)
    
    data = data['arr_0']
    return data

    data = data['arr_0'].tolist()
    k = (stage-2)//2+2 if stage%2 == 0 else (stage-1)//2+1
    dst = []

    for ii in range(k):
        layer = []
        for jj in range(2):
            key = '{}_{}'.format(ii, jj)
            dd = data[key]
            layer.append(dd)
        dst.append(layer)

    return dst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lim', type=int, default=3)
    parser.add_argument('--input_a', type=str, default='data/face_image.png')
    parser.add_argument('--input_b', type=str, default='data/face_image.png')
    parser.add_argument('--model_path', type=str, default='data/chainer_model')
    parser.add_argument('--output_path', type=str, default='data/mix_results')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    gen = net.Generator(512)

    ipath_gen = os.path.join(args.model_path, 'gen.npz')
    chainer.serializers.load_npz(ipath_gen, gen)

    if 0 <= args.gpu:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()

    data_a = load_latent(args.input_a, 14)
    data_b = load_latent(args.input_b, 14)
    data_a = gen.xp.array(data_a)
    data_b = gen.xp.array(data_b)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #for ii in range(args.lim):
        #data_b[ii] = None

    #data_a = 0.5*data_a + 0.5*data_b
    #img = gen.gen.generate_with_latents(data_a, 14)

    #latent_a = gen.mapping(data_a)
    #latent_b = gen.mapping(data_b)
    
    img = gen.gen(data_a, 14, w2=data_b, _lim = args.lim)

    fname_a = os.path.splitext(os.path.basename(args.input_a))[0]
    fname_b = os.path.splitext(os.path.basename(args.input_b))[0]

    dpath = os.path.join(args.output_path, '({})({})(lim_{}).png'.format(fname_a, fname_b, args.lim))

    img = chainer.cuda.to_cpu(img.data[0])
    img = np.asarray(np.clip(img*127.5+127.5, 0.0, 255.0), dtype=np.uint8)
    img = img.transpose((1, 2, 0))
    img = img[:,:,::-1]
    lib.util.imwrite(dpath, img)

    pass

if __name__ == '__main__':
    main()