import os
import sys
import argparse
import numpy as np

import chainer

from externals.chainer_stylegan.src.stylegan import net

import lib.sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/chainer_model')
    parser.add_argument('--output_path', type=str, default='data/random_samples')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--maximize_score', action='store_true')

    args = parser.parse_args()

    gen = net.Generator(512)

    ipath_gen = os.path.join(args.input_path, 'gen.npz')
    chainer.serializers.load_npz(ipath_gen, gen)

    if args.maximize_score:
        dis = net.Discriminator(512, True)
        ipath_dis = os.path.join(args.input_path, 'dis.npz')
        chainer.serializers.load_npz(ipath_dis, dis)
    else:
        dis = None

    ipath_wavg = os.path.join(args.input_path, 'w_avg.npz') 
    w_avg = np.load(ipath_wavg)['arr_0']

    if 0 <= args.gpu:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        if not dis is None:
            dis.to_gpu()
        w_avg = chainer.cuda.to_gpu(w_avg)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    np.random.seed(5)

    sampler = lib.sampler.StyleGANSampler(gen, w_avg, dis=dis)

    sampler(args.num, args.output_path)

    pass

if __name__ == '__main__':
    main()