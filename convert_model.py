import os
import sys
import lzma
import pickle
import argparse
import numpy as np

import chainer

from externals.tensorflow_stylegan import dnnlib
from externals.tensorflow_stylegan.dnnlib import tflib

from lib.import_tf_stylegan import import_generator, import_discriminator

sys.modules['dnnlib'] = dnnlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/2019-03-08-stylegan-animefaces-network-02051-021980.pkl.xz')
    parser.add_argument('--output_path', type=str, default='data/chainer_model')

    args = parser.parse_args()

    decoder = lzma.open(args.input_path)
    compressed_data = decoder.read()

    tflib.init_tf()
    _, D, Gs = pickle.loads(compressed_data)

    gen, w_avg = import_generator(Gs)
    dis = import_discriminator(D)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    opath_avg = os.path.join(args.output_path, 'w_avg.npz')
    np.savez(opath_avg, w_avg)

    opath_gen = os.path.join(args.output_path, 'gen.npz')
    chainer.serializers.save_npz(opath_gen, gen)

    opath_dis = os.path.join(args.output_path, 'dis.npz')
    chainer.serializers.save_npz(opath_dis, dis)

    pass

if __name__ == '__main__':
    main()