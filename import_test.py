import sys
import lzma
import pickle
import numpy as np

from externals.tensorflow_stylegan import dnnlib
from externals.tensorflow_stylegan.dnnlib import tflib

from lib.import_tf_stylegan import import_generator
import lib.sampler

sys.modules['dnnlib'] = dnnlib

decoder = lzma.open('data/2019-03-08-stylegan-animefaces-network-02051-021980.pkl.xz')
compressed_data = decoder.read()

tflib.init_tf()
G, D, Gs = pickle.loads(compressed_data)

gen, w_avg = import_generator(Gs)

np.random.seed(5)
sampler = lib.sampler.StyleGANSampler(gen, w_avg)
sampler(100, 'results')