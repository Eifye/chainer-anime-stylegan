import os
import tqdm
import chainer
from PIL import Image
import numpy as np

def convert_batch_images(x, rows, cols):
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3)) 
    return x

class StyleGANSampler(object):

    def __init__(self, generator, latent_avg = None, num_avg_latent = 20000):
        self.gen = generator
        self.latent_avg = latent_avg
        self.num_avg_latent = num_avg_latent
        pass

    def _init_latent_avg(self):
        w_batch_size = 100
        n_batches = self.num_avg_latent // w_batch_size
        xp = self.gen.xp
        ch = self.gen.mapping.ch
        w_avg = xp.zeros(ch).astype('f')
        mapping = self.gen.mapping

        for _ in tqdm.tqdm(range(n_batches)):
            z = chainer.Variable(xp.asarray(mapping.make_hidden(w_batch_size)))
            w_cur = mapping(z)
            w_avg = w_avg + xp.average(w_cur.data, axis=0)
        w_avg = w_avg / n_batches

        self.latent_avg = w_avg

    def __call__(self, num, odir, truncation_psi = 0.7, stage = 14):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            if self.latent_avg is None:
                self._init_latent_avg()

            mapping = self.gen.mapping
            gen = self.gen.gen
            
            enable_trunction_trick = truncation_psi != 1.0

            for ii in tqdm.tqdm(range(num)):
                zz = mapping.make_hidden(1)
                latent = mapping(zz).data
            
                if enable_trunction_trick:
                    delta = latent - self.latent_avg
                    latent = delta * truncation_psi + self.latent_avg
                
                x = gen(latent, stage)
                
                x = chainer.cuda.to_cpu(x.data)
                x = convert_batch_images(x, 1, 1)
                preview_path = os.path.join(odir, "sample_{0:05d}.png".format(ii))
                Image.fromarray(x).save(preview_path)

            pass

    pass
