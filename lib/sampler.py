import os
import tqdm
import chainer
import chainer.optimizers
import chainer.functions as F
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

    def __init__(self, generator, latent_avg = None, num_avg_latent = 20000, dis = None):
        self.gen = generator
        self.latent_avg = latent_avg
        self.num_avg_latent = num_avg_latent
        self.dis = dis
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

    def _maximize_score(self, latent, stage, truncation_psi, niter=70):

        opt = chainer.optimizers.RMSprop(0.001)
        _latent = chainer.links.Parameter(latent)

        if self.gen.gen.xp != np:
            _latent.to_gpu()

        opt.setup(_latent)

        for ii in range(niter):
            
            latent_i = self.gen.mapping(_latent())
        
            latent_i = self.truncation_trick(latent_i, truncation_psi)

            img = self.gen.gen(latent_i, stage)
            score = -self.dis(img, stage)
            score = F.mean(score)
            opt.target.zerograds()
            score.backward()
            opt.update()

            if float(score.data) < -17.:
                break

            if 1:
                ## debug
                import cv2
                _img = chainer.cuda.to_cpu(img.data[:,::-1])
                _img = convert_batch_images(_img, 1, 1)
                print(ii, "score=", score.data)
                cv2.imshow('dgb', _img)
                cv2.waitKey(10)

        return _latent().data, img

    def truncation_trick(self, latent, truncation_psi):
        enable_trunction_trick = truncation_psi != 1.0

        if enable_trunction_trick:
            delta = latent - self.latent_avg
            _latent = delta * truncation_psi + self.latent_avg
            return _latent
        else:
            return latent

    def __call__(self, num, odir, truncation_psi = 0.7, stage = 14):

        if self.latent_avg is None:
            self._init_latent_avg()

        mapping = self.gen.mapping
        gen = self.gen.gen
        
        for ii in tqdm.tqdm(range(num)):
            zz = mapping.make_hidden(1)
            
            if None is self.dis:
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    latent = mapping(zz).data
                    latent = self.truncation_trick(latent, truncation_psi)
                    x = gen(latent, stage)
            else:
                zz, x = self._maximize_score(zz, stage, truncation_psi)
            
            x = chainer.cuda.to_cpu(x.data)
            x = convert_batch_images(x, 1, 1)
            preview_path = os.path.join(odir, "sample_{0:05d}.png".format(ii))
            Image.fromarray(x).save(preview_path)

            preview_path = os.path.join(odir, "latent_{0:05d}.npz".format(ii))
            zz = chainer.cuda.to_cpu(zz)[0]
            np.savez(preview_path, zz)

    pass
