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

VGG_FNAMES = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']
VGG_WEIGHT = [0., 0., 0., 0.5, 0.5]

def convert_image2batch(x):
    x = x[np.newaxis]
    x = x.astype(np.float32)/127.5 - 1
    x = x.transpose((0, 3, 1, 2))
    
    return x

def make_hidden_link(gen, w_avg, stage):
    w_avg = w_avg[np.newaxis]

    # 共通の一つだけ    
    return chainer.links.Parameter(w_avg)

    # 層毎に別
    # k = (stage-2)//2+2 if stage%2 == 0 else (stage-1)//2+1
    # zz = chainer.functions.repeat(w_avg, k*2, axis=0)
    # zz = zz.reshape((k, 2, 1, -1)).data
    # dst = chainer.links.Parameter(zz)
    # return dst

def vgg_feature(vgg, image):
    if len(image.shape) == 3:
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis].astype(np.float32)
        image = vgg.xp.asarray(image)

    image = chainer.functions.resize_images(image, (224, 224))
    image = image[:,::-1,:,:]
    mean = vgg.xp.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1, -1, 1, 1))
    image = image - mean

    feature = vgg(image, layers=VGG_FNAMES)

    return feature

def find_latent(gen, dis, image, w_avg, odir, resumt = None, truncation_psi = 0.7, niter = 3000, stage=14, ch=512, lambda1 = 0.0001):
    LR_MAX = 0.02
    LR_MIN = 0.0002
    batch = convert_image2batch(image)
    zz_link = make_hidden_link(gen, w_avg, stage)
    opt = chainer.optimizers.RMSprop(LR_MAX)
    vgg = chainer.links.VGG16Layers().to_gpu()
    hh, ww, _ = image.shape

    if gen.xp != np:
        zz_link.to_gpu()
        batch = chainer.cuda.to_gpu(batch)

    opt.setup(zz_link)

    target_feature = vgg_feature(vgg, image)

    def perceptual_loss(image, target_feature):
        var_feature = vgg_feature(vgg, image)
        loss = 0

        for ii, fname in enumerate(VGG_FNAMES):
            loss += VGG_WEIGHT[ii] * chainer.functions.mean_squared_error(var_feature[fname], target_feature[fname])

        return loss

    def link2param(zz_link):
        return zz_link()

    print("iter,total_loss,piz_loss,perseptual_loss,discriminator_loss")

    tmp = chainer.functions.resize_images(batch, (512,512))
    target_dis = dis._hs(tmp.data, stage)

    for ii in range(niter):
        alpha = LR_MIN + 0.5*(LR_MAX-LR_MIN)*(1+np.cos(ii/niter*np.pi))
        opt.lr = alpha

        zz = link2param(zz_link)
        img = gen.gen(zz, stage)
        #層ごとに別特徴の場合
        #img = gen.gen.generate_with_latents(zz, stage)
        var_dis = dis._hs(img, stage)
        img = chainer.functions.resize_images(img, (hh, ww))

        disl = 0
        for kk in range(4, len(var_dis)):
            disl+=chainer.functions.mean_squared_error(var_dis[kk], target_dis[kk])
        disl = 0.0001*disl
        _img = img.data.copy()
        
        pix_loss = chainer.functions.mean_squared_error(img, batch)        

        img = img*127.5+127.5
        per_loss = lambda1*perceptual_loss(img, target_feature)
        loss = pix_loss + per_loss + disl

        opt.target.zerograds()
        loss.backward()
        opt.update()

        print("{},{},{},{},{}".format(ii, loss.data,pix_loss.data, per_loss.data, disl.data))

        if (ii%100)==0:
            ## debug
            _batch = chainer.cuda.to_cpu(batch[0, ::-1])
            _img = chainer.cuda.to_cpu(_img[0, ::-1])
            _img = np.concatenate((_img, _batch), axis=2)
            _img = np.asarray(np.clip(_img*127.5+127.5, 0.0, 255.0), dtype=np.uint8)
            _img = _img.transpose((1, 2, 0))
            dpath = os.path.join(odir, '{0:06d}.png'.format(ii))
            lib.util.imwrite(dpath, _img)

    latent = chainer.cuda.to_cpu(link2param(zz_link).data)
    img = chainer.cuda.to_cpu(img.data)
    return latent, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/face_image.png')
    parser.add_argument('--model_path', type=str, default='data/chainer_model')
    parser.add_argument('--output_path', type=str, default='data/found_latents')
    parser.add_argument('--lambda1', type=float, default=0.000008)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    gen = net.Generator(512)

    ipath_gen = os.path.join(args.model_path, 'gen.npz')
    chainer.serializers.load_npz(ipath_gen, gen)

    dis = net.Discriminator(512, True)
    ipath_dis = os.path.join(args.model_path, 'dis.npz')
    chainer.serializers.load_npz(ipath_dis, dis)

    ipath_wavg = os.path.join(args.model_path, 'w_avg.npz') 
    w_avg = np.load(ipath_wavg)['arr_0']

    if 0 <= args.gpu:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        w_avg = chainer.cuda.to_gpu(w_avg)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    img = lib.util.imread(args.input_path)[:,:,::-1]
    fname = os.path.splitext(os.path.basename(args.input_path))[0]

    subdir = os.path.join(args.output_path, fname)
    
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    latent, imgen = find_latent(gen, dis, img, w_avg, subdir, lambda1=args.lambda1)

    dpath = os.path.join(args.output_path, '{}.npz'.format(fname))
    data = latent
    np.savez(dpath, data)

    dpath = os.path.join(args.output_path, '{}.png'.format(fname))
    imgen = chainer.cuda.to_cpu(imgen[0, ::-1])
    imgen = np.asarray(np.clip(imgen, 0.0, 255.0), dtype=np.uint8)
    imgen = imgen.transpose((1, 2, 0))
    lib.util.imwrite(dpath, imgen)

    pass

if __name__ == '__main__':
    main()