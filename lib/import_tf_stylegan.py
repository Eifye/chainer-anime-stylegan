from externals.chainer_stylegan.src.stylegan import net

def import_mapping_network(gen_tf, gen_ch):
    
    var_tf = {kk:vv.eval() for kk, vv in gen_tf.vars.items()}
    
    nlayers = 8
    
    for ii in range(nlayers):
        name_weight = 'G_mapping/Dense{}/weight'.format(ii)
        name_bias = 'G_mapping/Dense{}/bias'.format(ii)
        
        gen_ch.mapping.l[2*ii].c.W.array[:] = var_tf[name_weight].T
        gen_ch.mapping.l[2*ii].c.b.array[:] = var_tf[name_bias] * gen_ch.mapping.l[2*ii].lrmul
        
    return 

def import_synthesis_network(gen_tf, gen_ch):
    var_tf = {kk:vv.eval() for kk, vv in gen_tf.vars.items()}
    res = 4
    
    # 1st layer
    gen_ch.gen.blocks[0].W.array[:] = var_tf['G_synthesis/4x4/Const/const']
    gen_ch.gen.blocks[0].n0.b.W.array[:] = var_tf['G_synthesis/4x4/Const/Noise/weight']
    gen_ch.gen.blocks[0].b0.b.array[:] = var_tf['G_synthesis/4x4/Const/bias']
    gen_ch.gen.blocks[0].s0.s.c.W.array[:] = var_tf['G_synthesis/4x4/Const/StyleMod/weight'][:, :512].T
    gen_ch.gen.blocks[0].s0.s.c.b.array[:] = var_tf['G_synthesis/4x4/Const/StyleMod/bias'][:512]
    gen_ch.gen.blocks[0].s0.b.c.W.array[:] = var_tf['G_synthesis/4x4/Const/StyleMod/weight'][:, 512:].T
    gen_ch.gen.blocks[0].s0.b.c.b.array[:] = var_tf['G_synthesis/4x4/Const/StyleMod/bias'][512:]
    gen_ch.gen.blocks[0].c1.c.W.array[:] = var_tf['G_synthesis/4x4/Conv/weight'].transpose((3,2,0,1))
    gen_ch.gen.blocks[0].n1.b.W.array[:] = var_tf['G_synthesis/4x4/Conv/Noise/weight']
    gen_ch.gen.blocks[0].b1.b.array[:] = var_tf['G_synthesis/4x4/Conv/bias']    
    gen_ch.gen.blocks[0].s1.s.c.W.array[:] = var_tf['G_synthesis/4x4/Conv/StyleMod/weight'][:, :512].T
    gen_ch.gen.blocks[0].s1.s.c.b.array[:] = var_tf['G_synthesis/4x4/Conv/StyleMod/bias'][:512]
    gen_ch.gen.blocks[0].s1.b.c.W.array[:] = var_tf['G_synthesis/4x4/Conv/StyleMod/weight'][:, 512:].T
    gen_ch.gen.blocks[0].s1.b.c.b.array[:] = var_tf['G_synthesis/4x4/Conv/StyleMod/bias'][512:]
    
    for ii in range(0, 7): # x2(16x16)..x8(512..512)
        res *= 2
        parent = "G_synthesis/{}x{}/Conv0_up/".format(res, res)
        gen_ch.gen.blocks[ii+1].c0.c.W.array[:] = var_tf[parent+'weight'].transpose((3,2,0,1))
        gen_ch.gen.blocks[ii+1].n0.b.W.array[:] = var_tf[parent+'Noise/weight']
        gen_ch.gen.blocks[ii+1].b0.b.array[:] = var_tf[parent+'bias']
        ndim = var_tf[parent+'bias'].shape[0]
        gen_ch.gen.blocks[ii+1].s0.s.c.W.array[:] = var_tf[parent+'StyleMod/weight'][:, :ndim].T
        gen_ch.gen.blocks[ii+1].s0.s.c.b.array[:] = var_tf[parent+'StyleMod/bias'][:ndim]
        gen_ch.gen.blocks[ii+1].s0.b.c.W.array[:] = var_tf[parent+'StyleMod/weight'][:, ndim:].T
        gen_ch.gen.blocks[ii+1].s0.b.c.b.array[:] = var_tf[parent+'StyleMod/bias'][ndim:]
        
        parent = "G_synthesis/{}x{}/Conv1/".format(res, res)
        
        gen_ch.gen.blocks[ii+1].c1.c.W.array[:] = var_tf[parent+'weight'].transpose((3,2,0,1))
        gen_ch.gen.blocks[ii+1].n1.b.W.array[:] = var_tf[parent+'Noise/weight']
        gen_ch.gen.blocks[ii+1].b1.b.array[:] = var_tf[parent+'bias']
        ndim = var_tf[parent+'bias'].shape[0]
        gen_ch.gen.blocks[ii+1].s1.s.c.W.array[:] = var_tf[parent+'StyleMod/weight'][:, :ndim].T
        gen_ch.gen.blocks[ii+1].s1.s.c.b.array[:] = var_tf[parent+'StyleMod/bias'][:ndim]
        gen_ch.gen.blocks[ii+1].s1.b.c.W.array[:] = var_tf[parent+'StyleMod/weight'][:, ndim:].T
        gen_ch.gen.blocks[ii+1].s1.b.c.b.array[:] = var_tf[parent+'StyleMod/bias'][ndim:]
        
    gen_ch.gen.outs[7].c.W.array[:] = var_tf['G_synthesis/ToRGB_lod0/weight'].transpose((3,2,0,1))
    gen_ch.gen.outs[7].c.b.array[:] = var_tf['G_synthesis/ToRGB_lod0/bias']

def import_generator(gen_tf):
    dst = net.Generator(512)
    import_mapping_network(gen_tf, dst)
    import_synthesis_network(gen_tf, dst)
    w_avg = gen_tf.vars['dlatent_avg'].eval()
    return dst, w_avg

def import_discriminator(dis_tf, dis_ch):
    var_tf = {kk:vv.eval() for kk, vv in dis_tf.vars.items()}
    
    # from rgb
    for ii in range(8):
        parent = 'FromRGB_lod{}/'.format(ii)
        dis_ch.ins[7-ii].c.W.array[:] = var_tf[parent + 'weight'].transpose((3,2,0,1))
        dis_ch.ins[7-ii].c.b.array[:] = var_tf[parent + 'bias']
        
    # last dense layer
    dis_ch.blocks[0].c0.c.W.array[:] = var_tf['4x4/Conv/weight'].transpose((3,2,0,1))
    dis_ch.blocks[0].c0.c.b.array[:] = var_tf['4x4/Conv/bias']
    dis_ch.blocks[0].l1.c.W.array[:] = var_tf['4x4/Dense0/weight'].T
    dis_ch.blocks[0].l1.c.b.array[:] = var_tf['4x4/Dense0/bias']
    dis_ch.blocks[0].l2.c.W.array[:] = var_tf['4x4/Dense1/weight'].T
    dis_ch.blocks[0].l2.c.b.array[:] = var_tf['4x4/Dense1/bias']
    
    # intermediate layer
    for ii in range(1, 8):
        res = 4*(2**(ii))
        parent = '{}x{}/'.format(res, res)
        dis_ch.blocks[ii].c0.c.W.array[:] = var_tf[parent + 'Conv0/weight'].transpose((3,2,0,1))
        dis_ch.blocks[ii].c0.c.b.array[:] = var_tf[parent + 'Conv0/bias']
        dis_ch.blocks[ii].c1.c.W.array[:] = var_tf[parent + 'Conv1_down/weight'].transpose((3,2,0,1))
        dis_ch.blocks[ii].c1.c.b.array[:] = var_tf[parent + 'Conv1_down/bias']
        
