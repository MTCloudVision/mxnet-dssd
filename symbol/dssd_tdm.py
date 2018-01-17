import mxnet as mx

def prediction_module(conv, num_filter, lr_mult=1):
    conv1_1 = mx.symbol.Convolution(
        data=conv, kernel=(1, 1), pad=(0, 0), num_filter=int(num_filter*0.25), attr={'lr_mult': '%f' % lr_mult}, name="{}conv1_1pre".format(conv.name))
    conv1_2 = mx.symbol.Convolution(
        data=conv1_1, kernel=(1, 1), pad=(0, 0), num_filter=int(num_filter*0.25), attr={'lr_mult': '%f' % lr_mult}, name="{}conv1_2pre".format(conv.name))
    conv1_3 = mx.symbol.Convolution(
        data=conv1_2, kernel=(1, 1), pad=(0, 0), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}conv1_3pre".format(conv.name))
    conv2 = mx.symbol.Convolution(
        data=conv1_2, kernel=(1, 1), pad=(0, 0), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}conv2pre".format(conv.name))
    element_sum1 = mx.symbol.broadcast_add(conv2, conv1_3)
    return element_sum1

def deconvolution_module(conv1, conv2, num_filter, upstrid = 2, level = 1, lr_mult=1):
    deconv2 = mx.symbol.Deconvolution(
        data=conv2, kernel=(2, 2), stride=(upstrid, upstrid),num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}de_{}".format(conv2.name, str(level)))
    conv2_1 = mx.symbol.Convolution(
        data=deconv2, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}de_conv_{}".format(conv2.name, str(level)))
    BN2 = mx.symbol.BatchNorm(data=conv2_1, name="{}de_conv_bn_{}".format(conv2.name, str(level)))
    conv1_1 = mx.symbol.Convolution(
        data=conv1, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}conv_{}".format(conv1.name, str(level)))
    BN1_1 = mx.symbol.BatchNorm(data=conv1_1, name="{}conv_bn_{}".format(conv1.name, str(level)))
    relu1_1 = mx.symbol.Activation(data=BN1_1, act_type="relu", name="{}conv_bn_relu_{}".format(conv1.name, str(level)))
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}conv_bn_relu_conv_{}".format(conv1.name, str(level)))
    BN1_2 = mx.symbol.BatchNorm(data=conv1_2, name="{}conv_bn_relu_conv_bn{}".format(conv1.name, str(level)))
    BN2_clip = mx.symbol.Crop(*[BN2, BN1_2])
    element_product1plus2up = mx.symbol.broadcast_mul(BN1_2, BN2_clip)
    relu = mx.symbol.Activation(data=element_product1plus2up, act_type="relu", name="conv{}product{}de".format(str(level),str(level+1)))
    return relu

def topdown_upsample(conv1, conv2, filter, upstrid = 2, level = 1,lr_mult=1):
    conv2_thin = mx.symbol.Convolution(
        data=conv2, kernel=(3, 3), pad=(1, 1), num_filter=int(filter*0.5), attr={'lr_mult': '%f' % lr_mult}, name="{}{}_thin".format(conv2.name, str(level)))
    relu2_thin = mx.symbol.Activation(data=conv2_thin, act_type="relu", name="{}{}_thin_relu".format(conv2.name, str(level)))
    if upstrid > 1:
      relu2_up = mx.symbol.UpSampling(relu2_thin, scale=upstrid, sample_type='nearest', name="{}{}_thin_relu_up".format(conv2.name, str(level)))
    else:
      relu2_up = relu2_thin
    conv1_thin = mx.symbol.Convolution(
        data=conv1, kernel=(3, 3), pad=(1, 1), num_filter=int(filter*0.5), attr={'lr_mult': '%f' % lr_mult}, name="{}{}_thin".format(conv1.name, str(level)))
    relu1_thin = mx.symbol.Activation(data=conv1_thin, act_type="relu", name="{}{}_thin_relu".format(conv1.name, str(level)))
    relu2_up_clip = mx.symbol.Crop(*[relu2_up, relu1_thin])
    concat1plus2up = mx.symbol.concat(*[relu1_thin, relu2_up_clip], dim=1, name="concat{}plus{}up".format(str(level), str(level+1)))
    return concat1plus2up

def construct_dssd_deconv_layer(from_layers, num_filters, topdown_layers, use_perdict_module,):
    dssd_from_layers = []
    anti_layers = from_layers[::-1]
    for k, from_layer in enumerate(anti_layers):
        if topdown_layers[::-1][k]<0:
            dssd_from_layers.append(from_layer)
            continue
        conv = anti_layers[k-1]
        concat_conv = deconvolution_module(from_layer,conv,num_filters[::-1][k-1],2,k)
        anti_layers[k] = concat_conv
        if use_perdict_module[::-1][k]>1:
            dssd_from_layers.append(prediction_module(concat_conv,num_filters[::-1][k-1]))
        else :
            dssd_from_layers.append(concat_conv)
    return dssd_from_layers[::-1]

def construct_topdown_upsample_layer(from_layers, num_filters):
    tdm_from_layers = []
    anti_layers = from_layers[::-1]
    for k, from_layer in enumerate(anti_layers):
        if k<3:
            tdm_from_layers.append(from_layer)
            continue
        conv = anti_layers[k-1]
        concat_conv = topdown_upsample(from_layer,conv,num_filters[::-1][k],2,k)
        anti_layers[k] = concat_conv
        tdm_from_layers.append(concat_conv)
    return tdm_from_layers[::-1]

def residual_unit(data, num_filter, stride, name, bn_mom=0.9, workspace=256):
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(stride,stride), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(stride,stride), no_bias=True,
                                      workspace=workspace, name=name + '_sc')
        shortcut_clip = mx.symbol.Crop(*[shortcut, conv3])
        return conv3 + shortcut_clip