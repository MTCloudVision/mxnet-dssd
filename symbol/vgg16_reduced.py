import mxnet as mx
'''
def legacy_conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    assert not use_batchnorm, "batchnorm not yet supported"
    bias = mx.symbol.Variable(name="conv{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="conv{}".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}{}".format(act_type, name))
    if use_batchnorm:
        relu = mx.symbol.BatchNorm(data=relu, name="bn{}".format(name))
    return conv, relu

def deconvolution_module(conv1, conv2, num_filter, upstrid = 2, level = 1):
    deconv2 = mx.symbol.Deconvolution(
        data=conv2, kernel=(2, 2), stride=(upstrid, upstrid),num_filter=num_filter, workspace=2048, name="{}de_{}".format(conv2.name, str(level)))
    conv2_1 = mx.symbol.Convolution(
        data=deconv2, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, workspace=2048, name="{}de_conv_{}".format(conv2.name, str(level)))
    BN2 = mx.symbol.BatchNorm(data=conv2_1, name="{}de_conv_bn_{}".format(conv2.name, str(level)))
    conv1_1 = mx.symbol.Convolution(
        data=conv1, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, workspace=2048, name="{}conv_{}".format(conv1.name, str(level)))
    BN1_1 = mx.symbol.BatchNorm(data=conv1_1, name="{}conv_bn_{}".format(conv1.name, str(level)))
    relu1_1 = mx.symbol.Activation(data=BN1_1, act_type="relu", name="{}conv_bn_relu_{}".format(conv1.name, str(level)))
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, workspace=2048, name="{}conv_bn_relu_conv_{}".format(conv1.name, str(level)))
    BN1_2 = mx.symbol.BatchNorm(data=conv1_2, name="{}conv_bn_relu_conv_bn{}".format(conv1.name, str(level)))
    element_product1plus2up = mx.symbol.broadcast_mul(BN1_2, BN2)
    relu = mx.symbol.Activation(data=element_product1plus2up, act_type="relu", name="conv{}product{}de".format(str(level),str(level+1)))
    return relu

def prediction_module(conv):
    conv1_1 = mx.symbol.Convolution(
        data=conv, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="{}conv1_1".format(conv.name))
    conv1_2 = mx.symbol.Convolution(
        data=conv1_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="{}conv1_2".format(conv.name))
    conv1_3 = mx.symbol.Convolution(
        data=conv1_2, kernel=(3, 3), pad=(1, 1), num_filter=1024, workspace=2048, name="{}conv1_3".format(conv.name))
    element_sum1 = mx.symbol.broadcast_add(conv, mx.symbol.reshape(data=conv1_3, shape=(-1, 1024, -1, -1)))
    conv2_1 = mx.symbol.Convolution(
        data=element_sum1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="{}conv2_1".format(conv.name))
    conv2_2 = mx.symbol.Convolution(
        data=conv2_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="{}conv2_2".format(conv.name))
    conv2_3 = mx.symbol.Convolution(
        data=conv2_2, kernel=(3, 3), pad=(1, 1), num_filter=1024, workspace=2048, name="{}conv2_3".format(conv.name))
    element_sum2 = mx.symbol.broadcast_add(conv1_3, mx.symbol.reshape(data=conv2_3, shape=(-1, 1024, -1, -1)), name="{}preditct".format(conv.name))
    return element_sum2

def topdown_upsample(conv1, conv2, filter1, filter2, upstrid = 2, level = 1):
    conv2_thin = mx.symbol.Convolution(
        data=conv2, kernel=(3, 3), pad=(1, 1), num_filter=filter2, workspace=2048, name="{}{}_thin".format(conv2.name, str(level)))
    relu2_thin = mx.symbol.Activation(data=conv2_thin, act_type="relu", name="{}{}_thin_relu".format(conv2.name, str(level)))
    if upstrid > 1:
      relu2_up = mx.symbol.UpSampling(relu2_thin, scale=upstrid, sample_type='nearest', name="{}{}_thin_relu_up".format(conv2.name, str(level)))
    else:
      relu2_up = relu2_thin
    conv1_thin = mx.symbol.Convolution(
        data=conv1, kernel=(3, 3), pad=(1, 1), num_filter=filter1, workspace=2048, name="{}{}_thin".format(conv1.name, str(level)))
    relu1_thin = mx.symbol.Activation(data=conv1_thin, act_type="relu", name="{}{}_thin_relu".format(conv1.name, str(level)))
    concat1plus2up = mx.symbol.concat(*[relu2_up, relu1_thin], dim=1, name="concat{}plus{}up".format(str(level), str(level+1)))
    return concat1plus2up
'''
def get_symbol(num_classes=1000, **kwargs):
    """
    VGG 16 layers network
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1, 1),
        pad=(1,1), name="pool5")
    # group 6
    conv6 = mx.symbol.Convolution(
        data=pool5, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=1024, name="fc6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    '''
    conv8_1, relu8_1 = legacy_conv_act_layer(relu7, "8_1", 256, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv8_2, relu8_2 = legacy_conv_act_layer(relu8_1, "8_2", 512, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv9_1, relu9_1 = legacy_conv_act_layer(relu8_2, "9_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv9_2, relu9_2 = legacy_conv_act_layer(relu9_1, "9_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv10_1, relu10_1 = legacy_conv_act_layer(relu9_2, "10_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv10_2, relu10_2 = legacy_conv_act_layer(relu10_1, "10_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv11_1, relu11_1 = legacy_conv_act_layer(relu10_2, "11_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv11_2, relu11_2 = legacy_conv_act_layer(relu11_1, "11_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv12_1, relu12_1 = legacy_conv_act_layer(relu11_2, "12_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv12_2, relu12_2 = legacy_conv_act_layer(relu12_1, "12_2", 256, kernel=(4,4), pad=(1,1), \
        stride=(1,1), act_type="relu", use_batchnorm=False)



    # top donw framwork
    # relu7_output_shape = relu7.infer_shape(data=(tmp_video_per_batch, n_channel, height, width))[1][0]
    # print('relu7:', relu7_output_shape)
   # conv11product12de = deconvolution_module(relu11_2, relu12_2, 256, 2, 11)
   # conv10product11de = deconvolution_module(relu10_2, relu11_2, 256, 2, 10)
    conv9product10de = deconvolution_module(relu9_2, relu10_2, 256, 2, 9)
    conv8product9de = deconvolution_module(relu8_2, conv9product10de, 256, 2, 8)
    conv7product8de = deconvolution_module(relu7, conv8product9de, 512, 2, 7)
    conv4product7de = deconvolution_module(relu4_3, conv7product8de, 512, 2, 4)
    '''
#    gpool = mx.symbol.Pooling(data=conv4product7de, pool_type='avg', kernel=(7, 7),
    gpool = mx.symbol.Pooling(data=relu7, pool_type='avg', kernel=(7, 7),
                              global_pool=True, name='global_pool')
    conv8 = mx.symbol.Convolution(data=gpool, num_filter=num_classes, kernel=(1, 1),
                                  name='fc8')
    flat = mx.symbol.Flatten(data=conv8)
    softmax = mx.symbol.SoftmaxOutput(data=flat, name='softmax')
    return softmax
