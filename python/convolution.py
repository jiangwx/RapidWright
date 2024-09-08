import torch
import torch.nn as nn
import numpy as np

k0 = c = 16

def convolution(ifm, weight, bias, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    in_height = layer['in_height']
    in_width = layer['in_width']
    in_channel = layer['in_channel']
    out_channel = layer['out_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    ofm = np.zeros((out_channel, out_height, out_width))
    for oh in range(out_height):
        for ow in range(out_width):
            for oc in range(out_channel):
                y_data = 0
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        ih = oh * stride_h - pad_h + kh * dilation_h
                        iw = ow * stride_w - pad_w + kw * dilation_w
                        if (0 <= ih < in_height) and (0 <= iw < in_width):
                            for ic in range(in_channel):
                                ifm_index = ic * in_height * in_width + ih * in_width + iw
                                wgt_index = oc * kernel_h * kernel_w * in_channel + ic * kernel_h * kernel_w + kh * kernel_w + kw
                                x_data = ifm[ifm_index]
                                w_data = weight[wgt_index]
                                y_data += x_data * w_data
                ofm[oc][oh][ow] = y_data + bias[oc]
    return ofm

def im2col_ifm_major(ifm, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    in_height = layer['in_height']
    in_width = layer['in_width']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    M = out_height * out_width
    K = in_channel * kernel_h * kernel_w

    ofm = np.zeros((M, K))
    for oh in range(out_height):
        for ow in range(out_width):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    ih = oh * stride_h - pad_h + kh * dilation_h
                    iw = ow * stride_w - pad_w + kw * dilation_w
                    if (0 <= ih < in_height) and (0 <= iw < in_width):
                        for ic in range(in_channel):
                            ifm_index = ic*in_height*in_width + ih*in_width + iw
                            m_index = oh*out_width + ow
                            k_index = ic*kernel_h*kernel_w + kh*kernel_w + kw
                            ofm[m_index][k_index] = ifm[ifm_index]
    return ofm

def im2col_CHW2MK(ifm, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    in_height = layer['in_height']
    in_width = layer['in_width']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']

    M = out_height * out_width
    K = in_channel * kernel_h * kernel_w
    ofm = np.zeros((M, K))

    for m in range(M):
        for k in range(K):
            oh = m // out_width
            ow = m % out_width
            ic = k // (kernel_h * kernel_w)
            kh = (k // kernel_w) % kernel_h
            kw = k % kernel_w
            ih = oh * stride_h - pad_h + kh * dilation_h
            iw = ow * stride_w - pad_w + kw * dilation_w
            if (0 <= ih < in_height) and (0 <= iw < in_width):
                ifm_index = ic * in_height * in_width + ih * in_width + iw
                ofm[m][k] = ifm[ifm_index]
    return ofm

def ifm_CHW2CHWc(ifm, layer):
    in_channel = layer['in_channel']
    in_height = layer['in_height']
    in_width = layer['in_width']
    C = (in_channel + c - 1) // c
    ifm_CHW = ifm.reshape((in_channel, in_height, in_width))
    ifm_HWC = ifm_CHW.transpose((1, 2, 0))
    ifm_HWC_ = np.zeros((in_height, in_width, C*c))
    for ic in range(in_channel):
        ifm_HWC_[:, :, ic] = ifm_HWC[:, :, ic]
    ifm_HWCc = ifm_HWC_.reshape((in_height, in_width, C, c))
    ifm_CHWc = ifm_HWCc.transpose((2, 0, 1, 3))
    return ifm_CHWc

def wt_OCICKhKw2OCCKhKwc(weight, layer):
    out_channel = layer['out_channel']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    C = (in_channel + c - 1) // c
    weight_OCICKhKw = weight.reshape((out_channel, in_channel, kernel_h, kernel_w))
    weight_OCKhKwIC = weight_OCICKhKw.transpose((0, 2, 3, 1))
    weight_OCKhKwIC_ = np.zeros((out_channel, kernel_h, kernel_w, C*c))
    for ic in range(in_channel):
        weight_OCKhKwIC_[:, :, :, ic] = weight_OCKhKwIC[:, :, :, ic]
    weight_OCKhKwCc = weight_OCKhKwIC_.reshape((out_channel, kernel_h, kernel_w, C, c))
    weight_OCCKhKwc = weight_OCKhKwCc.transpose((0, 3, 1, 2, 4))
    return weight_OCCKhKwc

def wt_expand_ic(weight, layer):
    out_channel = layer['out_channel']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    C = (in_channel + c - 1) // c
    weight_OCICKhKw = weight.reshape((out_channel, in_channel, kernel_h, kernel_w))
    weight_OCCcKhKw = np.zeros((out_channel, C*c, kernel_h, kernel_w))
    for ic in range(in_channel):
        weight_OCCcKhKw[:, ic, :, :] = weight_OCICKhKw[:, ic, :, :]
    return weight_OCCcKhKw.reshape((out_channel, C*c*kernel_h*kernel_w))

def im2col_CHWc2MK(ifm, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    in_height = layer['in_height']
    in_width = layer['in_width']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    C = (in_channel + c - 1) // c
    M = out_height * out_width
    K = C * c * kernel_h * kernel_w
    ofm = np.zeros((M, K))

    for m in range(M):
        for k in range(K):
            oh = m // out_width
            ow = m % out_width
            ic = k // (kernel_h * kernel_w)
            kh = (k // kernel_w) % kernel_h
            kw = k % kernel_w
            ih = oh * stride_h - pad_h + kh * dilation_h
            iw = ow * stride_w - pad_w + kw * dilation_w
            if (0 <= ih < in_height) and (0 <= iw < in_width):
                c_idx = ic % c
                C_idx = ic // c
                ofm[m][k] = ifm[C_idx][ih][iw][c_idx]
    return ofm

def test_convolution():
    # 定义输入参数
    in_height = np.random.randint(1, 100)
    in_width = np.random.randint(1, 100)
    in_channel = np.random.randint(1, 100)
    out_channel = np.random.randint(1, 100)
    kernel_h = np.random.randint(1,5)
    kernel_w = np.random.randint(1,5)
    stride_h = np.random.randint(1,5)
    stride_w = np.random.randint(1,5)
    pad_h = np.random.randint(0,kernel_h)
    pad_w = np.random.randint(0,kernel_w)
    dilation_h = np.random.randint(1,in_height+1)
    dilation_w = np.random.randint(1,in_width+1)
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if(out_height <= 0 or out_width <= 0):
        return False
    # 定义卷积层参数
    layer = {
        'in_height': in_height,
        'in_width': in_width,
        'out_height': out_height,
        'out_width': out_width,
        'in_channel': in_channel,
        'out_channel': out_channel,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'dilation_h': dilation_h,
        'dilation_w': dilation_w
    }
    print(layer)

    # 创建输入数据
    ifm = np.random.randint(-128, 127, size=(in_channel, in_height, in_width))
    weight = np.random.randint(-128, 127, size=(out_channel, in_channel, kernel_h, kernel_w))
    bias = np.random.randint(-128, 127, size=out_channel)

    # 先转成CHWc, 再使用im2col和gemm
    ifm_CHWc = ifm_CHW2CHWc(ifm.flatten(), layer)
    weight_OCCcKhKw = wt_expand_ic(weight.flatten(), layer)
    ifm_im2col = im2col_CHWc2MK(ifm_CHWc, layer)
    print(ifm_CHWc.shape, weight_OCCcKhKw.shape, ifm_im2col.shape, weight_OCCcKhKw.shape)
    ofm_im2col = np.matmul(ifm_im2col, weight_OCCcKhKw.transpose())
    ofm_im2col_t = ofm_im2col.transpose()
    for i in range(out_channel):
        ofm_im2col_t[i] += bias[i]

    # 使用torch.nn.Conv2d
    ifm_torch = torch.tensor(ifm, dtype=torch.float32)
    weight_torch = torch.tensor(weight, dtype=torch.float32)
    bias_torch = torch.tensor(bias, dtype=torch.float32)

    conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(kernel_h, kernel_w),
                        stride=(stride_h, stride_w), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w))
    # 将权重和偏置转换为合适的格式
    conv_layer.weight.data = torch.nn.Parameter(weight_torch)
    conv_layer.bias.data = torch.nn.Parameter(bias_torch)

    # 前向传播
    ofm_torch = conv_layer(ifm_torch)

    # 将PyTorch张量转换为NumPy数组
    ofm_torch_np = ofm_torch.detach().numpy().flatten()

    diff = ofm_torch_np - ofm_im2col_t.flatten()
    if(np.abs(diff).sum() == 0.0):
        print("img2col + gemm pass")
    else:
        print("img2col + gemm fail")
    return True

test_num = 50
while(test_num >= 0):
    if(test_convolution()):
        test_num -= 1