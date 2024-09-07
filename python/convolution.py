import torch
import torch.nn as nn
import numpy as np

def convolution(ifm, weight, bias, layer):
    out_height = (layer['in_height'] + 2 * layer['pad_h'] - layer['dilation_h'] * (layer['kernel_h'] - 1) - 1) // layer['stride_h'] + 1
    out_width = (layer['in_width'] + 2 * layer['pad_w'] - layer['dilation_w'] * (layer['kernel_w'] - 1) - 1) // layer['stride_w'] + 1
    layer['out_height'] = out_height
    layer['out_width'] = out_width
    print(layer)
    ofm = np.zeros((layer['out_channel'], out_height, out_width))

    for oh in range(out_height):
        for ow in range(out_width):
            for oc in range(layer['out_channel']):
                y_data = 0
                for ic in range(layer['in_channel']):
                    for kh in range(layer['kernel_h']):
                        for kw in range(layer['kernel_w']):
                            in_height = oh * layer['stride_h'] - layer['pad_h'] + kh * layer['dilation_h']
                            in_width = ow * layer['stride_w'] - layer['pad_w'] + kw * layer['dilation_w']
                            if (0 <= in_height < layer['in_height']) and (0 <= in_width < layer['in_width']):
                                ifm_index = ic * layer['in_height'] * layer['in_width'] + in_height * layer['in_width'] + in_width
                                wgt_index = oc * layer['kernel_h'] * layer['kernel_w'] * layer['in_channel'] + ic * layer['kernel_h'] * layer['kernel_w'] + kh * layer['kernel_w'] + kw
                                x_data = ifm[ifm_index]
                                w_data = weight[wgt_index]
                                y_data += x_data * w_data
                ofm[oc][oh][ow] = y_data + bias[oc]
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
    dilation_h = np.random.randint(1,in_height)
    dilation_w = np.random.randint(1,in_width)
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if(out_height <= 0 or out_width <= 0):
        return False
    # 定义卷积层参数
    layer = {
        'in_height': in_height,
        'in_width': in_width,
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

    # 创建输入数据
    ifm = np.random.randint(-128, 127, size=(in_channel, in_height, in_width))
    weight = np.random.randint(-128, 127, size=(out_channel, in_channel, kernel_h, kernel_w))
    bias = np.random.randint(-128, 127, size=out_channel)

    # 调用卷积函数
    ofm_custom = convolution(ifm.flatten(), weight.flatten(), bias.flatten(), layer)

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
    ofm_torch_np = ofm_torch.detach().numpy()

    diff = ofm_torch_np - ofm_custom
    if(np.abs(diff).sum() == 0.0):
        print("pass")
    else:
        print("fail")
    return True

test_num = 10
while(test_num >= 0):
    if(test_convolution()):
        test_num -= 1