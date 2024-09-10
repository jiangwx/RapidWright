from lib import *

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
    ifm_CHWc = CHW2CHWc(ifm, layer)
    weight_OCCKhKwc = OCICKhKw2OCCKhKwc(weight, layer)
    ifm_M1K1M0K0 = CHWc2M1K1M0K0(ifm_CHWc, layer)
    weight_K1N1K0N0 = OCCKhKwc2K1N1K0N0(weight_OCCKhKwc, layer)
    bias_N1N0 = N2N1N0(bias)
    ofm_M1N1M0N0 = matmul_m1k1m0k0_k1n1k0n0(ifm_M1K1M0K0, weight_K1N1K0N0, bias_N1N0)
    ofm_MN = CONV_M1N1M0N02MN(ofm_M1N1M0N0)
    ofm_CHW = MN2CHW(ofm_MN, layer)

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

    diff = ofm_torch_np - ofm_CHW.flatten()
    if(np.abs(diff).sum() == 0.0):
        print("img2col + gemm pass")
    else:
        print("img2col + gemm fail")
    return True

test_num = 10
while(test_num > 0):
    if(test_convolution()):
        test_num -= 1