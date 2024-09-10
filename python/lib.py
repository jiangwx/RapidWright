import numpy as np
import torch
import torch.nn as nn

M0 = 8
N0 = 4
K0 = c = 16

M_L1 = 24
N_L1 = 25
K_L1 = 17

M_L0 = 13
N_L0 = 14
K_L0 = 7

def ceil_div(x, y):
    return (x + y - 1) // y

def MK2K1MK0(matrix_mk):
    matrix_M = matrix_mk.shape[0]
    matrix_K = matrix_mk.shape[1]
    K1 = (matrix_K + K0 - 1) // K0
    matrix_mk1k0_ = np.zeros((matrix_M, K1*K0))
    matrix_mk1k0_[:, :matrix_K] = matrix_mk
    matrix_mk1k0 = matrix_mk1k0_.reshape((matrix_M, K1, K0))
    matrix_k1mk0 = matrix_mk1k0.transpose((1, 0, 2))
    return matrix_k1mk0

def KN2N1KN0(matrix_kn):
    matrix_K = matrix_kn.shape[0]
    matrix_N = matrix_kn.shape[1]
    N1 = (matrix_N + N0 - 1) // N0
    matrix_kn1n0_ = np.zeros((matrix_K, N1*N0))
    matrix_kn1n0_[:, :matrix_N] = matrix_kn
    matrix_kn1n0 = matrix_kn1n0_.reshape((matrix_K, N1, N0))
    matrix_n1kn0 = matrix_kn1n0.transpose((1, 0, 2))
    return matrix_n1kn0

def K1MK02M1K1M0K0(matrix_k1mk0):
    K1 = matrix_k1mk0.shape[0]
    M = matrix_k1mk0.shape[1]
    K0 = matrix_k1mk0.shape[2]
    M1 = (M + M0 - 1) // M0
    matrix_m1k1m0k0 = np.zeros((M1, K1, M0, K0))
    for m1 in range(M1):
        for k1 in range(K1):
            for m0 in range(M0):
                m = m1 * M0 + m0
                if(m < M):
                    matrix_m1k1m0k0[m1][k1][m0] = matrix_k1mk0[k1][m]
    return matrix_m1k1m0k0

def N1KN02K1N1K0N0(matrix_n1kn0):
    N1 = matrix_n1kn0.shape[0]
    K = matrix_n1kn0.shape[1]
    N0 = matrix_n1kn0.shape[2]
    K1 = (K + K0 - 1) // K0
    matrix_k1n1k0n0 = np.zeros((K1, N1, K0, N0))
    for k1 in range(K1):
        for n1 in range(N1):
            for k0 in range(K0):
                k = k1 * K0 + k0
                if(k < K):
                    matrix_k1n1k0n0[k1][n1][k0] = matrix_n1kn0[n1][k]
    return matrix_k1n1k0n0

def M1N1M0N02MN(matrix_m1n1m0n0, layer):
    M1 = matrix_m1n1m0n0.shape[0]
    N1 = matrix_m1n1m0n0.shape[1]
    M0 = matrix_m1n1m0n0.shape[2]
    N0 = matrix_m1n1m0n0.shape[3]
    M = layer['M']
    N = layer['N']
    matrix_mn = np.zeros((M, N))
    for m1 in range(M1):
        for n1 in range(N1):
            for m0 in range(M0):
                for n0 in range(N0):
                    m = m1 * M0 + m0
                    n = n1 * N0 + n0
                    if(m < M and n < N):
                        matrix_mn[m][n] = matrix_m1n1m0n0[m1][n1][m0][n0]
    return matrix_mn

def CONV_M1N1M0N02MN(ifm):
    M1 = ifm.shape[0]
    N1 = ifm.shape[1]
    M0 = ifm.shape[2]
    N0 = ifm.shape[3]
    M = M1 * M0
    N = N1 * N0
    ofm = np.zeros((M, N))
    for m1 in range(M1):
        for n1 in range(N1):
            for m0 in range(M0):
                for n0 in range(N0):
                    m = m1 * M0 + m0
                    n = n1 * N0 + n0
                    ofm[m][n] = ifm[m1][n1][m0][n0]
    return ofm

def MK2M1K1M0K0(matrix_mk):
    matrix_M = matrix_mk.shape[0]
    matrix_K = matrix_mk.shape[1]
    M1 = (matrix_M + M0 - 1) // M0
    K1 = (matrix_K + K0 - 1) // K0
    matrix_m1k1m0k0 = np.zeros((M1, K1, M0, K0))
    for m1 in range(M1):
        for k1 in range(K1):
            for m0 in range(M0):
                for k0 in range(K0):
                    m = m1 * M0 + m0
                    k = k1 * K0 + k0
                    if(m < matrix_M and k < matrix_K):
                        matrix_m1k1m0k0[m1][k1][m0][k0] = matrix_mk[m][k]
    return matrix_m1k1m0k0

def KN2K1N1K0N0(matrix_kn):
    matrix_K = matrix_kn.shape[0]
    matrix_N = matrix_kn.shape[1]
    K1 = (matrix_K + K0 - 1) // K0
    N1 = (matrix_N + N0 - 1) // N0
    matrix_k1n1k0n0 = np.zeros((K1, N1, K0, N0))
    for k1 in range(K1):
        for n1 in range(N1):
            for k0 in range(K0):
                for n0 in range(N0):
                    k = k1 * K0 + k0
                    n = n1 * N0 + n0
                    if(k < matrix_K and n < matrix_N):
                        matrix_k1n1k0n0[k1][n1][k0][n0] = matrix_kn[k][n]
    return matrix_k1n1k0n0

def N2N1N0(bias):
    N = bias.shape[0]
    N1 = (N + N0 - 1) // N0
    bias_N1N0 = np.zeros((N1, N0))
    for n1 in range(N1):
        for n0 in range(N0):
            n = n1 * N0 + n0
            if(n < N):
                bias_N1N0[n1][n0] = bias[n]
    return bias_N1N0

def matmul_m1k1m0k0_k1n1k0n0(matrix_m1k1m0k0, matrix_k1n1k0n0, bias_n1n0):
    M1 = matrix_m1k1m0k0.shape[0]
    K1 = matrix_m1k1m0k0.shape[1]
    M0 = matrix_m1k1m0k0.shape[2]
    N1 = matrix_k1n1k0n0.shape[1]
    N0 = matrix_k1n1k0n0.shape[3]
    matrix_m1n1m0n0 = np.zeros((M1, N1, M0, N0))
    for m1 in range(M1):
        for n1 in range(N1):
            temp = np.zeros((M0, N0))
            for k1 in range(K1):
                temp += np.matmul(matrix_m1k1m0k0[m1][k1], matrix_k1n1k0n0[k1][n1])
            for n0 in range(N0):
                temp[:, n0] += bias_n1n0[n1][n0]
            matrix_m1n1m0n0[m1][n1] = temp
    return matrix_m1n1m0n0

def matmul_mk_kn(matrix_mk, matrix_kn, bias):
    M = matrix_mk.shape[0]
    K = matrix_mk.shape[1]
    N = matrix_kn.shape[1]
    matrix_mn = np.matmul(matrix_mk, matrix_kn)
    for m in range(M):
        for n in range(N):
            matrix_mn[m][n] += bias[n]
    return matrix_mn

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

def CHW2CHWc(ifm, layer):
    in_channel = layer['in_channel']
    in_height = layer['in_height']
    in_width = layer['in_width']
    C = (in_channel + c - 1) // c
    ifm_HWC = ifm.transpose((1, 2, 0))
    ifm_HWC_ = np.zeros((in_height, in_width, C*c))
    for ic in range(in_channel):
        ifm_HWC_[:, :, ic] = ifm_HWC[:, :, ic]
    ifm_HWCc = ifm_HWC_.reshape((in_height, in_width, C, c))
    ifm_CHWc = ifm_HWCc.transpose((2, 0, 1, 3))
    return ifm_CHWc

def CHWc2M1K1M0K0(ifm, layer):
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
    C = ifm.shape[0]
    H = ifm.shape[1]
    W = ifm.shape[2]

    M = out_height * out_width
    K = C * c * kernel_h * kernel_w
    M1 = (M + M0 - 1) // M0
    K1 = (K + K0 - 1) // K0
    ofm = np.zeros((M1, K1, M0, K0))
    for m1 in range(M1):
        for k1 in range(K1):
            for m0 in range(M0):
                m = m1 * M0 + m0
                oh = m // out_width
                ow = m % out_width
                # k = C_idx*kernel_h*kernel_w*c + kh*kernel_w*c + kw*c + ic
                C_idx = k1 // (kernel_h * kernel_w)
                kh = (k1 // kernel_w) % kernel_h
                kw = k1 % kernel_w
                ih = oh * stride_h - pad_h + kh * dilation_h
                iw = ow * stride_w - pad_w + kw * dilation_w
                if (0 <= ih < in_height) and (0 <= iw < in_width):
                    ofm[m1][k1][m0] = ifm[C_idx][ih][iw]
    return ofm

def OCICKhKw2OCCKhKwc(weight, layer):
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

def OCCKhKwc2K1N1K0N0(weight, layer):
    out_channel = layer['out_channel']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    C = (in_channel + c - 1) // c
    N = out_channel
    K = C * kernel_h * kernel_w * c
    N1 = (N + N0 - 1) // N0
    K1 = K // K0
    assert( K % K0 == 0)

    weight_KN = weight.reshape((N, K)).transpose()
    weight_K1N1K0N0 = np.zeros((K1, N1, K0, N0))
    for k1 in range(K1):
        for n1 in range(N1):
            for k0 in range(K0):
                for n0 in range(N0):
                    k = k1 * K0 + k0
                    n = n1 * N0 + n0
                    if(k < K and n < N):
                        weight_K1N1K0N0[k1][n1][k0][n0] = weight_KN[k][n]
    return weight_K1N1K0N0

def matmul_m1k1m0k0_k1n1k0n0(ifm, weight, bias):
    assert(ifm.shape[1] == weight.shape[0])
    assert(ifm.shape[3] == weight.shape[2])
    K1 = ifm.shape[1]
    K0 = ifm.shape[3]
    M1 = ifm.shape[0]
    M0 = ifm.shape[2]
    N1 = weight.shape[1]
    N0 = weight.shape[3]

    ofm = np.zeros((M1, N1, M0, N0))
    for m1 in range(M1):
        for n1 in range(N1):
            temp = np.zeros((M0, N0))
            for k1 in range(K1):
                temp += np.matmul(ifm[m1][k1], weight[k1][n1])
            for n0 in range(N0):
                temp[:, n0] += bias[n1][n0]
            ofm[m1][n1] = temp
    return ofm

def MN2CHW(fm, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    out_channel = layer['out_channel']
    ofm_CHW = np.zeros((out_channel, out_height, out_width))
    for oc in range(out_channel):
        for oh in range(out_height):
            for ow in range(out_width):
                ofm_CHW[oc][oh][ow] = fm[oh*out_width+ow][oc]
    return ofm_CHW
