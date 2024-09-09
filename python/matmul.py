import torch
import torch.nn as nn
import numpy as np

K0 = c = 16
M0 = 8
N0 = 4

def matrix_MK2M1K1M0K0(matrix_mk):
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

def matrix_KN2K1N1K0N0(matrix_nk):
    matrix_K = matrix_nk.shape[0]
    matrix_N = matrix_nk.shape[1]
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
                        matrix_k1n1k0n0[k1][n1][k0][n0] = matrix_nk[k][n]
    return matrix_k1n1k0n0


def bias_N2N02N1N0(bias, layer):
    N = layer['N']
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

def matrix_M1N1M0N02MN(matrix_m1n1m0n0, layer):
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

def test_matmul():
    # 定义输入参数
    matrix_M = np.random.randint(1, 100)
    matrix_N = np.random.randint(1, 100)
    matrix_K = np.random.randint(1, 100)

    # 定义矩阵乘法参数
    layer = {
        'M': matrix_M,
        'N': matrix_N,
        'K': matrix_K
    }
    print(layer)

    # 创建输入数据
    left_matrix = np.random.randint(-128, 127, size=(matrix_M, matrix_K))
    right_matrix = np.random.randint(-128, 127, size=(matrix_K, matrix_N))
    bias_vector = np.random.randint(-128, 127, size=(matrix_N))

    left_matrix_m1k1m0k0 = matrix_MK2M1K1M0K0(left_matrix)
    right_matrix_k1n1k0n0 = matrix_KN2K1N1K0N0(right_matrix)
    bias_n1n0 = bias_N2N02N1N0(bias_vector, layer)
    result_matrix_m1n1m0n0 = matmul_m1k1m0k0_k1n1k0n0(left_matrix_m1k1m0k0, right_matrix_k1n1k0n0, bias_n1n0)
    result_matrix_mn = matrix_M1N1M0N02MN(result_matrix_m1n1m0n0, layer)

    golden_matrix_mn = matmul_mk_kn(left_matrix, right_matrix, bias_vector)

    diff = result_matrix_mn - golden_matrix_mn
    if(np.abs(diff).sum() == 0.0):
        print("gemm pass")
    else:
        print("gemm fail")
    return True

test_num = 10
while(test_num > 0):
    if(test_matmul()):
        test_num -= 1