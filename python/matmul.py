from lib import *

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

    left_matrix_k1mk0 = MK2K1MK0(left_matrix)
    right_matrix_n1kn0 = KN2N1KN0(right_matrix)
    left_matrix_m1k1m0k0 = K1MK02M1K1M0K0(left_matrix_k1mk0)
    right_matrix_k1n1k0n0 = N1KN02K1N1K0N0(right_matrix_n1kn0)
    bias_n1n0 = N2N1N0(bias_vector)
    result_matrix_m1n1m0n0 = matmul_m1k1m0k0_k1n1k0n0(left_matrix_m1k1m0k0, right_matrix_k1n1k0n0, bias_n1n0)
    result_matrix_mn = M1N1M0N02MN(result_matrix_m1n1m0n0, layer)

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