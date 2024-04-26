import torch
from torch import float32, int8, float16, bfloat16, float8_e4m3fn as float8


def quantization(x, s, z, alpha_q, beta_q, target_dtype="float16") -> torch.Tensor:
    if target_dtype == "float16":
        x_q = (1 / s * x + z).to(float16)
    elif target_dtype == "float8":
        x_q = (1 / s * x + z).to(float8)
    elif target_dtype == "bfloat":
        x_q = (1 / s * x + z).to(bfloat16)
    elif target_dtype == "int8":
        # x_q = np.round(1 / s * x + z, decimals=0)
        x_q = (1 / s * x + z).to(int8)
    else:
        raise ValueError("Unsupported target_dtype: {}".format(target_dtype))

    # x_q = torch.clamp(x_q, min=alpha_q, max=beta_q)
    return x_q


# def quantization_int8(x, s, z):
#     x_q = quantization(x, s, z, alpha_q=-128, beta_q=127)
#     x_q = x_q.int8()
#
#     return x_q


def dequantization(x_q, s, z):
    # x_q - z might go outside the quantization range.
    x_q = x_q.to(float32)
    x = s * (x_q - z)

    return x


def generate_quantization_constants(alpha, beta, alpha_q, beta_q):
    # Affine quantization mapping
    s = (beta - alpha) / (beta_q - alpha_q)
    z = (beta * alpha_q - alpha * beta_q) / (beta - alpha)

    return s, z


def generate_quantization_int8_constants(alpha, beta):
    b = 8
    alpha_q = -2 ** (b - 1)
    beta_q = 2 ** (b - 1) - 1

    s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q)

    return s, z


def quantization_matrix_multiplication(a_q, b_q, s, z, q_type="float16"):
    """
    Matrix multiplication with quantization. Takes the following inputs:
    :param a_q: quantized matrix A
    :param b_q: quantized matrix B
    :param s: list of s of the matrices
    :param z: list of z of the matrices
    :param q_type: quantization type
    :return: Y: matrix Y
    """
    n = a_q.shape[0]
    sa, sb = s
    za, zb = z
    Y = torch.zeros(n, n)
    accum_a = torch.zeros(n)
    accum_b = torch.zeros(n)
    for j in range(n):
        accum_a[j] = a_q[:, j].sum().to(float32)
        accum_b[j] = b_q[:, j].sum().to(float32)

    for i in range(n):
        for j in range(n):
            Y[i, j] = (a_q[i, :] @ b_q[:, j]).sum().to(float32)
            Y[i, j] -= accum_a[j] * zb[0]
            Y[i, j] -= accum_b[j] * za[0]
            Y[i, j] += (n * za[0] * zb[0]).to(float32)

    # Y = a_q @ b_q - zb * a_q.sum(dim=0) - za * b_q.sum(dim=0) + n * za * zb

    Y = sa * sb * Y
    return Y.to(float32)


def main():
    # Set random seed for reproducibility
    random_seed = 0
    n = 100
    torch.manual_seed(random_seed)
    q_type = "int8"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_alpha = 1
    max_diff = 1

    alpha_q, beta_q = 127, -127
    max_diff_q = -127

    alpha = (torch.rand(2, 1) * 2 - 1) * max_alpha
    beta = torch.clamp(alpha + (torch.rand(2, 1) - 1) * max_diff, -max_alpha * torch.ones(2, 1), alpha)


    print("alpha: ", alpha)
    print("beta: ", beta)
    print("alpha_q: ", alpha_q)
    print("beta_q: ", beta_q)

    s, z = generate_quantization_constants(alpha, beta, alpha_q, beta_q)
    print("s: ", s)
    print("z: ", z)

    a = torch.rand(n, n) * 2 * alpha[0] - beta[0]
    b = torch.rand(n, n) * 2 * alpha[1] - beta[1]

    a_q = quantization(a, s[0], z[0], alpha_q, beta_q, target_dtype=q_type)
    b_q = quantization(b, s[1], z[1], alpha_q, beta_q, target_dtype=q_type)

    print("a: ", a)
    print("a_q: ", a_q)

    print("b: ", b)
    print("b_q: ", b_q)

    Y_q = quantization_matrix_multiplication(a_q, b_q, s, z, q_type)
    Y = a @ b
    print("Y_q: ", Y_q)
    print("Y:", Y)

    error = torch.mean((Y_q - Y) ** 2)
    print("error: ", error)


if __name__ == "__main__":
    main()
    print("Done!")
