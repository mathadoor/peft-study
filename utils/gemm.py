import torch
from torch import float32, int8, int16, int32

bits = 8
alpha_q, beta_q = -2**(bits-1), 2**(bits-1)-1

def quantization(x, s, z):

    x_q = torch.round(1 / s * x + z)
    x_q = torch.clamp(x_q, alpha_q, beta_q)
    # x_q = np.round(1 / s * x + z, decimals=0)
    # x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)

    return x_q


def quantization_int8(x, s, z):

    x_q = quantization(x, s, z)
    x_q = x_q.to(int8)
    return x_q


def dequantization(x_q, s, z):

    # x_q - z might go outside the quantization range.
    x_q = x_q.int()
    x = s * (x_q - z)
    x = x.to(float32)

    return x


def generate_quantization_constants(alpha, beta):

    # Affine quantization mapping
    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

    return s, z


def generate_quantization_int8_constants(alpha, beta):

    b = 8

    s, z = generate_quantization_constants(alpha=alpha, beta=beta)
    return s, z


def quantization_matrix_multiplication_int8(A_q, B_q, s_A, z_A, s_B, z_B, s_Y, z_Y):

    p = B_q.shape[0]

    # Y_q_simulated is FP32
    Y_q_simulated = torch.zeros(A_q.shape[0], B_q.shape[1], dtype=int32)
    # outer produce
    for k in range(p):
        Y_q_simulated += torch.einsum("i,j->ij", (A_q[:, k].to(int) - z_A), (B_q[k, :].to(int) - z_B))

    Y_q_simulated = s_A * s_B * Y_q_simulated / s_Y + z_Y

    Y_q_simulated = torch.round(Y_q_simulated)
    Y_q_simulated = torch.clamp(Y_q_simulated, min=alpha_q, max=beta_q)
    Y_q_simulated = Y_q_simulated.to(int8)
    return Y_q_simulated


def main():

    # Set random seed for reproducibility
    random_seed = 0
    torch.random.manual_seed(random_seed)

    # Random matrices

    m = 100
    p = 100
    n = 100

    # X
    alpha_X = -1
    beta_X = 1
    s_X, z_X = generate_quantization_int8_constants(alpha=30*alpha_X, beta=30*beta_X)
    X = torch.randn(m, p) * (beta_X - alpha_X) + alpha_X
    X_q = quantization_int8(x=X, s=s_X, z=z_X)
    X_q_dq = dequantization(x_q=X_q, s=s_X, z=z_X)

    # W
    alpha_W = -1
    beta_W = 1
    s_W, z_W = generate_quantization_int8_constants(alpha=30*alpha_W, beta=30*beta_W)
    W =  torch.randn(p, n) * (beta_W - alpha_W) + alpha_W
    W_q = quantization_int8(x=W, s=s_W, z=z_W)
    W_q_dq = dequantization(x_q=W_q, s=s_W, z=z_W)

    # Y
    alpha_Y = -1
    beta_Y = 1
    s_Y, z_Y = generate_quantization_int8_constants(alpha=900*alpha_Y, beta=900*beta_Y)
    Y_expected = torch.matmul(X, W)
    Y_q_expected = quantization_int8(x=Y_expected, s=s_Y, z=z_Y)

    Y_expected_prime = torch.matmul(X_q_dq, W_q_dq)
    Y_expected_prime_q = quantization_int8(x=Y_expected_prime, s=s_Y, z=z_Y)
    Y_expected_prime_q_dq = dequantization(x_q=Y_expected_prime_q,
                                           s=s_Y,
                                           z=z_Y)

    print("Expected FP32 Y:")
    print(Y_expected)
    print("Expected FP32 Y Quantized:")
    print(Y_q_expected)

    Y_q_simulated = quantization_matrix_multiplication_int8(A_q=X_q,
                                                            B_q=W_q,
                                                            s_A=s_X,
                                                            z_A=z_X,
                                                            s_B=s_W,
                                                            z_B=z_W,
                                                            s_Y=s_Y,
                                                            z_Y=z_Y)
    Y_simulated = dequantization(x_q=Y_q_simulated, s=s_Y, z=z_Y)

    print("Expected Quantized Y_q from Quantized Matrix Multiplication:")
    print(Y_q_simulated)
    print(
        "Expected Quantized Y_q from Quantized Matrix Multiplication Dequantized:"
    )
    print(Y_simulated)

    # Ensure the algorithm implementation is correct
    rtol = 1e-5
    assert torch.isclose(Y_simulated, Y_expected_prime_q_dq, rtol=rtol).all()
    assert torch.isclose(Y_q_simulated, Y_expected_prime_q, rtol=rtol).all()
    # assert (np.array_equal(Y_q_simulated, Y_expected_prime_q))


if __name__ == "__main__":

    main()