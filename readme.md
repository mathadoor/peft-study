## Objective
The purpose of this document is to present a report on the pros and cons of quantization. In lieu of that, I will 
specifically target the definition, and applicability of this technique. The first section will deal with defining what 
quantization and its various forms. The next section briefly talks about where it can be applied. The following section 
will discuss the experimental setup and finally, the last section will discuss the results. 

## Definitions
The idea of quantization is to project a higher precision floating point number to a lower precision one. The target
data type can be either an integer or a floating point number. Before we delve into the specifics, let's first 
understand how a floating point number is represented. 

A float number of 32 bit precision is represented by splitting the bits into three parts: sign bit, exponent, and 
mantissa. The sign bit is 1 bit that indicates whether the number is positive or negative. The exponent is 8 bits that 
represent the power of 2 with a bias. The mantissa is the remaining 23 bits that represent the fractional part of the 
number. Now, let us say we want to represent the number 0.375 in 32 bit precision. First, we convert the number to binary.
$ 0.375 = 2^{-2} + 2^{-3} = 0.011 $ in binary. The next step is to normalize the number. We shift the binary point to the
left until the first bit is 1. $ 0.011 = 1.1 \times 2^{-2} $. The sign bit is 0. The exponent is -2, but we add a bias
to account for the negative exponents. The bias is 127, so the exponent is 125. The mantissa is 11000000000000000000000.
Therefore, the 64 bit binary representation of 0.375 is [0][01111101][10000000000000000000000]. 

Notice the number of mantissa bits represent the precision of the number, while the exponent bits are needed to represent 
a wide range of numbers. Thus by reducing the number of bits in the representation, we can reduce the memory footprint 
of storing the number. This has a downstream effect on the computational cost of operations. As such, quantization is
a technique used to reduce the memory footprint as well as the computational cost of operations. This can be utilized in 
deep learning models to load bigger models on smaller devices while also speeding up the computation. However, there are 
a few concerns that need to be addressed before we can use quantization. First, we need to quantify the extent of 
gain in memory and speed. Second, we need to understand how the model accuracy is affected. Finally, we need to 
understand how to quantize the model. In the next section, we will discuss various forms of quantization techniques.

## Forms of Quantization
Let us first understand how quantization is applied to a model. The process can be seen as that of finding a linear mapping 
from a larger set of discrete numbers to a smaller set. These numbers correspond to the weights or activations in some 
part of the model. Suppose the source set of numbers lie in the range $x \in [a, b]$ and the target set of numbers lie in the
range $x_q \in [c, d]$. The linear mapping is $x = \alpha (x_q - \beta$). Here, $\alpha$ is the scale factor and $\beta$ is the
zero point. 

Now, let us see how we can categorize quantization methods. The first way is to categorize them based on the
data type of the target number. The target number can be either an integer or a floating point number. In addition, we 
also consider the accumulation data type. The accumulation data type is the data type used to store the intermediate 
results of the computation. The second way to categorize quantization techniques is based on the granularity of the
quantization. The granularity can be at the level of the tensor, layer, or model itself. 

The third way is to categorize based on whether the quantization is done post training or during training. Within post
training quantization, we can further categorize based on whether the quantization is done statically or dynamically. 
Static quantization is when the scale factor and zero point are calculated once and used throughout the inference process.
A calibration dataset is needed to estimate the appropriate quantization parameters. Dynamic quantization is when the 
quantization parameters are calculated on the fly. This is useful when the input data distribution changes during inference.
Finally, during training quantization is when the quantization is done during the training process itself. The model is 
quantized during training so that the optimizer can adjust the weights to account for the quantization error. Next, we will
discuss the effect of quantization on computation.

## Experimental Setup
### Matrix Multiplication
The first experiment is to understand the effect of quantization on matrix multiplication. The matrix multiplication is 
done using the PyTorch library. Both the accuracy and the time taken to multiply two matrices are recorded for different
quantization levels. We consider float32 as the base. The quantization levels are FP16, FP8, bfloat16, int8, and int4. The
accumulation data type is kept as float32. The matrix sizes are varied from 10 x 10, 100 x 100, 1000 x 1000, 10000 x 10000.
Each matrix is sampled from a normal distribution with mean 0 and standard deviation 1.

Now let us see how quantized matrix multiplication works. Suppose we want to perform the operation $Y = A \time B$. Here,
$Y \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times k}$, and $B \in \mathbb{R}^{k \times n}$. Each element in 
Y is computed as follows: $y_{ij} = \sum_{l=1}^{k} a_{il} \times b_{lj}$. As such, we will need to do k floating point 
multiplications and additions to compute each element in Y. Thus, we will need to perform $m \times n \times k$ floating
point multiplications and additions to compute the entire matrix Y. We know that floating point operations are expensive.
Thus, we can reduce the cost of computation by quantizing the matrices A and B. Let us see how this is done.

Let's replace each element in the equation by its quantized value. 
$y_{ij} = \sum_{l=1}^{k} \alpha_a (a_{q, il} - \beta_a) \times \alpha_b (b_{q, lj} - \beta_b) $.
$y_{ij} = \alpha_a \alpha_b \sum_{l=1}^{k} (a_{q, il} - \beta_a) \times (b_{q, lj} - \beta_b) $.
$y_{ij} = \alpha_a \alpha_b (\sum_{l=1}^{k} a_{q, il} \times b_{q, lj} - \beta_a \sum_{l=1}^{k} b_{q, lj} - \beta_b \sum_{l=1}^{k} a_{q, il} + k * \beta_a \beta_b) $.
Now, let us substitute the expression for the $y_{ij}$ in terms of quantized values. 
$y_{q, ij} = \frac{beta_y}{alpha_y} + \frac{\alpha_a \alpha_b}{alpha_b} (y_{q, ij} - \beta_a \sum_{l=1}^{k} b_{q, lj} - \beta_b \sum_{l=1}^{k} a_{q, il} + k * \beta_a \beta_b) $.
### Linear Regression
### Computer Vision Tasks
### Language Modeling Tasks
