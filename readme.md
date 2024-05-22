## Introduction
Fine-tuning Large Language Model (LLM) is a necessary endeavor that an ML practitionar must undertake to tune the model 
parameters on their data for a performance sensitive applications. However, the compute requirements to fine-tune are 
prohibitively expensive. The compute budget must account for both the model weights as well as the gradient updates 
during tuning. Parameter efficient fine-tuning techniques address one or both of these concerns by freezing and/or 
[[Quantization|quantizing]] the model parameters. It further introduces parameters for learning the new concept in a 
smaller subspace than the original parameter space. Out of these PEFTs, Low Rank Adaptation based methods have enjoyed 
popularity in recent years. This document concerns itself with the application of LoRA.

The methods are compared in this study with the zero-shot performance and full parameter fine tuning wherever 
applicable. The training time, inference speed, and memory requirements are used to perform the comparison. I will only 
study RoBERTa and finetune it to the task of SQuAD for now to limit the scope of this test. In the next section, I will 
discuss the mathematical formulation of LoRA in detail. The following section will discuss the experiments performed. 
Finally, the results and discussion section is presented. 

## Low Rank Adaptation
Neural networks composes a series of linear transformation of the input with intermediate non-linear activations. 
Typically, rectified linear function is used for activation. When adapting a pre-trained network to some dataset, all 
of the models are fine-tuned. Since the weights of the network are adjusted using the loss function gradient, an 
additional memory must be reserved to store these updates before applying. 

The idea behind low rank adaptation is to freeze the original network parameters and add a product of two matrices to 
each linear transformation. These newly introduced matrices have a much smaller inner dimension as a result of which 
the compute requirements for training are reduced dramatically. Suppose, $P_f$ represents the frozen parameters of some 
linear layer and the newly introduced low-rank matrices are $A$ and $B$. Then, the parameters of the model to be adapted 
are computed as $P_{new} = P_f + AB$. Here, the internal dimension of $A$ and $B$ are specified with a hyperparameter 
$r$, which is also used to signify the rank of these matrices. 

Training the matrices $A$ and $B$ requires the network to be trained is the same as the original network in the 
beginning. This can be achieved by setting either $A$, $B$ or both to be 0. I will investigate all of these scenarios. 
Following this, comes rank selection. I will try two settings one in which the rank is computed based on the rank of the 
original matrix and the second in which the rank is specified as a hyperparameter. The next section will describe 
different settings I intend to test. 

## Experimentation
I discovered a few avenues for exploration in the previous section that answer how LoRA based tuning is affected by 
network initialization and rank selection. But first, the benchmark against which to compare the performance must be 
selected. Since the need for parameter efficient tuning is realized much strongly in Large Language Models, we will 
focus on language tasks. As mentioned previously, it is important to note the power of these models across different 
sizes. As such, we will consider BERT as our smaller model and gradually increase model complexity to LLAMA-8B. 
Specifically, we will consider the base, and large for initial benchmarking. For now, I am going to start with SQUAD.

As noted earlier, I will consider four settings - random initialization for both matrices, 0 initialization for one of 
them and then for both of them. Along side, I will also consider different ranks. First, I should look at the rank of 
each parameter matrix. 

## Results