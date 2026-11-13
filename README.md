# Experiment 5: Neural Network Layers Using CUDA

## 1. Aim

To implement neural network layers using CUDA, execute two-layer forward propagation on an NVIDIA GPU, validate the results against a CPU implementation, and compare CPU and GPU execution performance.

## 2. Objectives

1. Implement matrix multiplication with bias, ReLU, and Sigmoid using CUDA.
2. Execute two-layer neural network forward propagation on the GPU.
3. Validate GPU results against the CPU implementation.
4. Measure execution time and calculate GPU speedup over three runs.

## 3. Hardware and Software Requirements

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation |
| GPU Architecture | Ada Lovelace |

### Software

| Component | Specification |
|---|---|
| Operating System | Ubuntu / Linux |
| GPU Programming Platform | CUDA |
| Compiler | NVCC |
| Programming Language | CUDA C/C++ |
| GPU Driver | NVIDIA Driver |

## 4. CUDA Environment Verification

The NVIDIA GPU and CUDA development environment were verified before execution.

**Commands:**

```bash
nvidia-smi
nvcc --version
```

**Detected GPU:**

```text
NVIDIA RTX 4500 Ada Generation
```

The GPU was detected successfully, and the CUDA compiler was available.

## 5. Neural Network Architecture

A two-layer neural network was implemented using CUDA.

```text
             INPUT
          256 x 128
               |
               v
       MATRIX MULTIPLICATION
               |
          ADD BIAS
               |
               v
        ReLU ACTIVATION
               |
               v
          HIDDEN LAYER
           256 x 64
               |
               v
       MATRIX MULTIPLICATION
               |
          ADD BIAS
               |
               v
      SIGMOID ACTIVATION
               |
               v
          OUTPUT LAYER
           256 x 10
```

### Network Dimensions

| Layer | Dimensions |
|---|---:|
| Input | 256 × 128 |
| Hidden | 256 × 64 |
| Output | 256 × 10 |

### Forward Propagation Equations

**Hidden layer:**

```text
Hidden = ReLU(Input × W1 + b1)
```

**Output layer:**

```text
Output = Sigmoid(Hidden × W2 + b2)
```

The experiment performs forward propagation only. Training and backpropagation are not included.

## 6. CUDA Kernel Implementation

Three individual CUDA operations were implemented and tested.

### 6.1 Matrix Multiplication with Bias

```text
Y = X × W + b
```

Performs matrix multiplication and adds the corresponding bias to each output element.

### 6.2 ReLU Activation

```text
ReLU(x) = max(0, x)
```

Negative values are converted to zero, while positive values remain unchanged.

### 6.3 Sigmoid Activation

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

Converts the output values into the range [0, 1].

### Kernel Validation Results

| CUDA Kernel | Result |
|---|---|
| Matrix Multiplication + Bias | PASSED |
| ReLU | PASSED |
| Sigmoid | PASSED |

All three kernel tests passed in all three experimental runs.

## 7. CPU and GPU Implementation

Two implementations were used for performance and correctness comparison.

**CPU implementation:**
- Performs matrix multiplication with bias.
- Applies ReLU activation.
- Performs the second matrix multiplication.
- Applies Sigmoid activation.

**GPU implementation:**
- Executes the CUDA matrix multiplication kernel.
- Executes the ReLU kernel.
- Executes the second matrix multiplication kernel.
- Executes the Sigmoid kernel.

The CPU implementation acts as the reference for validating the GPU output.

## 8. Compilation and Execution

### Verify GPU

```bash
nvidia-smi
```

### Verify CUDA Compiler

```bash
nvcc --version
```

### Compile CUDA Program

```bash
nvcc neural_network.cu -o neural_network
```

### Execute Three Experimental Runs

```bash
mkdir -p results

./neural_network | tee results/run1.txt

./neural_network | tee results/run2.txt

./neural_network | tee results/run3.txt
```

### View Results

```bash
cat results/run1.txt
cat results/run2.txt
cat results/run3.txt
```

## 9. Experimental Output

### GPU Detection

```text
EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA

GPU detected: NVIDIA RTX 4500 Ada Generation
```

### Individual Kernel Testing

```text
Testing individual CUDA kernels...

Matrix multiplication + bias: PASSED
ReLU: PASSED
Sigmoid: PASSED
```

### Forward Propagation Dimensions

```text
Forward-pass dimensions:

Input: 256 x 128
Hidden layer: 256 x 64
Output: 256 x 10
```

The individual CUDA kernels passed their tests, and the complete two-layer forward pass was executed.

## 10. Performance Evaluation

The CPU and GPU implementations were evaluated over three experimental runs.

Three execution measurements were considered:

- **CPU time:** Time taken by the CPU reference implementation.
- **GPU kernel-only time:** Time spent executing the GPU kernels.
- **GPU end-to-end time:** Overall GPU execution time, including the additional measured overhead.

### 10.1 Average Performance Results

The following values are the actual averages shared from the three experimental runs.

| Parameter | Average Value |
|---|---:|
| CPU Execution Time | 0.581175 ms |
| GPU Kernel-Only Time | 0.027898 ms |
| GPU End-to-End Time | 0.203128 ms |
| Kernel-Only Speedup (reported) | 21.76× |
| End-to-End Speedup (reported) | 3.03× |
| Maximum Output Difference | 0.000000 |

### 10.2 Kernel-Only Speedup Calculation

Formula:

```text
Speedup = CPU Execution Time / GPU Execution Time
```

Substituting the average values:

```text
Speedup = 0.581175 / 0.027898

        = 20.83× approximately
```

**Important:** The previously reported kernel-only speedup was 21.76×. However, dividing the displayed average CPU time by the displayed average kernel-only time gives approximately 20.83×. The 21.76× value may represent the average of individual run speedups rather than the ratio of average times.

### 10.3 End-to-End Speedup Calculation

```text
Speedup = CPU Execution Time / GPU End-to-End Time

        = 0.581175 / 0.203128

        = 2.86× approximately
```

The previously reported end-to-end speedup was 3.03×. As with kernel-only speedup, this may be the average of individual run speedups.

**Note:** The exact three-run speedup averages should be retained from the original measurements. The ratios calculated above are based on the displayed average execution times.

### 10.4 Execution Time Comparison

| Implementation | Execution Time |
|---|---:|
| CPU | 0.581175 ms |
| GPU Kernel-Only | 0.027898 ms |
| GPU End-to-End | 0.203128 ms |

The GPU kernel-only execution time is significantly lower than the CPU execution time. The GPU end-to-end time is higher than the kernel-only time because of additional execution overhead.

### 10.5 CPU and GPU Output Comparison

```text
Maximum output difference: 0.000000
```

The CPU and GPU implementations produced matching outputs at the reported precision, demonstrating numerical correctness for the tested inputs.

## 11. Results and Observations

1. The NVIDIA RTX 4500 Ada Generation GPU was detected successfully.
2. CUDA compilation and execution were successful.
3. Matrix multiplication with bias passed validation.
4. ReLU activation passed validation.
5. Sigmoid activation passed validation.
6. The two-layer forward propagation was executed with input dimensions 256 × 128, hidden dimensions 256 × 64, and output dimensions 256 × 10.
7. All three kernel tests passed across the three experimental runs.
8. The maximum CPU-GPU output difference was 0.000000.
9. The average CPU execution time was 0.581175 ms.
10. The average GPU kernel-only execution time was 0.027898 ms.
11. The average GPU end-to-end execution time was 0.203128 ms.
12. GPU execution demonstrated a performance improvement over the CPU implementation.

## 12. Conclusion

The experiment successfully implemented neural network layers using CUDA and executed two-layer forward propagation on the NVIDIA RTX 4500 Ada Generation GPU.

Matrix multiplication with bias, ReLU, and Sigmoid CUDA kernels were implemented and validated successfully. The GPU implementation was compared with the CPU reference implementation, and the maximum output difference was 0.000000 at the reported precision.

Performance measurements were conducted over three experimental runs. The average CPU execution time was 0.581175 ms, GPU kernel-only execution time was 0.027898 ms, and GPU end-to-end execution time was 0.203128 ms.

The experiment demonstrated the effectiveness of GPU acceleration in reducing neural network execution time compared with CPU execution.

**Result:** Neural network layers were successfully implemented and executed using CUDA, and their correctness and performance were evaluated.

## 13. Experimental Evidence

The following screenshots should be maintained in the `screenshots/` folder:

| No. | Evidence |
|---|---|
| 1 | NVIDIA GPU verification using `nvidia-smi` |
| 2 | CUDA compiler verification using `nvcc --version` |
| 3 | Successful CUDA compilation |
| 4 | Matrix multiplication, ReLU, and Sigmoid kernel validation |
| 5 | Two-layer forward propagation output |
| 6 | First experimental run |
| 7 | Second experimental run |
| 8 | Third experimental run |
| 9 | CPU and GPU performance comparison |
| 10 | Saved experimental results |

## 14. Final Project Structure

```text
Experiment-5/
│
├── neural_network.cu
├── neural_network
│
├── results/
│   ├── run1.txt
│   ├── run2.txt
│   ├── run3.txt
│   └── comparison_results.txt
│
├── screenshots/
│   ├── gpu_verification.png
│   ├── cuda_version.png
│   ├── compilation.png
│   ├── kernel_validation.png
│   ├── forward_pass.png
│   └── performance_results.png
│
└── README.md
```

*The structure is a recommended organization; retain the actual filenames in your project.*

---

**Final Result: Experiment 5 – Neural Network Layers Using CUDA successfully implemented, executed, validated, and evaluated for CPU-GPU performance comparison.*
