# Fast 4-way parallel radix sort for WebGPU

This is a WebGPU implementation for the radix sort algorithm as described in the paper [Fast 4-way parallel radix sorting on GPUs](https://www.sci.utah.edu/~csilva/papers/cgf.pdf]).

- Sort large arrays of integers on GPU using WGSL compute shaders
- Sort both a buffer of `keys` and associated `values` at the same time. The sort is made based on the `keys` buffer.
- Supports arrays of arbitrary size

## Parallel Prefix Sum (Scan)

This algorithm relies on another widely used parallel algorithm called the prefix sum (or scan).

Thus, this repository also contains a WebGPU implementation of the method described in the following publication: [Parallel Prefix Sum (Scan) with CUDA](https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf).

- Includes the "Work Efficient Parallel Scan" optimization
- Recurses on itself until the input fits one workgroup
- Supports arrays of arbitrary size

## Performance Tests

I've made a minimal web demo on which you can run the algorithm locally: https://webgpu-radix-sort.vercel.app/

The following tests were done on a laptop using an Intel Core i9 @ 2.90GHz (CPU) and a NVIDIA RTX 3080TI (GPU). The vertical axis is logarithmic.

![results](./results.jpg)

## Usage

```
/**
* @param {GPUDevice} device
* @param {GPUBuffer} keys - Buffer containing the keys to sort
* @param {GPUBuffer} values - Buffer containing the associated values
* @param {number} count - Number of elements to sort
* @param {number} bit_count - Number of bits per element (default: 32)
* @param {object} workgroup_size - Workgroup size in x and y dimensions. (x * y) must be a power of two
* @param {boolean} local_shuffle - Enable "local shuffling" optimization for the radix sort kernel (default: false)
* @param {boolean} avoid_bank_conflicts - Enable "avoiding bank conflicts" optimization for the prefix sum kernel (default: false)
*/

const radixSortKernel = new RadixSortKernel({
        device,
        keys: keysBuffer,
        values: valuesBuffer,
        count: keys.length,
        workgroup_size: { x: 16, y: 16 },
        bit_count: bit_count,
})

...

const pass = encoder.beginComputePass()
radixSortKernel.dispatch(pass)
```

## Current limitations

- Only supports integer sorting (up to 32 bits)
- Only supports bit count that are multiple of 4
- Only supports array up to 16,776,960 elements (using the default workgroup size of 16x16. Can be increased to 67,107,840 elements by requesting for a workgroup size of 32x32)

## Implementation detalis

### 1) Fast 4-way parallel radix sort

#### Local shuffling and coalesced mapping

In the original paper, a section describes how the data is locally shuffled (sorted) within the workgroups before computing the prefix block sum. This is done in order to address the issue of non-coalseced writing on the global memory.
By sorting the input data locally, it improves the memory read/write patterns during the final reordering step, resulting in a 36% performance increase in the original paper.
However, adding this process to my WebGPU implementation didn't seem to have such an impact on the performance. This can be explained by the fact that this paper was designed for an old version of CUDA (2009) and graphics card architectures have evolved since, being more optimized "out of the box" today.
For this reason, this process is disabled by default, but it can be enabled with the parameter `local_shuffle`.

#### Order checking

To improve performance in cases where the input data is already sorted or nearly sorted, the original paper describes a method that will initially scan the input array before each pass of the algorithm. In the case where the input array is sorted, the algorithm will exit early and prevent unecessary calculations. This can be useful if the data is sorted every frame with few changes between each frame for instance.

In WebGPU however, this check is not as easy to achieve while keeping optimal performances, mainly because every pass and their attributes needs to be encoded prior to being sent to the GPU, without a way to conditionally choose to stop the execution from the GPU.

I've made some tests with a custom `CheckSortKernel` that would do a parallel reduction on the input array to check if it is sorted or not and store the result in a GPU buffer. This way I could use the `dispatchWorkgroupsIndirect` method to dynamically change the number of workgroups that are created for the other pipelines of the algorithm.
However, I observed a strong negative impact on the performance and it didn't seem to improve already-sorted arrays that much so I preferred not to include it until I find a more optimized way to do it.

### 2) Parallel Prefix Sum with CUDA

#### Avoiding Bank Conflicts

In the original publication, the final version of the algorithm contains an optimization that aims at improving shared memory access patterns in order to reduce bank conflicts. 
Bank conflicts happen when multiple threads are accessing the same memory bank at the same time, resulting in slower sequential processing.
To fix this issue, the authors introduced a macro that offsets every shared memory access within their algorithm in a clever way such that the bank conflics are minimized.
Similar to the above note on `Local shuffling and coalesced mapping`, this algorithm was designed for 2007-ish versions of CUDA and adding it to my WebGPU implementation didn't seem to have an impact on the performance.
It's disabled by default but can be enabled using the `avoid_bank_conflicts` parameter.

## Project structure

```
.
├── index.html                          # Demo page for performance profiling
├── src
│   ├── index.js                        # Entry point and example usage
│   ├── tests.js                        # Utilities for profiling and testing
│   │
│   ├── kernels                      
│   │   ├── RadixSortKernel.js          # 4-way Radix Sort kernel definition
│   │   ├── PrefixSumKernel.js          # Parallel Prefix Sum kernel definition
│   │   
│   ├── shaders                         # Contains the WGSL shader sources as javascript strings
│       ├── radix_sort.js               # Compute local prefix sums and block sums
│       ├── radix_sort_reorder.js       # Reorder data to sorted position
│       ├── prefix_sum.js               # Parallel Prefix Sum (scan) algorithm   
│       │
│       ├── optimizations               # Contains shaders including optimizations (see "Implementation" section)
│           ├── radix_sort_local_shuffle.js
│           ├── prefix_sum_no_bank_conflict.js      
```

## References

- [Fast 4-way parallel radix sorting on GPUs](https://www.sci.utah.edu/~csilva/papers/cgf.pdf])
- [Parallel Prefix Sum (Scan) with CUDA](https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf)