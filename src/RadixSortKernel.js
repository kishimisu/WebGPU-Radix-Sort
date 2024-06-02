import PrefixSumKernel from "./PrefixSumKernel"
import radixSortSource from "./shaders/radix_sort"
import radixSortSource_LocalShuffle from "./shaders/optimizations/radix_sort_local_shuffle"
import reorderSource from "./shaders/radix_sort_reorder"
import CheckSortKernel from "./CheckSortKernel"
import { create_buffer_from_data, find_optimal_dispatch_size } from "./utils"

class RadixSortKernel {
    /**
     * Perform a parallel radix sort on the GPU given a buffer of keys and (optionnaly) values
     * Note: The buffers are sorted in-place.
     * 
     * Based on "Fast 4-way parallel radix sorting on GPUs"
     * https://www.sci.utah.edu/~csilva/papers/cgf.pdf]
     * 
     * @param {GPUDevice} device
     * @param {GPUBuffer} keys - Buffer containing the keys to sort
     * @param {GPUBuffer} values - (optional) Buffer containing the associated values
     * @param {number} count - Number of elements to sort
     * @param {number} bit_count - Number of bits per element (default: 32)
     * @param {object} workgroup_size - Workgroup size in x and y dimensions. (x * y) must be a power of two
     * @param {boolean} check_order - Enable "order checking" optimization. Can improve performance if the data needs to be sorted in real-time and doesn't change much. (default: false)
     * @param {boolean} local_shuffle - Enable "local shuffling" optimization for the radix sort kernel (default: false)
     * @param {boolean} avoid_bank_conflicts - Enable "avoiding bank conflicts" optimization for the prefix sum kernel (default: false)
     */
    constructor({
        device,
        keys,
        values,
        count,
        bit_count = 32,
        workgroup_size = { x: 16, y: 16 },
        check_order = false,
        local_shuffle = false,
        avoid_bank_conflicts = false,
    } = {}) {
        if (device == null) throw new Error('No device provided')
        if (keys == null) throw new Error('No keys buffer provided')
        if (!Number.isInteger(count) || count <= 0) throw new Error('Invalid count parameter')
        if (!Number.isInteger(bit_count) || bit_count <= 0 || bit_count > 32) throw new Error(`Invalid bit_count parameter: ${bit_count}`)
        if (!Number.isInteger(workgroup_size.x) || !Number.isInteger(workgroup_size.y)) throw new Error('Invalid workgroup_size parameter')
        if (bit_count % 4 != 0) throw new Error('bit_count must be a multiple of 4')

        this.device = device
        this.count = count
        this.bit_count = bit_count
        this.workgroup_size = workgroup_size
        this.check_order = check_order
        this.local_shuffle = local_shuffle
        this.avoid_bank_conflicts = avoid_bank_conflicts

        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y
        this.workgroup_count = Math.ceil(count / this.threads_per_workgroup)
        this.prefix_block_workgroup_count = 4 * this.workgroup_count

        this.has_values = (values != null) // Is the values buffer provided ?

        this.dispatchSize = {}  // Dispatch dimension x and y
        this.shaderModules = {} // GPUShaderModules
        this.kernels = {}       // PrefixSumKernel & CheckSortKernels
        this.pipelines = []     // List of passes
        this.buffers = {        // GPUBuffers
            keys: keys,
            values: values
        }       

        // Create shader modules from wgsl code
        this.create_shader_modules()
        
        // Create multi-pass pipelines
        this.create_pipelines()
    }

    create_shader_modules() {
        // Remove every occurence of "values" in the shader code if values buffer is not provided
        const remove_values = (source) => {
            return source.split('\n')
                         .filter(line => !line.toLowerCase().includes('values'))
                         .join('\n')
        }

        const blockSumSource = this.local_shuffle ? radixSortSource_LocalShuffle : radixSortSource
        
        this.shaderModules = {
            blockSum: this.device.createShaderModule({
                label: 'radix-sort-block-sum',
                code: this.has_values ? blockSumSource : remove_values(blockSumSource),
            }),
            reorder: this.device.createShaderModule({
                label: 'radix-sort-reorder',
                code: this.has_values ? reorderSource : remove_values(reorderSource),
            })
        }
    }

    create_pipelines() {    
        // Block prefix sum kernel    
        this.create_prefix_sum_kernel()

        // Indirect dispatch buffers
        const dispatchData = this.calculate_dispatch_sizes()

        // GPU buffers
        this.create_buffers(dispatchData)

        // Check sort kernels
        this.create_check_sort_kernels(dispatchData)

        // Radix sort passes for every 2 bits
        for (let bit = 0; bit < this.bit_count; bit += 2) {
            // Swap buffers every pass
            const even      = (bit % 4 == 0)
            const inKeys    = even ? this.buffers.keys : this.buffers.tmpKeys
            const inValues  = even ? this.buffers.values : this.buffers.tmpValues
            const outKeys   = even ? this.buffers.tmpKeys : this.buffers.keys
            const outValues = even ? this.buffers.tmpValues : this.buffers.values

            // Compute local prefix sums and block sums
            const blockSumPipeline = this.create_block_sum_pipeline(inKeys, inValues, bit)
            
            // Reorder keys and values
            const reorderPipeline = this.create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit)

            this.pipelines.push({ blockSumPipeline, reorderPipeline })
        }
    }

    create_prefix_sum_kernel() {
        // Prefix Block Sum buffer (4 element per workgroup)
        const prefixBlockSumBuffer = this.device.createBuffer({
            label: 'radix-sort-prefix-block-sum',
            size: this.prefix_block_workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        // Create block prefix sum kernel
        const prefixSumKernel = new PrefixSumKernel({ 
            device: this.device,
            data: prefixBlockSumBuffer, 
            count: this.prefix_block_workgroup_count,
            workgroup_size: this.workgroup_size,
            avoid_bank_conflicts: this.avoid_bank_conflicts,
        })

        this.kernels.prefixSum = prefixSumKernel
        this.buffers.prefixBlockSum = prefixBlockSumBuffer
    }

    calculate_dispatch_sizes() {
        // Radix sort dispatch size
        const dispatchSize = find_optimal_dispatch_size(this.device, this.workgroup_count)

        // Prefix sum dispatch sizes
        const prefixSumDispatchSize = this.kernels.prefixSum.get_dispatch_chain()

        // Check sort element count (fast/full)
        const check_sort_fast_count = Math.min(this.count, this.threads_per_workgroup * 4)
        const check_sort_full_count = this.count - check_sort_fast_count
        const start_full = check_sort_fast_count - 1

        // Check sort dispatch sizes
        const dispatchSizesFast = CheckSortKernel.find_optimal_dispatch_chain(this.device, check_sort_fast_count, this.workgroup_size)
        const dispatchSizesFull = CheckSortKernel.find_optimal_dispatch_chain(this.device, check_sort_full_count, this.workgroup_size)

        // Initial dispatch sizes
        const initialDispatch = [
            dispatchSize.x, dispatchSize.y, 1, // Radix Sort + Reorder
            ...dispatchSizesFast.slice(0, 3),  // Check sort fast
            ...prefixSumDispatchSize           // Prefix Sum
        ]

        // Dispatch offsets in main buffer
        this.dispatchOffsets = {
            radix_sort: 0,
            check_sort_fast: 3 * 4,
            prefix_sum: 6 * 4
        }

        this.dispatchSize = dispatchSize
        this.initialDispatch = initialDispatch

        return {
            initialDispatch,
            dispatchSizesFull,
            check_sort_fast_count, 
            check_sort_full_count, 
            start_full 
        }
    }

    create_buffers(dispatchData) {
        // Keys and values double buffering
        const tmpKeysBuffer = this.device.createBuffer({
            label: 'radix-sort-tmp-keys',
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })
        const tmpValuesBuffer = !this.has_values ? null : this.device.createBuffer({
            label: 'radix-sort-tmp-values',
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        // Local Prefix Sum buffer (1 element per item)
        const localPrefixSumBuffer = this.device.createBuffer({
            label: 'radix-sort-local-prefix-sum',
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        this.buffers.tmpKeys = tmpKeysBuffer
        this.buffers.tmpValues = tmpValuesBuffer
        this.buffers.localPrefixSum = localPrefixSumBuffer

        // Only create indirect dispatch buffers when check_order optimization is enabled
        if (!this.check_order) {
            return
        }

        // Dispatch sizes (radix sort, check sort, prefix sum)
        const dispatchBuffer = create_buffer_from_data({
            device: this.device, 
            label: 'radix-sort-dispatch-size',
            data: dispatchData.initialDispatch, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT
        })
        const originalDispatchBuffer = create_buffer_from_data({
            device: this.device, 
            label: 'radix-sort-dispatch-size-original',
            data: dispatchData.initialDispatch, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })

        // Dispatch sizes (full sort)
        const checkSortFullDispatchBuffer = create_buffer_from_data({
            label: 'check-sort-full-dispatch-size',
            device: this.device, 
            data: dispatchData.dispatchSizesFull,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT
        })
        const checkSortFullOriginalDispatchBuffer = create_buffer_from_data({
            label: 'check-sort-full-dispatch-size-original',
            device: this.device, 
            data: dispatchData.dispatchSizesFull,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })

        // Flag to tell if the data is sorted
        const isSortedBuffer = create_buffer_from_data({
            label: 'is-sorted',
            device: this.device, 
            data: new Uint32Array([0]), 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        this.buffers.dispatchSize = dispatchBuffer
        this.buffers.originalDispatchSize = originalDispatchBuffer
        this.buffers.checkSortFullDispatchSize = checkSortFullDispatchBuffer
        this.buffers.originalCheckSortFullDispatchSize = checkSortFullOriginalDispatchBuffer
        this.buffers.isSorted = isSortedBuffer
    }

    create_check_sort_kernels(checkSortPartitionData) {
        if (!this.check_order) {
            return
        }

        const { check_sort_fast_count, check_sort_full_count, start_full } = checkSortPartitionData

        // Create the full pass
        const checkSortFull = new CheckSortKernel({
            mode: 'full',
            device: this.device,
            data: this.buffers.keys,
            result: this.buffers.dispatchSize,
            original: this.buffers.originalDispatchSize,
            is_sorted: this.buffers.isSorted,
            count: check_sort_full_count,
            start: start_full,
            workgroup_size: this.workgroup_size
        })

        // Create the fast pass
        const checkSortFast = new CheckSortKernel({
            mode: 'fast',
            device: this.device,
            data: this.buffers.keys,
            result: this.buffers.checkSortFullDispatchSize,
            original: this.buffers.originalCheckSortFullDispatchSize,
            is_sorted: this.buffers.isSorted,
            count: check_sort_fast_count,
            workgroup_size: this.workgroup_size
        })

        const initialDispatchElementCount = this.initialDispatch.length / 3

        if (checkSortFast.threads_per_workgroup < checkSortFull.pipelines.length || checkSortFull.threads_per_workgroup < initialDispatchElementCount) {
            console.warn(`Warning: workgroup size is too small to enable check sort optimization, disabling...`)
            this.check_order = false
            return
        }

        // Create the reset pass
        const checkSortReset = new CheckSortKernel({
            mode: 'reset',
            device: this.device,
            data: this.buffers.keys,
            original: this.buffers.originalDispatchSize,
            result: this.buffers.dispatchSize,
            is_sorted: this.buffers.isSorted,
            count: initialDispatchElementCount,
            workgroup_size: find_optimal_dispatch_size(this.device, initialDispatchElementCount)
        })

        this.kernels.checkSort = {
            reset: checkSortReset,
            fast: checkSortFast,
            full: checkSortFull,
        }
    }

    create_block_sum_pipeline(inKeys, inValues, bit) {
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'radix-sort-block-sum',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: this.local_shuffle ? 'storage' : 'read-only-storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                ...(this.local_shuffle && this.has_values ? [{
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                }] : [])
            ]
        })

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: inKeys }
                },
                {
                    binding: 1,
                    resource: { buffer: this.buffers.localPrefixSum }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.prefixBlockSum }
                },
                // "Local shuffle" optimization needs access to the values buffer
                ...(this.local_shuffle && this.has_values ? [{
                    binding: 3,
                    resource: { buffer: inValues }
                }] : [])
            ]
        })

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout ]
        })

        const blockSumPipeline = this.device.createComputePipeline({
            label: 'radix-sort-block-sum',
            layout: pipelineLayout,
            compute: {
                module: this.shaderModules.blockSum,
                entryPoint: 'radix_sort',
                constants: {
                    'WORKGROUP_SIZE_X': this.workgroup_size.x,
                    'WORKGROUP_SIZE_Y': this.workgroup_size.y,
                    'WORKGROUP_COUNT': this.workgroup_count,
                    'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
                    'ELEMENT_COUNT': this.count,
                    'CURRENT_BIT': bit,
                }
            }
        })

        return {
            pipeline: blockSumPipeline,
            bindGroup
        }
    }

    create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit) {
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'radix-sort-reorder',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                ...(this.has_values ? [
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' }
                    },
                    {
                        binding: 5,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    }
                ] : [])
            ]
        })

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: inKeys }
                },
                {
                    binding: 1,
                    resource: { buffer: outKeys }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.localPrefixSum }
                },
                {
                    binding: 3,
                    resource: { buffer: this.buffers.prefixBlockSum }
                },
                ...(this.has_values ? [
                    {
                        binding: 4,
                        resource: { buffer: inValues }
                    },
                    {
                        binding: 5,
                        resource: { buffer: outValues }
                    }
                ] : [])
            ]
        })

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout ]
        })

        const reorderPipeline = this.device.createComputePipeline({
            label: 'radix-sort-reorder',
            layout: pipelineLayout,
            compute: {
                module: this.shaderModules.reorder,
                entryPoint: 'radix_sort_reorder',
                constants: {
                    'WORKGROUP_SIZE_X': this.workgroup_size.x,
                    'WORKGROUP_SIZE_Y': this.workgroup_size.y,
                    'WORKGROUP_COUNT': this.workgroup_count,
                    'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
                    'ELEMENT_COUNT': this.count,
                    'CURRENT_BIT': bit,
                }
            }
        })

        return {
            pipeline: reorderPipeline,
            bindGroup
        }
    }

    /**
     * Encode all pipelines into the current pass
     * 
     * @param {GPUComputePassEncoder} pass 
     */
    dispatch(pass) {
        if (!this.check_order) {
            this.#dispatchPipelines(pass)
        }
        else {
            this.#dispatchPipelinesIndirect(pass)
        }
    }

    /**
     * Dispatch workgroups from CPU args
     */
    #dispatchPipelines(pass) {
        for (let i = 0; i < this.bit_count / 2; i++) {
            const { blockSumPipeline, reorderPipeline } = this.pipelines[i]
            
            // Compute local prefix sums and block sums
            pass.setPipeline(blockSumPipeline.pipeline)
            pass.setBindGroup(0, blockSumPipeline.bindGroup)
            pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1)

            // Compute block sums prefix sum
            this.kernels.prefixSum.dispatch(pass)

            // Reorder keys and values
            pass.setPipeline(reorderPipeline.pipeline)
            pass.setBindGroup(0, reorderPipeline.bindGroup)
            pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1)
        }
    }

    /**
     * Dispatch workgroups from indirect GPU buffers (used when check_order is enabled)
     */
    #dispatchPipelinesIndirect(pass) {
        // Reset the `dispatch` and `is_sorted` buffers
        this.kernels.checkSort.reset.dispatch(pass)
        
        for (let i = 0; i < this.bit_count / 2; i++) {
            const { blockSumPipeline, reorderPipeline } = this.pipelines[i]

            if (i % 2 == 0) {
                // Check if the data is sorted every 2 passes
                this.kernels.checkSort.fast.dispatch(pass, this.buffers.dispatchSize, this.dispatchOffsets.check_sort_fast)
                this.kernels.checkSort.full.dispatch(pass, this.buffers.checkSortFullDispatchSize)
            }
            
            // Compute local prefix sums and block sums
            pass.setPipeline(blockSumPipeline.pipeline)
            pass.setBindGroup(0, blockSumPipeline.bindGroup)
            pass.dispatchWorkgroupsIndirect(this.buffers.dispatchSize, this.dispatchOffsets.radix_sort)

            // Compute block sums prefix sum
            this.kernels.prefixSum.dispatch(pass, this.buffers.dispatchSize, this.dispatchOffsets.prefix_sum)

            // Reorder keys and values
            pass.setPipeline(reorderPipeline.pipeline)
            pass.setBindGroup(0, reorderPipeline.bindGroup)
            pass.dispatchWorkgroupsIndirect(this.buffers.dispatchSize, this.dispatchOffsets.radix_sort)
        }
    }
}

export default RadixSortKernel