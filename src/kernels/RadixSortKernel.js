import PrefixSumKernel from "./PrefixSumKernel.js"
import radixSortSource from "../shaders/radix_sort.js"
import radixSortSource_LocalShuffle from "../shaders/optimizations/radix_sort_local_shuffle.js"
import reorderSource from "../shaders/radix_sort_reorder.js"

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
        local_shuffle = false,
        avoid_bank_conflicts = false,
    }) {
        this.device = device
        this.count = count
        this.bit_count = bit_count
        this.workgroup_size = workgroup_size
        this.local_shuffle = local_shuffle
        this.avoid_bank_conflicts = avoid_bank_conflicts

        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y
        this.workgroup_count = Math.ceil(count / this.threads_per_workgroup)
        this.prefix_block_workgroup_count = 4 * this.workgroup_count

        this.has_values = (values != null)

        this.shaderModules = {}
        this.buffers = {}
        this.pipelines = []

        // Create shader modules from wgsl code
        this.create_shader_modules()

        // Create GPU buffers
        this.create_buffers(keys, values)
        
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

    create_buffers(keys, values) {
        // Keys and values double buffering
        const tmpKeysBuffer = this.device.createBuffer({
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })
        const tmpValuesBuffer = !this.has_values ? null : this.device.createBuffer({
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        // Local Prefix Sum buffer (1 element per item)
        const localPrefixSumBuffer = this.device.createBuffer({
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        // Prefix Block Sum buffer (4 element per workgroup)
        const prefixBlockSumBuffer = this.device.createBuffer({
            size: this.prefix_block_workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })
        
        this.buffers = {
            keys: keys,
            values: values,
            tmpKeys: tmpKeysBuffer,
            tmpValues: tmpValuesBuffer,
            localPrefixSum: localPrefixSumBuffer,
            prefixBlockSum: prefixBlockSumBuffer,
        }
    }

    // Create radix sort passes for every 2 bits
    create_pipelines() {
        for (let bit = 0; bit < this.bit_count; bit += 2) {
            // Swap buffers every pass
            const even      = (bit % 4 == 0)
            const inKeys    = even ? this.buffers.keys : this.buffers.tmpKeys
            const inValues  = even ? this.buffers.values : this.buffers.tmpValues
            const outKeys   = even ? this.buffers.tmpKeys : this.buffers.keys
            const outValues = even ? this.buffers.tmpValues : this.buffers.values

            // Compute local prefix sums and block sums
            const blockSumPipeline = this.create_block_sum_pipeline(inKeys, inValues, bit)

            // Compute block sums prefix sums
            const prefixSumKernel = new PrefixSumKernel({ 
                device: this.device,
                data: this.buffers.prefixBlockSum, 
                count: this.prefix_block_workgroup_count,
                workgroup_size: this.workgroup_size,
                avoid_bank_conflicts: this.avoid_bank_conflicts,
            })
            
            // Reorder keys and values
            const reorderPipeline = this.create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit)

            this.pipelines.push({ blockSumPipeline, prefixSumKernel, reorderPipeline })
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
        for (const { blockSumPipeline, prefixSumKernel, reorderPipeline } of this.pipelines) {            
            pass.setPipeline(blockSumPipeline.pipeline)
            pass.setBindGroup(0, blockSumPipeline.bindGroup)
            pass.dispatchWorkgroups(this.workgroup_count, 1, 1)

            prefixSumKernel.dispatch(pass)

            pass.setPipeline(reorderPipeline.pipeline)
            pass.setBindGroup(0, reorderPipeline.bindGroup)
            pass.dispatchWorkgroups(this.workgroup_count, 1, 1)
        }
    }
}

export default RadixSortKernel