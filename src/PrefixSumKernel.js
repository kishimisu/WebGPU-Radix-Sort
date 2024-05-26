import prefixSumSource from "./shaders/prefix_sum"
import prefixSumSource_NoBankConflict from "./shaders/optimizations/prefix_sum_no_bank_conflict"

class PrefixSumKernel {
    /**
     * Perform a parallel prefix sum on the given data buffer
     * 
     * Based on "Parallel Prefix Sum (Scan) with CUDA"
     * https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
     * 
     * @param {GPUDevice} device
     * @param {GPUBuffer} data - Buffer containing the data to process
     * @param {number} count - Max number of elements to process
     * @param {object} workgroup_size - Workgroup size in x and y dimensions. (x * y) must be a power of two
     * @param {boolean} avoid_bank_conflicts - Use the "Avoid bank conflicts" optimization from the original publication
     */
    constructor({
        device,
        data,
        count,
        workgroup_size = { x: 16, y: 16 },
        avoid_bank_conflicts = false
    }) {
        this.device = device
        this.workgroup_size = workgroup_size
        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y
        this.items_per_workgroup = 2 * this.threads_per_workgroup // 2 items are processed per thread

        if (Math.log2(this.threads_per_workgroup) % 1 !== 0) 
            throw new Error(`workgroup_size.x * workgroup_size.y must be a power of two. (current: ${this.threads_per_workgroup})`)

        this.pipelines = []

        this.shaderModule = this.device.createShaderModule({
            label: 'prefix-sum',
            code: avoid_bank_conflicts ? prefixSumSource_NoBankConflict : prefixSumSource,
        })

        this.create_pass_recursive(data, count)
    }

    create_pass_recursive(data, count) {
        // Numbers of workgroups needed to process all items
        const block_count = Math.ceil(count / this.items_per_workgroup)

        // Create buffer for block sums
        const blockSumBuffer = this.device.createBuffer({
            size: block_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        // Create bind group and pipeline layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                }
            ]
        })

        const bindGroup = this.device.createBindGroup({
            label: 'prefix-sum-bind-group',
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: data }
                },
                {
                    binding: 1,
                    resource: { buffer: blockSumBuffer }
                }
            ]
        })

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout ]
        })

        // Per-workgroup (block) prefix sum
        const scanPipeline = this.device.createComputePipeline({
            label: 'prefix-sum-scan-pipeline',
            layout: pipelineLayout,
            compute: {
                module: this.shaderModule,
                entryPoint: 'reduce_downsweep',
                constants: {
                    'WORKGROUP_SIZE_X': this.workgroup_size.x,
                    'WORKGROUP_SIZE_Y': this.workgroup_size.y,
                    'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
                    'ITEMS_PER_WORKGROUP': this.items_per_workgroup
                }
            }
        })

        this.pipelines.push({ pipeline: scanPipeline, bindGroup, block_count })

        if (block_count > 1) {
            // Prefix sum on block sums
            this.create_pass_recursive(blockSumBuffer, block_count)

            // Add block sums to local prefix sums
            const blockSumPipeline = this.device.createComputePipeline({
                label: 'prefix-sum-add-block-pipeline',
                layout: pipelineLayout,
                compute: {
                    module: this.shaderModule,
                    entryPoint: 'add_block_sums',
                    constants: {
                        'WORKGROUP_SIZE_X': this.workgroup_size.x,
                        'WORKGROUP_SIZE_Y': this.workgroup_size.y,
                        'THREADS_PER_WORKGROUP': this.threads_per_workgroup
                    }
                }
            })

            this.pipelines.push({ pipeline: blockSumPipeline, bindGroup, block_count })
        }
    }

    dispatch(pass) {
        for (const { pipeline, bindGroup, block_count } of this.pipelines) {
            pass.setPipeline(pipeline)
            pass.setBindGroup(0, bindGroup)
            pass.dispatchWorkgroups(block_count)
        }
    }
}

export default PrefixSumKernel