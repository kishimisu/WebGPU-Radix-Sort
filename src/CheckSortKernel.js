import checkSortSource from "./shaders/check_sort"
import { find_optimal_dispatch_size } from "./utils"

class CheckSortKernel {
    /**
     * CheckSortKernel - Performs a parralel reduction to check if an array is sorted.
     * 
     * @param {GPUDevice} device
     * @param {GPUBuffer} data - The buffer containing the data to check
     * @param {GPUBuffer} result - The result dispatch size buffer
     * @param {GPUBuffer} original - The original dispatch size buffer
     * @param {GPUBuffer} is_sorted - 1-element buffer to store whether the array is sorted
     * @param {number} count - The number of elements to check
     * @param {number} start - The index to start checking from
     * @param {boolean} mode - The type of check sort kernel ('reset', 'fast', 'full')
     * @param {object} workgroup_size - The workgroup size in x and y dimensions
     */
    constructor({
        device,
        data,
        result,
        original,
        is_sorted,
        count,
        start = 0,
        mode = 'full',
        workgroup_size = { x: 16, y: 16 },
    }) {
        this.device = device
        this.count = count
        this.start = start
        this.mode = mode
        this.workgroup_size = workgroup_size
        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y

        this.pipelines = []

        this.buffers = {
            data, 
            result, 
            original, 
            is_sorted,
            outputs: []
        }

        this.create_passes_recursive(data, count)
    }

    // Find the best dispatch size for each pass to minimize unused workgroups
    static find_optimal_dispatch_chain(device, item_count, workgroup_size) {
        const threads_per_workgroup = workgroup_size.x * workgroup_size.y
        const sizes = []

        do {
            // Number of workgroups required to process all items
            const target_workgroup_count = Math.ceil(item_count / threads_per_workgroup)
    
            // Optimal dispatch size and updated workgroup count
            const dispatchSize = find_optimal_dispatch_size(device, target_workgroup_count)
    
            sizes.push(dispatchSize.x, dispatchSize.y, 1)
            item_count = target_workgroup_count
        } while (item_count > 1)
    
        return sizes
    }

    create_passes_recursive(buffer, count, passIndex = 0) {
        const workgroup_count = Math.ceil(count / this.threads_per_workgroup)

        const isFirstPass = passIndex === 0
        const isLastPass = workgroup_count <= 1

        const label = `check-sort-${this.mode}-${passIndex}`

        const outputBuffer = isLastPass ? this.buffers.result : this.device.createBuffer({
            label: label,
            size: workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        const bindGroupLayout = this.device.createBindGroupLayout({
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
                // Last pass bindings
                ...(isLastPass ? [{
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                }, {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                }] : []),
            ]
        })

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: buffer }
                },
                {
                    binding: 1,
                    resource: { buffer: outputBuffer }
                },
                // Last pass buffers
                ...(isLastPass ? [{
                    binding: 2,
                    resource: { buffer: this.buffers.original }
                }, {
                    binding: 3,
                    resource: { buffer: this.buffers.is_sorted }
                }] : []),
            ]
        })

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        })

        const element_count = isFirstPass ? this.start + count : count
        const start_element = isFirstPass ? this.start : 0

        const checkSortPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: this.device.createShaderModule({
                    label: label,
                    code: checkSortSource(isFirstPass, isLastPass, this.mode),
                }),
                entryPoint: this.mode == 'reset' ? 'reset' : 'check_sort',
                constants: {
                    'ELEMENT_COUNT': element_count,
                    'WORKGROUP_SIZE_X': this.workgroup_size.x,
                    'WORKGROUP_SIZE_Y': this.workgroup_size.y,
                    ...(this.mode != 'reset' && { 
                        'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
                        'START_ELEMENT': start_element,
                    })
                },
            }
        })

        this.buffers.outputs.push(outputBuffer)
        this.pipelines.push({ pipeline: checkSortPipeline, bindGroup })
        
        if (!isLastPass) {
            this.create_passes_recursive(outputBuffer, workgroup_count, passIndex + 1)
        }
    }

    dispatch(pass, dispatchSize, offset = 0) {
        for (let i = 0; i < this.pipelines.length; i++) {
            const { pipeline, bindGroup } = this.pipelines[i]

            const dispatchIndirect = this.mode != 'reset' && (this.mode == 'full' || i < this.pipelines.length - 1)

            pass.setPipeline(pipeline)
            pass.setBindGroup(0, bindGroup)

            if (dispatchIndirect)
                pass.dispatchWorkgroupsIndirect(dispatchSize, offset + i * 3 * 4)
            else
                // Only the reset kernel and the last dispatch of the fast check kernel are constant to (1, 1, 1)
                pass.dispatchWorkgroups(1, 1, 1)
        }
    }
}

export default CheckSortKernel