'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

const prefixSumSource = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    temp[ELM_TID]     = select(items[ELM_GID], 0, ELM_GID >= ELEMENT_COUNT);
    temp[ELM_TID + 1] = select(items[ELM_GID + 1], 0, ELM_GID + 1 >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        let last_offset = ITEMS_PER_WORKGROUP - 1;

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (ELM_GID >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID] = temp[ELM_TID];

    if (ELM_GID + 1 >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID + 1] = temp[ELM_TID + 1];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}`;

/**
 * Prefix sum with optimization to avoid bank conflicts
 * 
 * (see Implementation section in README for details)
 */
const prefixSumNoBankConflictSource = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

const NUM_BANKS: u32 = 32;
const LOG_NUM_BANKS: u32 = 5;

fn get_offset(offset: u32) -> u32 {
    // return offset >> LOG_NUM_BANKS; // Conflict-free
    return (offset >> NUM_BANKS) + (offset >> (2 * LOG_NUM_BANKS)); // Zero bank conflict
}

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    let ai: u32 = TID;
    let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1);
    let s_ai = ai + get_offset(ai);
    let s_bi = bi + get_offset(bi);
    let g_ai = ai + WID * 2;
    let g_bi = bi + WID * 2;
    temp[s_ai] = select(items[g_ai], 0, g_ai >= ELEMENT_COUNT);
    temp[s_bi] = select(items[g_bi], 0, g_bi >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            ai += get_offset(ai);
            bi += get_offset(bi);
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        var last_offset = ITEMS_PER_WORKGROUP - 1;
        last_offset += get_offset(last_offset);

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            ai += get_offset(ai);
            bi += get_offset(bi);

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (g_ai < ELEMENT_COUNT) {
        items[g_ai] = temp[s_ai];
    }
    if (g_bi < ELEMENT_COUNT) {
        items[g_bi] = temp[s_bi];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}`;

/**
 * Find the best dispatch size x and y dimensions to minimize unused workgroups
 * 
 * @param {GPUDevice} device - The GPU device
 * @param {int} workgroup_count - Number of workgroups to dispatch
 * @returns 
 */
function find_optimal_dispatch_size(device, workgroup_count) {
    const dispatchSize = { 
        x: workgroup_count, 
        y: 1
    };

    if (workgroup_count > device.limits.maxComputeWorkgroupsPerDimension) {
        const x = Math.floor(Math.sqrt(workgroup_count));
        const y = Math.ceil(workgroup_count / x);
        
        dispatchSize.x = x;
        dispatchSize.y = y;
    }

    return dispatchSize
}

function create_buffer_from_data({device, label, data, usage = 0}) {
    const dispatchSizes = device.createBuffer({
        label: label,
        usage: usage,
        size: data.length * 4,
        mappedAtCreation: true
    });

    const dispatchData = new Uint32Array(dispatchSizes.getMappedRange());
    dispatchData.set(data);
    dispatchSizes.unmap();

    return dispatchSizes
}

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
        this.device = device;
        this.workgroup_size = workgroup_size;
        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;
        this.items_per_workgroup = 2 * this.threads_per_workgroup; // 2 items are processed per thread

        if (Math.log2(this.threads_per_workgroup) % 1 !== 0) 
            throw new Error(`workgroup_size.x * workgroup_size.y must be a power of two. (current: ${this.threads_per_workgroup})`)

        this.pipelines = [];

        this.shaderModule = this.device.createShaderModule({
            label: 'prefix-sum',
            code: avoid_bank_conflicts ? prefixSumNoBankConflictSource : prefixSumSource,
        });

        this.create_pass_recursive(data, count);
    }

    create_pass_recursive(data, count) {
        // Find best dispatch x and y dimensions to minimize unused threads
        const workgroup_count = Math.ceil(count / this.items_per_workgroup);
        const dispatchSize = find_optimal_dispatch_size(this.device, workgroup_count);
        
        // Create buffer for block sums        
        const blockSumBuffer = this.device.createBuffer({
            label: 'prefix-sum-block-sum',
            size: workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

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
        });

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
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout ]
        });

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
                    'ITEMS_PER_WORKGROUP': this.items_per_workgroup,
                    'ELEMENT_COUNT': count,
                }
            }
        });

        this.pipelines.push({ pipeline: scanPipeline, bindGroup, dispatchSize });

        if (workgroup_count > 1) {
            // Prefix sum on block sums
            this.create_pass_recursive(blockSumBuffer, workgroup_count);

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
                        'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
                        'ELEMENT_COUNT': count,
                    }
                }
            });

            this.pipelines.push({ pipeline: blockSumPipeline, bindGroup, dispatchSize });
        }
    }

    get_dispatch_chain() {
        return this.pipelines.flatMap(p => [ p.dispatchSize.x, p.dispatchSize.y, 1 ])
    }

    /**
     * Encode the prefix sum pipeline into the current pass.
     * If dispatchSizeBuffer is provided, the dispatch will be indirect (dispatchWorkgroupsIndirect)
     * 
     * @param {GPUComputePassEncoder} pass 
     * @param {GPUBuffer} dispatchSizeBuffer - (optional) Indirect dispatch buffer
     * @param {int} offset - (optional) Offset in bytes in the dispatch buffer. Default: 0
     */
    dispatch(pass, dispatchSizeBuffer, offset = 0) {
        for (let i = 0; i < this.pipelines.length; i++) {
            const { pipeline, bindGroup, dispatchSize } = this.pipelines[i];
            
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);

            if (dispatchSizeBuffer == null)
                pass.dispatchWorkgroups(dispatchSize.x, dispatchSize.y, 1);
            else
                pass.dispatchWorkgroupsIndirect(dispatchSizeBuffer, offset + i * 3 * 4);
        }
    }
}

const radixSortSource = /* wgsl */ `

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    // Extract 2 bits from the input
    let elm = select(input[GID], 0, GID >= ELEMENT_COUNT);
    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;

    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

    // If the workgroup is inactive, prevent block_sums buffer update
    var LAST_THREAD: u32 = 0xffffffff; 

    if (WORKGROUP_ID < WORKGROUP_COUNT) {
        // Otherwise store the index of the last active thread in the workgroup
        LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;
    }

    // Initialize parameters for double-buffering
    let TPW = THREADS_PER_WORKGROUP + 1;
    var swapOffset: u32 = 0;
    var inOffset:  u32 = TID;
    var outOffset: u32 = TID + TPW;

    // 4-way prefix sum
    for (var b: u32 = 0; b < 4; b++) {
        // Initialize local prefix with bitmask
        let bitmask = select(0u, 1u, extract_bits == b);
        s_prefix_sum[inOffset + 1] = bitmask;
        workgroupBarrier();

        var prefix_sum: u32 = 0;

        // Prefix sum
        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {
            if (TID >= offset) {
                prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
            } else {
                prefix_sum = s_prefix_sum[inOffset];
            }

            s_prefix_sum[outOffset] = prefix_sum;
            
            // Swap buffers
            outOffset = inOffset;
            swapOffset = TPW - swapOffset;
            inOffset = TID + swapOffset;
            
            workgroupBarrier();
        }

        // Store prefix sum for current bit
        bit_prefix_sums[b] = prefix_sum;

        if (TID == LAST_THREAD) {
            // Store block sum to global memory
            let total_sum: u32 = prefix_sum + bitmask;
            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;
        }

        // Swap buffers
        outOffset = inOffset;
        swapOffset = TPW - swapOffset;
        inOffset = TID + swapOffset;
    }

    if (GID < ELEMENT_COUNT) {
        // Store local prefix sum to global memory
        local_prefix_sums[GID] = bit_prefix_sums[extract_bits];
    }
}`;

/**
 * Radix sort with "local shuffle and coalesced mapping" optimization
 * 
 * (see Implementation section in README for details)
 */
const radixSortCoalescedSource = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;
var<workgroup> s_prefix_sum_scan: array<u32, 4>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    // Extract 2 bits from the input
    var elm: u32 = 0;
    var val: u32 = 0;
    if (GID < ELEMENT_COUNT) {
        elm = input[GID];
        val = values[GID];
    }
    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;

    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

    // If the workgroup is inactive, prevent block_sums buffer update
    var LAST_THREAD: u32 = 0xffffffff; 

    if (WORKGROUP_ID < WORKGROUP_COUNT) {
        // Otherwise store the index of the last active thread in the workgroup
        LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;
    }

    // Initialize parameters for double-buffering
    let TPW = THREADS_PER_WORKGROUP + 1;
    var swapOffset: u32 = 0;
    var inOffset:  u32 = TID;
    var outOffset: u32 = TID + TPW;

    // 4-way prefix sum
    for (var b: u32 = 0; b < 4; b++) {
        // Initialize local prefix with bitmask
        let bitmask = select(0u, 1u, extract_bits == b);
        s_prefix_sum[inOffset + 1] = bitmask;
        workgroupBarrier();

        var prefix_sum: u32 = 0;

        // Prefix sum
        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {
            if (TID >= offset) {
                prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
            } else {
                prefix_sum = s_prefix_sum[inOffset];
            }

            s_prefix_sum[outOffset] = prefix_sum;

            // Swap buffers
            outOffset = inOffset;
            swapOffset = TPW - swapOffset;
            inOffset = TID + swapOffset;
            
            workgroupBarrier();
        }

        // Store prefix sum for current bit
        bit_prefix_sums[b] = prefix_sum;

        if (TID == LAST_THREAD) {
            // Store block sum to global memory
            let total_sum: u32 = prefix_sum + bitmask;
            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;
        }

        // Swap buffers
        outOffset = inOffset;
        swapOffset = TPW - swapOffset;
        inOffset = TID + swapOffset;
    }

    let prefix_sum = bit_prefix_sums[extract_bits];   

    // Scan bit prefix sums
    if (TID == LAST_THREAD) {
        var sum: u32 = 0;
        bit_prefix_sums[extract_bits] += 1;
        for (var i: u32 = 0; i < 4; i++) {
            s_prefix_sum_scan[i] = sum;
            sum += bit_prefix_sums[i];
        }
    }
    workgroupBarrier();

    if (GID < ELEMENT_COUNT) {
        // Compute new position
        let new_pos: u32 = prefix_sum + s_prefix_sum_scan[extract_bits];

        // Shuffle elements locally
        input[WID + new_pos] = elm;
        values[WID + new_pos] = val;
        local_prefix_sums[WID + new_pos] = prefix_sum;
    }
}`;

const radixSortReorderSource = /* wgsl */ `

@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;
@group(0) @binding(4) var<storage, read> inputValues: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputValues: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort_reorder(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) { 
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    if (GID >= ELEMENT_COUNT) {
        return;
    }

    let k = inputKeys[GID];
    let v = inputValues[GID];

    let local_prefix = local_prefix_sum[GID];

    // Calculate new position
    let extract_bits = (k >> CURRENT_BIT) & 0x3;
    let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;
    let sorted_position = prefix_block_sum[pid] + local_prefix;
    
    outputKeys[sorted_position] = k;
    outputValues[sorted_position] = v;
}`;

const checkSortSource = (isFirstPass = false, isLastPass = false, kernelMode = 'full') => /* wgsl */ `

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> original: array<u32>;
@group(0) @binding(3) var<storage, read_write> is_sorted: u32;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;
override START_ELEMENT: u32;

var<workgroup> s_data: array<u32, THREADS_PER_WORKGROUP>;

// Reset dispatch buffer and is_sorted flag
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reset(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    if (TID >= ELEMENT_COUNT) {
        return;
    }

    if (TID == 0) {
        is_sorted = 0u;
    }

    let ELM_ID = TID * 3;

    output[ELM_ID] = original[ELM_ID];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn check_sort(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP + START_ELEMENT;
    let GID = TID + WID; // Global thread ID

    // Load data into shared memory
    ${ isFirstPass ? first_pass_load_data : "s_data[TID] = select(0u, input[GID], GID < ELEMENT_COUNT);" }

    // Perform parallel reduction
    for (var d = 1u; d < THREADS_PER_WORKGROUP; d *= 2u) {      
        workgroupBarrier();  
        if (TID % (2u * d) == 0u) {
            s_data[TID] += s_data[TID + d];
        }
    }
    workgroupBarrier();

    // Write reduction result
    ${ isLastPass ? last_pass(kernelMode) : write_reduction_result }
}`;

const write_reduction_result = /* wgsl */ `
    if (TID == 0) {
        output[WORKGROUP_ID] = s_data[0];
    }
`;

const first_pass_load_data = /* wgsl */ `
    let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

    // Load current element into shared memory
    // Also load next element for comparison
    let elm = select(0u, input[GID], GID < ELEMENT_COUNT);
    let next = select(0u, input[GID + 1], GID < ELEMENT_COUNT-1);
    s_data[TID] = elm;
    workgroupBarrier();

    s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);
`;

const last_pass = (kernelMode) => /* wgsl */ `
    let fullDispatchLength = arrayLength(&output);
    let dispatchIndex = TID * 3;

    if (dispatchIndex >= fullDispatchLength) {
        return;
    }

    ${kernelMode == 'full' ? last_pass_full : last_pass_fast}
`;

// If the fast check kernel is sorted and the data isn't already sorted, run the full check
const last_pass_fast = /* wgsl */ `
    output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u);
`;

// If the full check kernel is sorted, set the flag to 1 and skip radix sort passes
const last_pass_full = /* wgsl */ `
    if (TID == 0 && s_data[0] == 0) {
        is_sorted = 1u;
    }

    output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] != 0);
`;

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
        this.device = device;
        this.count = count;
        this.start = start;
        this.mode = mode;
        this.workgroup_size = workgroup_size;
        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;

        this.pipelines = [];

        this.buffers = {
            data, 
            result, 
            original, 
            is_sorted,
            outputs: []
        };

        this.create_passes_recursive(data, count);
    }

    // Find the best dispatch size for each pass to minimize unused workgroups
    static find_optimal_dispatch_chain(device, item_count, workgroup_size) {
        const threads_per_workgroup = workgroup_size.x * workgroup_size.y;
        const sizes = [];

        do {
            // Number of workgroups required to process all items
            const target_workgroup_count = Math.ceil(item_count / threads_per_workgroup);
    
            // Optimal dispatch size and updated workgroup count
            const dispatchSize = find_optimal_dispatch_size(device, target_workgroup_count);
    
            sizes.push(dispatchSize.x, dispatchSize.y, 1);
            item_count = target_workgroup_count;
        } while (item_count > 1)
    
        return sizes
    }

    create_passes_recursive(buffer, count, passIndex = 0) {
        const workgroup_count = Math.ceil(count / this.threads_per_workgroup);

        const isFirstPass = passIndex === 0;
        const isLastPass = workgroup_count <= 1;

        const label = `check-sort-${this.mode}-${passIndex}`;

        const outputBuffer = isLastPass ? this.buffers.result : this.device.createBuffer({
            label: label,
            size: workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

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
        });

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
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        const element_count = isFirstPass ? this.start + count : count;
        const start_element = isFirstPass ? this.start : 0;

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
        });

        this.buffers.outputs.push(outputBuffer);
        this.pipelines.push({ pipeline: checkSortPipeline, bindGroup });
        
        if (!isLastPass) {
            this.create_passes_recursive(outputBuffer, workgroup_count, passIndex + 1);
        }
    }

    dispatch(pass, dispatchSize, offset = 0) {
        for (let i = 0; i < this.pipelines.length; i++) {
            const { pipeline, bindGroup } = this.pipelines[i];

            const dispatchIndirect = this.mode != 'reset' && (this.mode == 'full' || i < this.pipelines.length - 1);

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);

            if (dispatchIndirect)
                pass.dispatchWorkgroupsIndirect(dispatchSize, offset + i * 3 * 4);
            else
                // Only the reset kernel and the last dispatch of the fast check kernel are constant to (1, 1, 1)
                pass.dispatchWorkgroups(1, 1, 1);
        }
    }
}

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

        this.device = device;
        this.count = count;
        this.bit_count = bit_count;
        this.workgroup_size = workgroup_size;
        this.check_order = check_order;
        this.local_shuffle = local_shuffle;
        this.avoid_bank_conflicts = avoid_bank_conflicts;

        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;
        this.workgroup_count = Math.ceil(count / this.threads_per_workgroup);
        this.prefix_block_workgroup_count = 4 * this.workgroup_count;

        this.has_values = (values != null); // Is the values buffer provided ?

        this.dispatchSize = {};  // Dispatch dimension x and y
        this.shaderModules = {}; // GPUShaderModules
        this.kernels = {};       // PrefixSumKernel & CheckSortKernels
        this.pipelines = [];     // List of passes
        this.buffers = {        // GPUBuffers
            keys: keys,
            values: values
        };       

        // Create shader modules from wgsl code
        this.create_shader_modules();
        
        // Create multi-pass pipelines
        this.create_pipelines();
    }

    create_shader_modules() {
        // Remove every occurence of "values" in the shader code if values buffer is not provided
        const remove_values = (source) => {
            return source.split('\n')
                         .filter(line => !line.toLowerCase().includes('values'))
                         .join('\n')
        };

        const blockSumSource = this.local_shuffle ? radixSortCoalescedSource : radixSortSource;
        
        this.shaderModules = {
            blockSum: this.device.createShaderModule({
                label: 'radix-sort-block-sum',
                code: this.has_values ? blockSumSource : remove_values(blockSumSource),
            }),
            reorder: this.device.createShaderModule({
                label: 'radix-sort-reorder',
                code: this.has_values ? radixSortReorderSource : remove_values(radixSortReorderSource),
            })
        };
    }

    create_pipelines() {    
        // Block prefix sum kernel    
        this.create_prefix_sum_kernel();

        // Indirect dispatch buffers
        const dispatchData = this.calculate_dispatch_sizes();

        // GPU buffers
        this.create_buffers(dispatchData);

        // Check sort kernels
        this.create_check_sort_kernels(dispatchData);

        // Radix sort passes for every 2 bits
        for (let bit = 0; bit < this.bit_count; bit += 2) {
            // Swap buffers every pass
            const even      = (bit % 4 == 0);
            const inKeys    = even ? this.buffers.keys : this.buffers.tmpKeys;
            const inValues  = even ? this.buffers.values : this.buffers.tmpValues;
            const outKeys   = even ? this.buffers.tmpKeys : this.buffers.keys;
            const outValues = even ? this.buffers.tmpValues : this.buffers.values;

            // Compute local prefix sums and block sums
            const blockSumPipeline = this.create_block_sum_pipeline(inKeys, inValues, bit);
            
            // Reorder keys and values
            const reorderPipeline = this.create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit);

            this.pipelines.push({ blockSumPipeline, reorderPipeline });
        }
    }

    create_prefix_sum_kernel() {
        // Prefix Block Sum buffer (4 element per workgroup)
        const prefixBlockSumBuffer = this.device.createBuffer({
            label: 'radix-sort-prefix-block-sum',
            size: this.prefix_block_workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Create block prefix sum kernel
        const prefixSumKernel = new PrefixSumKernel({ 
            device: this.device,
            data: prefixBlockSumBuffer, 
            count: this.prefix_block_workgroup_count,
            workgroup_size: this.workgroup_size,
            avoid_bank_conflicts: this.avoid_bank_conflicts,
        });

        this.kernels.prefixSum = prefixSumKernel;
        this.buffers.prefixBlockSum = prefixBlockSumBuffer;
    }

    calculate_dispatch_sizes() {
        // Radix sort dispatch size
        const dispatchSize = find_optimal_dispatch_size(this.device, this.workgroup_count);

        // Prefix sum dispatch sizes
        const prefixSumDispatchSize = this.kernels.prefixSum.get_dispatch_chain();

        // Check sort element count (fast/full)
        const check_sort_fast_count = Math.min(this.count, this.threads_per_workgroup * 4);
        const check_sort_full_count = this.count - check_sort_fast_count;
        const start_full = check_sort_fast_count - 1;

        // Check sort dispatch sizes
        const dispatchSizesFast = CheckSortKernel.find_optimal_dispatch_chain(this.device, check_sort_fast_count, this.workgroup_size);
        const dispatchSizesFull = CheckSortKernel.find_optimal_dispatch_chain(this.device, check_sort_full_count, this.workgroup_size);

        // Initial dispatch sizes
        const initialDispatch = [
            dispatchSize.x, dispatchSize.y, 1, // Radix Sort + Reorder
            ...dispatchSizesFast.slice(0, 3),  // Check sort fast
            ...prefixSumDispatchSize           // Prefix Sum
        ];

        // Dispatch offsets in main buffer
        this.dispatchOffsets = {
            radix_sort: 0,
            check_sort_fast: 3 * 4,
            prefix_sum: 6 * 4
        };

        this.dispatchSize = dispatchSize;
        this.initialDispatch = initialDispatch;

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
        });
        const tmpValuesBuffer = !this.has_values ? null : this.device.createBuffer({
            label: 'radix-sort-tmp-values',
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Local Prefix Sum buffer (1 element per item)
        const localPrefixSumBuffer = this.device.createBuffer({
            label: 'radix-sort-local-prefix-sum',
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.buffers.tmpKeys = tmpKeysBuffer;
        this.buffers.tmpValues = tmpValuesBuffer;
        this.buffers.localPrefixSum = localPrefixSumBuffer;

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
        });
        const originalDispatchBuffer = create_buffer_from_data({
            device: this.device, 
            label: 'radix-sort-dispatch-size-original',
            data: dispatchData.initialDispatch, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Dispatch sizes (full sort)
        const checkSortFullDispatchBuffer = create_buffer_from_data({
            label: 'check-sort-full-dispatch-size',
            device: this.device, 
            data: dispatchData.dispatchSizesFull,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT
        });
        const checkSortFullOriginalDispatchBuffer = create_buffer_from_data({
            label: 'check-sort-full-dispatch-size-original',
            device: this.device, 
            data: dispatchData.dispatchSizesFull,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Flag to tell if the data is sorted
        const isSortedBuffer = create_buffer_from_data({
            label: 'is-sorted',
            device: this.device, 
            data: new Uint32Array([0]), 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.buffers.dispatchSize = dispatchBuffer;
        this.buffers.originalDispatchSize = originalDispatchBuffer;
        this.buffers.checkSortFullDispatchSize = checkSortFullDispatchBuffer;
        this.buffers.originalCheckSortFullDispatchSize = checkSortFullOriginalDispatchBuffer;
        this.buffers.isSorted = isSortedBuffer;
    }

    create_check_sort_kernels(checkSortPartitionData) {
        if (!this.check_order) {
            return
        }

        const { check_sort_fast_count, check_sort_full_count, start_full } = checkSortPartitionData;

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
        });

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
        });

        const initialDispatchElementCount = this.initialDispatch.length / 3;

        if (checkSortFast.threads_per_workgroup < checkSortFull.pipelines.length || checkSortFull.threads_per_workgroup < initialDispatchElementCount) {
            console.warn(`Warning: workgroup size is too small to enable check sort optimization, disabling...`);
            this.check_order = false;
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
        });

        this.kernels.checkSort = {
            reset: checkSortReset,
            fast: checkSortFast,
            full: checkSortFull,
        };
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
        });

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
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout ]
        });

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
        });

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
        });

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
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout ]
        });

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
        });

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
            this.#dispatchPipelines(pass);
        }
        else {
            this.#dispatchPipelinesIndirect(pass);
        }
    }

    /**
     * Dispatch workgroups from CPU args
     */
    #dispatchPipelines(pass) {
        for (let i = 0; i < this.bit_count / 2; i++) {
            const { blockSumPipeline, reorderPipeline } = this.pipelines[i];
            
            // Compute local prefix sums and block sums
            pass.setPipeline(blockSumPipeline.pipeline);
            pass.setBindGroup(0, blockSumPipeline.bindGroup);
            pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1);

            // Compute block sums prefix sum
            this.kernels.prefixSum.dispatch(pass);

            // Reorder keys and values
            pass.setPipeline(reorderPipeline.pipeline);
            pass.setBindGroup(0, reorderPipeline.bindGroup);
            pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1);
        }
    }

    /**
     * Dispatch workgroups from indirect GPU buffers (used when check_order is enabled)
     */
    #dispatchPipelinesIndirect(pass) {
        // Reset the `dispatch` and `is_sorted` buffers
        this.kernels.checkSort.reset.dispatch(pass);
        
        for (let i = 0; i < this.bit_count / 2; i++) {
            const { blockSumPipeline, reorderPipeline } = this.pipelines[i];

            if (i % 2 == 0) {
                // Check if the data is sorted every 2 passes
                this.kernels.checkSort.fast.dispatch(pass, this.buffers.dispatchSize, this.dispatchOffsets.check_sort_fast);
                this.kernels.checkSort.full.dispatch(pass, this.buffers.checkSortFullDispatchSize);
            }
            
            // Compute local prefix sums and block sums
            pass.setPipeline(blockSumPipeline.pipeline);
            pass.setBindGroup(0, blockSumPipeline.bindGroup);
            pass.dispatchWorkgroupsIndirect(this.buffers.dispatchSize, this.dispatchOffsets.radix_sort);

            // Compute block sums prefix sum
            this.kernels.prefixSum.dispatch(pass, this.buffers.dispatchSize, this.dispatchOffsets.prefix_sum);

            // Reorder keys and values
            pass.setPipeline(reorderPipeline.pipeline);
            pass.setBindGroup(0, reorderPipeline.bindGroup);
            pass.dispatchWorkgroupsIndirect(this.buffers.dispatchSize, this.dispatchOffsets.radix_sort);
        }
    }
}

exports.PrefixSumKernel = PrefixSumKernel;
exports.RadixSortKernel = RadixSortKernel;
//# sourceMappingURL=radix-sort-cjs.js.map
