'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

const prefixSumSource = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WID = wid.x * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    temp[ELM_TID]     = items[ELM_GID];
    temp[ELM_TID + 1] = items[ELM_GID + 1];

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

        blockSums[wid.x] = temp[last_offset];
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
    items[ELM_GID]     = temp[ELM_TID];
    items[ELM_GID + 1] = temp[ELM_TID + 1];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let GID = wid.x * THREADS_PER_WORKGROUP + TID; // Global thread ID

    let ELM_ID = GID * 2;
    let blockSum = blockSums[wid.x];

    items[ELM_ID] += blockSum;
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

const NUM_BANKS: u32 = 32;
const LOG_NUM_BANKS: u32 = 5;

fn get_offset(offset: u32) -> u32 {
    // return offset >> LOG_NUM_BANKS; // Conflict-free
    return (offset >> NUM_BANKS) + (offset >> (2 * LOG_NUM_BANKS)); // Zero bank conflict
}

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WID = wid.x * THREADS_PER_WORKGROUP;
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
    temp[s_ai] = items[g_ai];
    temp[s_bi] = items[g_bi];

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

        blockSums[wid.x] = temp[last_offset];
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
    items[g_ai] = temp[s_ai];
    items[g_bi] = temp[s_bi];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let GID = wid.x * THREADS_PER_WORKGROUP + TID; // Global thread ID

    let ELM_ID = GID * 2;
    let blockSum = blockSums[wid.x];

    items[ELM_ID] += blockSum;
    items[ELM_ID + 1] += blockSum;
}`;

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
        // Numbers of workgroups needed to process all items
        const block_count = Math.ceil(count / this.items_per_workgroup);

        // Create buffer for block sums
        const blockSumBuffer = this.device.createBuffer({
            size: block_count * 4,
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
                    'ITEMS_PER_WORKGROUP': this.items_per_workgroup
                }
            }
        });

        this.pipelines.push({ pipeline: scanPipeline, bindGroup, block_count });

        if (block_count > 1) {
            // Prefix sum on block sums
            this.create_pass_recursive(blockSumBuffer, block_count);

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
            });

            this.pipelines.push({ pipeline: blockSumPipeline, bindGroup, block_count });
        }
    }

    dispatch(pass) {
        for (const { pipeline, bindGroup, block_count } of this.pipelines) {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(block_count);
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
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WID = wid.x * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    // Extract 2 bits from the input
    let elm = input[GID];
    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;

    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

    let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

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

        // Prefix sum
        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {
            if (TID >= offset) {
                s_prefix_sum[outOffset] = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
            } else {
                s_prefix_sum[outOffset] = s_prefix_sum[inOffset];
            }

            // Swap buffers
            outOffset = inOffset;
            swapOffset = TPW - swapOffset;
            inOffset = TID + swapOffset;
            
            workgroupBarrier();
        }

        // Store prefix sum for current bit
        let prefix_sum = s_prefix_sum[inOffset];
        bit_prefix_sums[b] = prefix_sum;

        if (TID == LAST_THREAD) {
            // Store block sum to global memory
            let total_sum: u32 = prefix_sum + bitmask;
            block_sums[b * WORKGROUP_COUNT + wid.x] = total_sum;
        }

        // Swap buffers
        outOffset = inOffset;
        swapOffset = TPW - swapOffset;
        inOffset = TID + swapOffset;
    }

    // Store local prefix sum to global memory
    local_prefix_sums[GID] = bit_prefix_sums[extract_bits];
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
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WID = wid.x * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    // Extract 2 bits from the input
    let elm = input[GID];
    let val = values[GID];
    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;

    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

    let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

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

        // Prefix sum
        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {
            if (TID >= offset) {
                s_prefix_sum[outOffset] = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
            } else {
                s_prefix_sum[outOffset] = s_prefix_sum[inOffset];
            }

            // Swap buffers
            outOffset = inOffset;
            swapOffset = TPW - swapOffset;
            inOffset = TID + swapOffset;
            
            workgroupBarrier();
        }

        // Store prefix sum for current bit
        let prefix_sum = s_prefix_sum[inOffset];
        bit_prefix_sums[b] = prefix_sum;

        if (TID == LAST_THREAD) {
            // Store block sum to global memory
            let total_sum: u32 = prefix_sum + bitmask;
            block_sums[b * WORKGROUP_COUNT + wid.x] = total_sum;
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
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let GID = TID + wid.x * THREADS_PER_WORKGROUP; // Global thread ID

    if (GID >= ELEMENT_COUNT) {
        return;
    }

    let k = inputKeys[GID];
    let v = inputValues[GID];

    let local_prefix = local_prefix_sum[GID];

    // Calculate new position
    let extract_bits = (k >> CURRENT_BIT) & 0x3;
    let pid = extract_bits * WORKGROUP_COUNT + wid.x;
    let sorted_position = prefix_block_sum[pid] + local_prefix;
    
    outputKeys[sorted_position] = k;
    outputValues[sorted_position] = v;
}`;

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
        this.device = device;
        this.count = count;
        this.bit_count = bit_count;
        this.workgroup_size = workgroup_size;
        this.local_shuffle = local_shuffle;
        this.avoid_bank_conflicts = avoid_bank_conflicts;

        this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;
        this.workgroup_count = Math.ceil(count / this.threads_per_workgroup);
        this.prefix_block_workgroup_count = 4 * this.workgroup_count;

        this.has_values = (values != null);

        this.shaderModules = {};
        this.buffers = {};
        this.pipelines = [];

        // Create shader modules from wgsl code
        this.create_shader_modules();

        // Create GPU buffers
        this.create_buffers(keys, values);
        
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

    create_buffers(keys, values) {
        // Keys and values double buffering
        const tmpKeysBuffer = this.device.createBuffer({
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const tmpValuesBuffer = !this.has_values ? null : this.device.createBuffer({
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Local Prefix Sum buffer (1 element per item)
        const localPrefixSumBuffer = this.device.createBuffer({
            size: this.count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Prefix Block Sum buffer (4 element per workgroup)
        const prefixBlockSumBuffer = this.device.createBuffer({
            size: this.prefix_block_workgroup_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        
        this.buffers = {
            keys: keys,
            values: values,
            tmpKeys: tmpKeysBuffer,
            tmpValues: tmpValuesBuffer,
            localPrefixSum: localPrefixSumBuffer,
            prefixBlockSum: prefixBlockSumBuffer,
        };
    }

    // Create radix sort passes for every 2 bits
    create_pipelines() {
        for (let bit = 0; bit < this.bit_count; bit += 2) {
            // Swap buffers every pass
            const even      = (bit % 4 == 0);
            const inKeys    = even ? this.buffers.keys : this.buffers.tmpKeys;
            const inValues  = even ? this.buffers.values : this.buffers.tmpValues;
            const outKeys   = even ? this.buffers.tmpKeys : this.buffers.keys;
            const outValues = even ? this.buffers.tmpValues : this.buffers.values;

            // Compute local prefix sums and block sums
            const blockSumPipeline = this.create_block_sum_pipeline(inKeys, inValues, bit);

            // Compute block sums prefix sums
            const prefixSumKernel = new PrefixSumKernel({ 
                device: this.device,
                data: this.buffers.prefixBlockSum, 
                count: this.prefix_block_workgroup_count,
                workgroup_size: this.workgroup_size,
                avoid_bank_conflicts: this.avoid_bank_conflicts,
            });
            
            // Reorder keys and values
            const reorderPipeline = this.create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit);

            this.pipelines.push({ blockSumPipeline, prefixSumKernel, reorderPipeline });
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
        for (const { blockSumPipeline, prefixSumKernel, reorderPipeline } of this.pipelines) {            
            pass.setPipeline(blockSumPipeline.pipeline);
            pass.setBindGroup(0, blockSumPipeline.bindGroup);
            pass.dispatchWorkgroups(this.workgroup_count, 1, 1);

            prefixSumKernel.dispatch(pass);

            pass.setPipeline(reorderPipeline.pipeline);
            pass.setBindGroup(0, reorderPipeline.bindGroup);
            pass.dispatchWorkgroups(this.workgroup_count, 1, 1);
        }
    }
}

exports.PrefixSumKernel = PrefixSumKernel;
exports.RadixSortKernel = RadixSortKernel;
//# sourceMappingURL=radix-sort-cjs.js.map
