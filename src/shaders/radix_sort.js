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
}`

export default radixSortSource;