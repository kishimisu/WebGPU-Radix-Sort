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
}`

const write_reduction_result = /* wgsl */ `
    if (TID == 0) {
        output[WORKGROUP_ID] = s_data[0];
    }
`

const first_pass_load_data = /* wgsl */ `
    let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

    // Load current element into shared memory
    // Also load next element for comparison
    let elm = select(0u, input[GID], GID < ELEMENT_COUNT);
    let next = select(0u, input[GID + 1], GID < ELEMENT_COUNT-1);
    s_data[TID] = elm;
    workgroupBarrier();

    s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);
`

const last_pass = (kernelMode) => /* wgsl */ `
    let fullDispatchLength = arrayLength(&output);
    let dispatchIndex = TID * 3;

    if (dispatchIndex >= fullDispatchLength) {
        return;
    }

    ${kernelMode == 'full' ? last_pass_full : last_pass_fast}
`

// If the fast check kernel is sorted and the data isn't already sorted, run the full check
const last_pass_fast = /* wgsl */ `
    output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u);
`

// If the full check kernel is sorted, set the flag to 1 and skip radix sort passes
const last_pass_full = /* wgsl */ `
    if (TID == 0 && s_data[0] == 0) {
        is_sorted = 1u;
    }

    output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] != 0);
`
export default checkSortSource