import { RadixSortKernel, PrefixSumKernel } from "../dist/esm/radix-sort-esm.js"


/** Test the radix sort kernel on GPU for integrity
 * 
 * @param {boolean} keys_and_values - Whether to include a values buffer in the test
 */
async function test_radix_sort(device, keys_and_values = false) {
    const {
        maxComputeInvocationsPerWorkgroup,
        maxStorageBufferBindingSize,
        maxBufferSize,
    } = device.limits

    const max_elements = Math.floor(Math.min(maxBufferSize, maxStorageBufferBindingSize) / 4)
    const workgroup_sizes = []

    console.log('max_elements:', max_elements)

    const sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    for (let workgroup_size_x of sizes) {
        for (let workgroup_size_y of sizes) {
            if (workgroup_size_x * workgroup_size_y <= maxComputeInvocationsPerWorkgroup) {
                workgroup_sizes.push({ x: workgroup_size_x, y: workgroup_size_y })
            }
        }
    }

    for (const workgroup_size of workgroup_sizes) 
    for (let exp = 2; exp < 8; exp++)
    {
        const element_count     = Math.floor(Math.min(max_elements, 10 ** exp) * (Math.random() * .1 + .9))
        const sub_element_count = Math.floor(element_count * Math.random() + 1)

        // Create random data
        const bit_count = 32
        const value_range = 2 ** bit_count - 1
        const keys = new Uint32Array(element_count).map(_ => Math.ceil(Math.random() * value_range))
        // const keys = new Float32Array(element_count).map(_ => Math.random() * 10)
        const values = new Uint32Array(element_count).map((_, i) => i)

        const check_order = Math.random() > .5
        const local_shuffle = Math.random() > .5
        const avoid_bank_conflicts = Math.random() > .5

        // Create GPU buffers
        const [keysBuffer, keysBufferMapped] = create_buffers(device, keys)
        const [valuesBuffer, valuesBufferMapped] = create_buffers(device, values)

        // Create kernel
        const kernel = new RadixSortKernel({
            device,
            keys: keysBuffer,
            values: keys_and_values ? valuesBuffer : null,
            count: sub_element_count,
            bit_count: bit_count,
            workgroup_size: workgroup_size,
            check_order: check_order,
            local_shuffle: local_shuffle,
            avoid_bank_conflicts: avoid_bank_conflicts,
        })

        // Create command buffer and compute pass
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()

        // Run kernel
        kernel.dispatch(pass)
        pass.end()

        // Copy result back to CPU
        encoder.copyBufferToBuffer(kernel.buffers.keys, 0, keysBufferMapped, 0, element_count * 4)
        if (keys_and_values)
            encoder.copyBufferToBuffer(kernel.buffers.values, 0, valuesBufferMapped, 0, element_count * 4)

        // Submit command buffer
        device.queue.submit([encoder.finish()])

        // Read result from GPU
        await keysBufferMapped.mapAsync(GPUMapMode.READ)
        const keysResult = new keys.constructor(keysBufferMapped.getMappedRange().slice())
        keysBufferMapped.unmap()

        // Check result
        const expected = keys.slice(0, sub_element_count).sort((a, b) => a - b)
        let isOK = expected.every((v, i) => v === keysResult[i])

        if (keys_and_values) {
            await valuesBufferMapped.mapAsync(GPUMapMode.READ)
            const valuesResult = new Uint32Array(valuesBufferMapped.getMappedRange().slice())
            valuesBufferMapped.unmap()

            isOK = isOK && valuesResult.every((v, i) => keysResult[i] == keys[v])
        }

        console.log('Test Radix Sort:', element_count, sub_element_count, workgroup_size, check_order, local_shuffle, avoid_bank_conflicts, isOK ? 'OK' : 'ERROR')

        if (!isOK) {
            console.log('keys', keys)
            console.log('keys results', keysResult)
            console.log('keys expected', expected)
            throw new Error('Radix sort error')
        }
    }
}

// Test the prefix sum kernel on GPU
async function test_prefix_sum(device) {
    const {
        maxComputeInvocationsPerWorkgroup,
        maxStorageBufferBindingSize,
        maxBufferSize,
    } = device.limits

    const max_elements = Math.floor(Math.min(maxBufferSize, maxStorageBufferBindingSize) / 4)
    const workgroup_sizes = []

    const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for (let workgroup_size_x of sizes) {
        for (let workgroup_size_y of sizes) {
            if (workgroup_size_x * workgroup_size_y <= maxComputeInvocationsPerWorkgroup) {
                workgroup_sizes.push({ x: workgroup_size_x, y: workgroup_size_y })
            }
        }
    }

    for (const workgroup_size of workgroup_sizes)
    for (let exp = 2; exp < 8; exp++) {
        const element_count     = Math.floor(Math.min(max_elements, 10 ** exp) * (Math.random() * .1 + .9))
        const sub_element_count = Math.floor(element_count * Math.random() + 1)

        // Create random data
        const data = new Uint32Array(element_count).map(_ => Math.floor(Math.random() * 8))

        // Create GPU buffers
        const [dataBuffer, dataBufferMapped] = create_buffers(device, data)

        // Create kernel
        const prefixSumKernel = new PrefixSumKernel({
            device,
            data: dataBuffer,
            count: sub_element_count,
            workgroup_size,
            avoid_bank_conflicts: false,
        })

        // Create command buffer and compute pass
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()

        // Run kernel
        prefixSumKernel.dispatch(pass)
        pass.end()

        // Copy result back to CPU
        encoder.copyBufferToBuffer(dataBuffer, 0, dataBufferMapped, 0, data.length * 4)

        // Submit command buffer
        device.queue.submit([encoder.finish()])

        // Read result from GPU
        await dataBufferMapped.mapAsync(GPUMapMode.READ)
        const dataMapped = new Uint32Array(dataBufferMapped.getMappedRange().slice())
        dataBufferMapped.unmap()

        // Check result
        const expected = prefix_sum_cpu(data.slice(0, sub_element_count))
        const isOK = expected.every((v, i) => v === dataMapped[i])

        console.log('workgroup_size', element_count, sub_element_count, workgroup_size, isOK ? 'OK' : 'ERROR')

        if (!isOK) {
            console.log('input', data)
            console.log('expected', expected)
            console.log('output', dataMapped)
            throw new Error('Prefix sum error')
        }
    }
}

// Create a GPUBuffer with data from an Uint32Array
// Also create a second buffer to read back from GPU
function create_buffers(device, data, usage = 0) {
    // Transfer data to GPU
    const dataBuffer = device.createBuffer({
        size: data.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | usage,
        mappedAtCreation: true
    })
    new data.constructor(dataBuffer.getMappedRange()).set(data)
    dataBuffer.unmap()
    
    // Create buffer to read back data from CPU
    const dataBufferMapped = device.createBuffer({
        size: data.length * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    return [dataBuffer, dataBufferMapped]
}

// Create a timestamp query object for measuring GPU time
function create_timestamp_query(device) {
    const timestampCount = 2
    const querySet = device.createQuerySet({
        type: "timestamp",
        count: timestampCount,
    })
    const queryBuffer = device.createBuffer({
        size: 8 * timestampCount,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    })
    const queryResultBuffer = device.createBuffer({
        size: 8 * timestampCount,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    const resolve = (encoder) => {    
        encoder.resolveQuerySet(querySet, 0, timestampCount, queryBuffer, 0)
        encoder.copyBufferToBuffer(queryBuffer, 0, queryResultBuffer, 0, 8 * timestampCount)
    }

    const get_timestamps = async () => {
        await queryResultBuffer.mapAsync(GPUMapMode.READ)
        const timestamps = new BigUint64Array(queryResultBuffer.getMappedRange().slice())
        queryResultBuffer.unmap()
        return timestamps
    }

    return {
        descriptor: {
            timestampWrites: {
                querySet: querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            },
        },
        resolve,
        get_timestamps
    }
}

// CPU version of the prefix sum algorithm
function prefix_sum_cpu(data) {
    const prefix_sum = []
    let sum = 0
    for (let i = 0; i < data.length; i++) {
        prefix_sum[i] = sum
        sum += data[i]
    }
    return prefix_sum
}

export {
    test_radix_sort,
    test_prefix_sum,
    create_buffers,
    create_timestamp_query,
}