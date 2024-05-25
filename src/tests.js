import RadixSortKernel from "./kernels/RadixSortKernel.js"
import PrefixSumKernel from "./kernels/PrefixSumKernel.js"

// Test the radix sort kernel on GPU
async function test_radix_sort(device) {
    const workgroup_sizes = []
    const max_threads_per_workgroup = device.limits.maxComputeInvocationsPerWorkgroup

    const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for (let workgroup_size_x of sizes) {
        for (let workgroup_size_y of sizes) {
            if (workgroup_size_x * workgroup_size_y <= max_threads_per_workgroup) {
                workgroup_sizes.push({ x: workgroup_size_x, y: workgroup_size_y })
            }
        }
    }
    
    for (const workgroup_size of workgroup_sizes) {
        const max_elements      = Math.floor(device.limits.maxComputeWorkgroupsPerDimension * workgroup_size.x * workgroup_size.y / 10)
        const element_count     = Math.floor(max_elements  * (Math.random() * .1 + .9))
        const sub_element_count = Math.floor(element_count * Math.random() + .5)

        // Create random data
        const bit_count = 32
        const value_range = 2 ** bit_count - 1
        const keys = new Uint32Array(element_count).map(_ => Math.ceil(Math.random() * value_range))
        const values = new Uint32Array(element_count).map((_, i) => i)

        // Create GPU buffers
        const [keysBuffer, keysBufferMapped] = create_buffers(device, keys)
        const [valuesBuffer, valuesBufferMapped] = create_buffers(device, values)

        // Create kernel
        const kernel = new RadixSortKernel({
            device,
            keys: keysBuffer,
            values: valuesBuffer,
            count: sub_element_count,
            workgroup_size: workgroup_size,
            bit_count: bit_count,
        })

        // Create command buffer and compute pass
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()

        // Run kernel
        kernel.dispatch(pass)
        pass.end()

        // Copy result back to CPU
        if (bit_count % 4 === 0) {
            encoder.copyBufferToBuffer(kernel.buffers.keys, 0, keysBufferMapped, 0, element_count * 4)
            encoder.copyBufferToBuffer(kernel.buffers.values, 0, valuesBufferMapped, 0, element_count * 4)
        }
        else {
            encoder.copyBufferToBuffer(kernel.buffers.tmpKeys, 0, keysBufferMapped, 0, element_count * 4)
            encoder.copyBufferToBuffer(kernel.buffers.tmpValues, 0, valuesBufferMapped, 0, element_count * 4)
        }

        // Submit command buffer
        device.queue.submit([encoder.finish()])

        // Read result from GPU
        await keysBufferMapped.mapAsync(GPUMapMode.READ)
        const keysResult = new Uint32Array(keysBufferMapped.getMappedRange().slice())
        keysBufferMapped.unmap()

        await valuesBufferMapped.mapAsync(GPUMapMode.READ)
        const valuesResult = new Uint32Array(valuesBufferMapped.getMappedRange().slice())
        valuesBufferMapped.unmap()

        // Check result
        const expected = keys.slice(0, sub_element_count).sort((a, b) => a - b)
        const isOK = expected.every((v, i) => v === keysResult[i]) && valuesResult.every((v, i) => keysResult[i] == keys[v])

        const workgroupCount = Math.ceil(element_count / kernel.workgroup_count)
        console.log('workgroup_size', workgroupCount, element_count, sub_element_count, workgroup_size, isOK ? 'OK' : 'ERROR')

        if (!isOK) {
            console.log('keys', keys)
            console.log('keys results', keysResult)
            console.log('keys expected', expected)
            console.log('values', values)
            console.log('values result', valuesResult)
            throw new Error('Radix sort error')
        }
    }
}

// Test the prefix sum kernel on GPU
async function test_prefix_sum(device) {
    const workgroup_sizes = []
    const max_threads_per_workgroup = device.limits.maxComputeInvocationsPerWorkgroup

    const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for (let workgroup_size_x of sizes) {
        for (let workgroup_size_y of sizes) {
            if (workgroup_size_x * workgroup_size_y <= max_threads_per_workgroup) {
                workgroup_sizes.push({ x: workgroup_size_x, y: workgroup_size_y })
            }
        }
    }
    
    for (const workgroup_size of workgroup_sizes) {
        const max_elements = device.limits.maxComputeWorkgroupsPerDimension * 2 * workgroup_size.x * workgroup_size.y
        const element_count     = Math.floor(max_elements  * (Math.random() * .1 + .9))
        const sub_element_count = Math.floor(element_count * Math.random() + .5)

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

        const workgroupCount = Math.ceil(element_count / prefixSumKernel.items_per_workgroup)
        console.log('workgroup_size', workgroupCount, element_count, sub_element_count, workgroup_size, isOK ? 'OK' : 'ERROR')

        if (!isOK) {
            console.log('input', data)
            console.log('expected', expected)
            console.log('output', dataMapped)
            throw new Error('Prefix sum error')
        }
    }
}

// Test the performance of the radix sort kernel on GPU
// and optionally compare it to the CPU sort
async function test_radix_sort_performance(
    device,
    element_count = 1_000_000, 
    bit_count = 32, 
    workgroup_size_x = 16, 
    workgroup_size_y = 16,
    test_cpu = true,
    local_shuffle = false,
    avoid_bank_conflicts = false
) {
    const max_range = 2 ** bit_count
    const keys   = new Uint32Array(element_count).map(_ => Math.floor(Math.random() * max_range))
    const values = new Uint32Array(element_count).map(_ => Math.floor(Math.random() * 1_000_000))

    // Create keys and values buffers on GPU
    const [keysBuffer] = create_buffers(device, keys)
    const [valuesBuffer] = create_buffers(device, values)

    // Create timestamp query
    const timestampCount = 2
    const querySet = device.createQuerySet({
        type: "timestamp",
        count: timestampCount,
    })
    const queryBuffer = device.createBuffer({
        size: 8 * timestampCount,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    })
    const queryBufferMapped = device.createBuffer({
        size: 8 * timestampCount,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    // Create radix sort kernel
    const radixSortKernel = new RadixSortKernel({
        device,
        keys: keysBuffer,
        values: valuesBuffer,
        count: keys.length,
        workgroup_size: { x: workgroup_size_x, y: workgroup_size_y },
        bit_count: bit_count,
        local_shuffle: local_shuffle,
        avoid_bank_conflicts: avoid_bank_conflicts,
    })

    // Run and time compute pass
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass({
        timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        }
    })

    radixSortKernel.dispatch(pass)

    pass.end()

    encoder.resolveQuerySet(querySet, 0, timestampCount, queryBuffer, 0)
    encoder.copyBufferToBuffer(queryBuffer, 0, queryBufferMapped, 0, 8 * timestampCount)

    device.queue.submit([encoder.finish()])

    await queryBufferMapped.mapAsync(GPUMapMode.READ)
    const timestamps = new BigUint64Array(queryBufferMapped.getMappedRange().slice())
    queryBufferMapped.unmap()

    const gpuMs = Number(timestamps[1] - timestamps[0]) / 1e6
    let cpuMs = null

    if (test_cpu) {
        // CPU Sort
        const start = performance.now()
        keys.sort((a, b) => a - b)
        cpuMs = performance.now() - start
    }

    return { 
        cpu: cpuMs,
        gpu: gpuMs
    }
}

// Create a GPUBuffer with data from an Uint32Array
// Also create a second buffer to read back from GPU
function create_buffers(device, data) {
    // Transfer data to GPU
    const dataBuffer = device.createBuffer({
        size: data.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true
    })
    new Uint32Array(dataBuffer.getMappedRange()).set(data)
    dataBuffer.unmap()
    
    // Create buffer to read back data from CPU
    const dataBufferMapped = device.createBuffer({
        size: data.length * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    return [dataBuffer, dataBufferMapped]
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
    test_radix_sort_performance,
    create_buffers
}