import { RadixSortKernel } from "../dist/esm/radix-sort-esm.js"
import { test_prefix_sum, test_radix_sort, test_radix_sort_performance, create_buffers } from "./tests.js"

const TEST_PERFORMANCES = true

window.onload = TEST_PERFORMANCES ? main_performance() : main_demo()

/// Example Usage ///
async function main_demo() {
    // Init WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) throw new Error('Could not create WebGPU device')

    // Test Prefix Sum Kernel
    if (false) {
        return test_prefix_sum(device)
    }

    // Test Radix Sort Kernel
    if (false) {
        return test_radix_sort(device)
    }

    // Create random data
    const bit_count = 32
    const max_range = 2 ** bit_count
    const element_count = 20_000_000
    const keys   = new Uint32Array(element_count).map(_ => Math.floor(Math.random() * max_range))
    const values = new Uint32Array(element_count).map(_ => Math.floor(Math.random() * 1_000_000)) // Optional

    // Create keys and values buffers on GPU
    const [keysBuffer, keysResultBuffer] = create_buffers(device, keys)
    const [valuesBuffer, valuesResultBuffer] = create_buffers(device, values)

    // Create radix sort kernel
    const radixSortKernel = new RadixSortKernel({
        device,
        keys: keysBuffer,
        values: valuesBuffer, // Optional
        count: keys.length,
        bit_count: bit_count,
        workgroup_size: { x: 16, y: 16 },
    })

    // Run compute pass
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()

    radixSortKernel.dispatch(pass)

    pass.end()

    // Copy data to result buffers
    encoder.copyBufferToBuffer(keysBuffer, 0, keysResultBuffer, 0, keys.length * 4)
    encoder.copyBufferToBuffer(valuesBuffer, 0, valuesResultBuffer, 0, values.length * 4)
    
    device.queue.submit([encoder.finish()])

    // Read results
    await keysResultBuffer.mapAsync(GPUMapMode.READ)
    await valuesResultBuffer.mapAsync(GPUMapMode.READ)
    const keysMapped = new Uint32Array(keysResultBuffer.getMappedRange().slice())
    const valuesMapped = new Uint32Array(valuesResultBuffer.getMappedRange().slice())
    keysResultBuffer.unmap()
    valuesResultBuffer.unmap()

    console.log({
        keys: keysMapped,
        values: valuesMapped
    })
}

/// Test performances CPU/GPU ///
async function main_performance() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice({
        requiredFeatures: ['timestamp-query'],
        requiredLimits: {
            // Example on how to allow sorting for a higher amount of elements:
            // maxBufferSize:               100_000_000 * 4,
            // maxStorageBufferBindingSize: 100_000_000 * 4,
        }
    })
    if (!device) throw new Error('Could not create WebGPU device')
    
    const max_exp = 7
    for (let exp = 1; exp <= max_exp; exp++) {
        const element_count = 10 ** exp
        let cpuSum = 0
        let gpuSum = 0

        let iterations = 1
        for (let i = 0; i < iterations; i++) {
            const {cpu, gpu} = await test_radix_sort_performance(device, element_count)
            cpuSum += cpu
            gpuSum += gpu
        }

        const cpuMs = cpuSum / iterations
        const gpuMs = gpuSum / iterations
        const increase = `x${(cpuMs / gpuMs).toFixed(2)}`

        // Update DOM
        document.body.innerHTML += (`Element count: ${element_count}`)
        document.body.innerHTML += ('<br>')
        document.body.innerHTML += (`<b>[${exp}/${max_exp}] CPU: ${cpuMs.toFixed(2)}ms, GPU: ${gpuMs.toFixed(2)}ms (${increase})</b>`)
        document.body.innerHTML += ('<br>')
        document.body.innerHTML += ('-'.repeat(40))
        document.body.innerHTML += ('<br>')
    }
}