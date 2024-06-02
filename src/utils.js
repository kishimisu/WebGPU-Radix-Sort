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
    }

    if (workgroup_count > device.limits.maxComputeWorkgroupsPerDimension) {
        const x = Math.floor(Math.sqrt(workgroup_count))
        const y = Math.ceil(workgroup_count / x)
        
        dispatchSize.x = x
        dispatchSize.y = y
    }

    return dispatchSize
}

function create_buffer_from_data({device, label, data, usage = 0}) {
    const dispatchSizes = device.createBuffer({
        label: label,
        usage: usage,
        size: data.length * 4,
        mappedAtCreation: true
    })

    const dispatchData = new Uint32Array(dispatchSizes.getMappedRange())
    dispatchData.set(data)
    dispatchSizes.unmap()

    return dispatchSizes
}

export {
    find_optimal_dispatch_size,
    create_buffer_from_data,
}