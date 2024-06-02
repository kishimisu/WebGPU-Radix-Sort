import { RadixSortKernel } from "../dist/esm/radix-sort-esm.js"
import { create_buffers, create_timestamp_query, test_radix_sort } from "./tests.js"

const settings = {
    element_count: 2 ** 20,
    bit_count: 32,
    workgroup_size: 16,
    check_order: false,
    local_shuffle: false,
    avoid_bank_conflicts: false,
    sort_mode: 'Keys',
    initial_sort: 'Random',
    consecutive_sorts: 1,
}

window.onload = async function main() {
    const gui = new GUI()

    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice({
        requiredFeatures: ['timestamp-query'],
        requiredLimits: {
            maxComputeInvocationsPerWorkgroup: 32 * 32,
        }
    })

    if (!device) {
        document.getElementById('info').innerHTML = 'WebGPU doesn\'t appear to be supported on this device.'
        document.getElementById('info').style.color = '#ff7a48'
        throw new Error('Could not create WebGPU device')
    }

    if (false) {
        test_radix_sort(device)
    }

    gui.onClickSort = () => onClickSort(device)
}

async function runSort(device, compare_against_cpu = true) {
    const keys = new Uint32Array(settings.element_count)
    const keys_range = 2 ** settings.bit_count

    switch (settings.initial_sort) {
        case 'Random':
            keys.forEach((_, i) => keys[i] = Math.floor(Math.random() * keys_range))
            break
        case 'Sorted':
            keys.forEach((_, i) => keys[i] = i)
            break
    }

    const [keysBuffer] = create_buffers(device, keys)

    let valuesBuffer = null

    if (settings.sort_mode === 'Keys & Values') {
        const values = new Uint32Array(settings.element_count).map(_ => Math.floor(Math.random() * 1_000_000))
        const buffers = create_buffers(device, values)
        valuesBuffer = buffers[0]
    }
    
    const kernel = new RadixSortKernel({
        device,
        keys: keysBuffer,
        values: valuesBuffer,
        count: settings.element_count,
        bit_count: settings.bit_count,
        workgroup_size: { x: settings.workgroup_size, y: settings.workgroup_size },
        check_order: settings.check_order,
        local_shuffle: settings.local_shuffle,
        avoid_bank_conflicts: settings.avoid_bank_conflicts,
    })

    const query = create_timestamp_query(device)

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass(query.descriptor)
    kernel.dispatch(pass)
    pass.end()

    query.resolve(encoder)
    device.queue.submit([encoder.finish()])

    const timestamps = await query.get_timestamps()

    const times = {
        cpu: null,
        gpu: Number(timestamps[1] - timestamps[0]) / 1e6
    }

    if (compare_against_cpu) {
        const start = performance.now()
        keys.sort((a, b) => a - b)
        times.cpu = performance.now() - start
    }
    
    return times
}

async function onClickSort(device) {
    let cpuTime = 0
    let gpuTime = 0

    const output = document.getElementById('results')
    if (output.children[0].id === 'info') output.innerHTML = ''
    const result = createElm(output, 'div', 'result')

    result.innerHTML += `[${output.children.length}] `
    result.innerHTML += `Sorting ${makeColor(prettifyNumber(settings.element_count), '#ff9933')} ${settings.sort_mode.toLowerCase()} of ${makeColor(settings.bit_count, '#ff9933')} bits`
    result.innerHTML += `<br>Initial sort: ${settings.initial_sort}, Workgroup size: ${settings.workgroup_size}x${settings.workgroup_size}`
    result.innerHTML += `<br>Optimizations: (${settings.check_order}, ${settings.local_shuffle}, ${settings.avoid_bank_conflicts})`

    for (let i = 0; i < settings.consecutive_sorts; i++) {
        const times = await runSort(device, i == 0)
        cpuTime += times.cpu
        gpuTime += times.gpu
    }

    const gpuAverage = gpuTime / settings.consecutive_sorts

    result.innerHTML += `<br>> CPU Reference: ${makeColor(cpuTime.toFixed(2) + 'ms', '#abff33')}, ` 
    result.innerHTML += `GPU Average (${makeColor(settings.consecutive_sorts, '#ff9933')} sorts): ${makeColor(gpuAverage.toFixed(2) + 'ms', '#abff33')}, ` 
    result.innerHTML += `Speedup: ${makeColor('x'+(cpuTime / gpuAverage).toFixed(2), cpuTime / gpuAverage >= 1 ? '#abff33' : '#ff3333')}<br><br>`
    result.scrollIntoView({ behavior: 'smooth', block: 'end' })
}

class GUI {
    constructor(device) {
        this.dom = document.getElementById('gui')

        this.createTitle('Radix Sort Kernel')
        this.createSlider(settings, 'element_count', 'Element Count', settings.element_count, 1e4, 2 ** 24, 1, true, false)
        this.createSlider(settings, 'bit_count', 'Bit Count', settings.bit_count, 4, 32, 4, false, false)
        this.createSlider(settings, 'workgroup_size', 'Workgroup Size', settings.workgroup_size, 2, 5, 1, false, true)
    
        this.createTitle('Optimizations')
        this.createCheckbox(settings, 'check_order', 'Check If Sorted', settings.check_order)
        this.createCheckbox(settings, 'local_shuffle', 'Local Shuffle', settings.local_shuffle)
        this.createCheckbox(settings, 'avoid_bank_conflicts', 'Avoid Bank Conflicts', settings.avoid_bank_conflicts)
    
        this.createTitle('Testing')
        this.createDropdown(settings, 'initial_sort', 'Initial Sort', ['Random', 'Sorted'], 0)
        this.createDropdown(settings, 'sort_mode', 'Sort Mode', ['Keys', 'Keys & Values'], 0)
        this.createSlider(settings, 'consecutive_sorts', 'Consecutive Sorts', settings.consecutive_sorts, 1, 20, 1, false, false)
    
        this.createButton('Run Radix Sort', () => this.onClickSort(device))

        this.addHints()
    }

    addHints() {
        const elements = this.dom.querySelectorAll('.gui-ctn')
        const hints = [
            '',
            'Number of elements to sort',
            'Number of bits to sort',
            'Workgroup size in x and y dimensions',
            '',
            'Check if the data is sorted after each pass to stop the kernel early',
            'Use local shuffle optimization (does not seem to improve performance)',
            'Avoid bank conflicts in shared memory (does not seem to improve performance)',
            '',
            'Initial order of the elements',
            'Whether to use values in addition to keys',
            'Number of consecutive sorts to run',
            'Run the Radix Sort kernel !'
        ]
        elements.forEach((elm, index) => {
            elm.title = hints[index]
        })
    }

    createTitle(name) {
        const ctn = createElm(this.dom, 'div', 'gui-ctn')
        createElm(ctn, 'div', 'gui-title', { innerText: name })
    }

    createSlider(object, prop, name, value = 0, min = 0, max = 1, step = 0.01, logarithmic = false, power_of_two = false, callback = null) {
        const logMin = Math.log(min)
        const logMax = Math.log(max)
        const scale = (logMax - logMin) / (max - min)
        const convert = (x) => logarithmic ? Math.round(Math.exp(logMin + scale * (x - min))) : power_of_two ? Math.pow(2, x) : x
        const convert_back = (x) => logarithmic ? (Math.log(x) - logMin) / scale + min : power_of_two ? Math.log2(x) : x

        const ctn = createElm(this.dom, 'div', 'gui-ctn')
        const label = createElm(ctn, 'label', 'gui-label', { innerText: name })
        const input = createElm(ctn, 'div', 'gui-input')
        const slider = createElm(input, 'input', 'gui-slider', { type: 'range', min, max, step, value: convert_back(value) })
        const val = createElm(input, 'span', 'gui-value', { innerText: prettifyNumber(value) })

        slider.addEventListener('input', () => {
            let newValue = slider.value;
            if (logarithmic || power_of_two ) {
                newValue = convert(parseFloat(slider.value))
            }
            object[prop] = parseFloat(newValue)
            val.innerText = prettifyNumber(newValue)
            if (callback) callback(parseFloat(newValue))
        })
    }

    createCheckbox(object, prop, name, value = false, callback = null) {
        const ctn = createElm(this.dom, 'div', 'gui-ctn')
        const label = createElm(ctn, 'label', 'gui-label', { innerText: name })
        const input = createElm(ctn, 'div', 'gui-input')
        const check = createElm(input, 'input', 'gui-checkbox', { type: 'checkbox', checked: value })
        const val = createElm(input, 'span', 'gui-value', { innerText: value ? 'true' : 'false' })
        check.addEventListener('change', () => {
            object[prop] = check.checked
            val.innerText = check.checked ? 'true' : 'false'
            if (callback) callback(check.checked)
        })
    }

    createDropdown(object, prop, name, options, value = 0, callback = null) {
        const ctn = createElm(this.dom, 'div', 'gui-ctn')
        const label = createElm(ctn, 'label', 'gui-label', { innerText: name })
        const input = createElm(ctn, 'div', 'gui-input')
        const select = createElm(input, 'select', 'gui-select')
        const val = createElm(input, 'span', 'gui-value', { innerText: options[value] })
        options.forEach((option, index) => {
            const opt = createElm(select, 'option', '', { innerText: option, value: index })
            if (index === value) opt.selected = true
        })
        select.addEventListener('change', () => {
            object[prop] = options[parseInt(select.value)]
            val.innerText = options[parseInt(select.value)] 
            if (callback) callback(parseInt(select.value))
        })
    }

    createButton(name, callback) {
        const ctn = createElm(this.dom, 'div', 'gui-ctn')
        const button = createElm(ctn, 'button', 'gui-button', { innerText: name })
        button.addEventListener('click', callback)
    }
}

const createElm = (parent, tag, className, args = {}) => {
    const elm = document.createElement(tag)
    elm.className = className
    Object.keys(args).forEach(key => elm[key] = args[key])
    parent.appendChild(elm)
    return elm
}

const makeColor = (val, col) => `<span style="color:${col}">${val}</span>`
const prettifyNumber = (x) => x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")