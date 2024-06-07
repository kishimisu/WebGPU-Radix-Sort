(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.RadixSort = {}));
})(this, (function (exports) { 'use strict';

  function _arrayLikeToArray(r, a) {
    (null == a || a > r.length) && (a = r.length);
    for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
    return n;
  }
  function _arrayWithoutHoles(r) {
    if (Array.isArray(r)) return _arrayLikeToArray(r);
  }
  function _assertClassBrand(e, t, n) {
    if ("function" == typeof e ? e === t : e.has(t)) return arguments.length < 3 ? t : n;
    throw new TypeError("Private element is not present on this object");
  }
  function _checkPrivateRedeclaration(e, t) {
    if (t.has(e)) throw new TypeError("Cannot initialize the same private elements twice on an object");
  }
  function _classCallCheck(a, n) {
    if (!(a instanceof n)) throw new TypeError("Cannot call a class as a function");
  }
  function _classPrivateMethodInitSpec(e, a) {
    _checkPrivateRedeclaration(e, a), a.add(e);
  }
  function _defineProperties(e, r) {
    for (var t = 0; t < r.length; t++) {
      var o = r[t];
      o.enumerable = o.enumerable || !1, o.configurable = !0, "value" in o && (o.writable = !0), Object.defineProperty(e, _toPropertyKey(o.key), o);
    }
  }
  function _createClass(e, r, t) {
    return r && _defineProperties(e.prototype, r), t && _defineProperties(e, t), Object.defineProperty(e, "prototype", {
      writable: !1
    }), e;
  }
  function _defineProperty(e, r, t) {
    return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, {
      value: t,
      enumerable: !0,
      configurable: !0,
      writable: !0
    }) : e[r] = t, e;
  }
  function _iterableToArray(r) {
    if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
  }
  function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  function ownKeys(e, r) {
    var t = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var o = Object.getOwnPropertySymbols(e);
      r && (o = o.filter(function (r) {
        return Object.getOwnPropertyDescriptor(e, r).enumerable;
      })), t.push.apply(t, o);
    }
    return t;
  }
  function _objectSpread2(e) {
    for (var r = 1; r < arguments.length; r++) {
      var t = null != arguments[r] ? arguments[r] : {};
      r % 2 ? ownKeys(Object(t), !0).forEach(function (r) {
        _defineProperty(e, r, t[r]);
      }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) {
        Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r));
      });
    }
    return e;
  }
  function _toConsumableArray(r) {
    return _arrayWithoutHoles(r) || _iterableToArray(r) || _unsupportedIterableToArray(r) || _nonIterableSpread();
  }
  function _toPrimitive(t, r) {
    if ("object" != typeof t || !t) return t;
    var e = t[Symbol.toPrimitive];
    if (void 0 !== e) {
      var i = e.call(t, r || "default");
      if ("object" != typeof i) return i;
      throw new TypeError("@@toPrimitive must return a primitive value.");
    }
    return ("string" === r ? String : Number)(t);
  }
  function _toPropertyKey(t) {
    var i = _toPrimitive(t, "string");
    return "symbol" == typeof i ? i : i + "";
  }
  function _unsupportedIterableToArray(r, a) {
    if (r) {
      if ("string" == typeof r) return _arrayLikeToArray(r, a);
      var t = {}.toString.call(r).slice(8, -1);
      return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0;
    }
  }

  var prefixSumSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read_write> items: array<u32>;\n@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;\n\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride ITEMS_PER_WORKGROUP: u32;\noverride ELEMENT_COUNT: u32;\n\nvar<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn reduce_downsweep(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n    \n    let ELM_TID = TID * 2; // Element pair local ID\n    let ELM_GID = GID * 2; // Element pair global ID\n    \n    // Load input to shared memory\n    temp[ELM_TID]     = select(items[ELM_GID], 0, ELM_GID >= ELEMENT_COUNT);\n    temp[ELM_TID + 1] = select(items[ELM_GID + 1], 0, ELM_GID + 1 >= ELEMENT_COUNT);\n\n    var offset: u32 = 1;\n\n    // Up-sweep (reduce) phase\n    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n            temp[bi] += temp[ai];\n        }\n\n        offset *= 2;\n    }\n\n    // Save workgroup sum and clear last element\n    if (TID == 0) {\n        let last_offset = ITEMS_PER_WORKGROUP - 1;\n\n        blockSums[WORKGROUP_ID] = temp[last_offset];\n        temp[last_offset] = 0;\n    }\n\n    // Down-sweep phase\n    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {\n        offset >>= 1;\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n\n            let t: u32 = temp[ai];\n            temp[ai] = temp[bi];\n            temp[bi] += t;\n        }\n    }\n    workgroupBarrier();\n\n    // Copy result from shared memory to global memory\n    if (ELM_GID >= ELEMENT_COUNT) {\n        return;\n    }\n    items[ELM_GID] = temp[ELM_TID];\n\n    if (ELM_GID + 1 >= ELEMENT_COUNT) {\n        return;\n    }\n    items[ELM_GID + 1] = temp[ELM_TID + 1];\n}\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn add_block_sums(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    let ELM_ID = GID * 2;\n\n    if (ELM_ID >= ELEMENT_COUNT) {\n        return;\n    }\n\n    let blockSum = blockSums[WORKGROUP_ID];\n\n    items[ELM_ID] += blockSum;\n\n    if (ELM_ID + 1 >= ELEMENT_COUNT) {\n        return;\n    }\n\n    items[ELM_ID + 1] += blockSum;\n}";

  /**
   * Prefix sum with optimization to avoid bank conflicts
   * 
   * (see Implementation section in README for details)
   */
  var prefixSumNoBankConflictSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read_write> items: array<u32>;\n@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;\n\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride ITEMS_PER_WORKGROUP: u32;\noverride ELEMENT_COUNT: u32;\n\nconst NUM_BANKS: u32 = 32;\nconst LOG_NUM_BANKS: u32 = 5;\n\nfn get_offset(offset: u32) -> u32 {\n    // return offset >> LOG_NUM_BANKS; // Conflict-free\n    return (offset >> NUM_BANKS) + (offset >> (2 * LOG_NUM_BANKS)); // Zero bank conflict\n}\n\nvar<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn reduce_downsweep(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n    \n    let ELM_TID = TID * 2; // Element pair local ID\n    let ELM_GID = GID * 2; // Element pair global ID\n    \n    // Load input to shared memory\n    let ai: u32 = TID;\n    let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1);\n    let s_ai = ai + get_offset(ai);\n    let s_bi = bi + get_offset(bi);\n    let g_ai = ai + WID * 2;\n    let g_bi = bi + WID * 2;\n    temp[s_ai] = select(items[g_ai], 0, g_ai >= ELEMENT_COUNT);\n    temp[s_bi] = select(items[g_bi], 0, g_bi >= ELEMENT_COUNT);\n\n    var offset: u32 = 1;\n\n    // Up-sweep (reduce) phase\n    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n            ai += get_offset(ai);\n            bi += get_offset(bi);\n            temp[bi] += temp[ai];\n        }\n\n        offset *= 2;\n    }\n\n    // Save workgroup sum and clear last element\n    if (TID == 0) {\n        var last_offset = ITEMS_PER_WORKGROUP - 1;\n        last_offset += get_offset(last_offset);\n\n        blockSums[WORKGROUP_ID] = temp[last_offset];\n        temp[last_offset] = 0;\n    }\n\n    // Down-sweep phase\n    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {\n        offset >>= 1;\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n            ai += get_offset(ai);\n            bi += get_offset(bi);\n\n            let t: u32 = temp[ai];\n            temp[ai] = temp[bi];\n            temp[bi] += t;\n        }\n    }\n    workgroupBarrier();\n\n    // Copy result from shared memory to global memory\n    if (g_ai < ELEMENT_COUNT) {\n        items[g_ai] = temp[s_ai];\n    }\n    if (g_bi < ELEMENT_COUNT) {\n        items[g_bi] = temp[s_bi];\n    }\n}\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn add_block_sums(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    let ELM_ID = GID * 2;\n\n    if (ELM_ID >= ELEMENT_COUNT) {\n        return;\n    }\n\n    let blockSum = blockSums[WORKGROUP_ID];\n\n    items[ELM_ID] += blockSum;\n\n    if (ELM_ID + 1 >= ELEMENT_COUNT) {\n        return;\n    }\n\n    items[ELM_ID + 1] += blockSum;\n}";

  /**
   * Find the best dispatch size x and y dimensions to minimize unused workgroups
   * 
   * @param {GPUDevice} device - The GPU device
   * @param {int} workgroup_count - Number of workgroups to dispatch
   * @returns 
   */
  function find_optimal_dispatch_size(device, workgroup_count) {
    var dispatchSize = {
      x: workgroup_count,
      y: 1
    };
    if (workgroup_count > device.limits.maxComputeWorkgroupsPerDimension) {
      var x = Math.floor(Math.sqrt(workgroup_count));
      var y = Math.ceil(workgroup_count / x);
      dispatchSize.x = x;
      dispatchSize.y = y;
    }
    return dispatchSize;
  }
  function create_buffer_from_data(_ref) {
    var device = _ref.device,
      label = _ref.label,
      data = _ref.data,
      _ref$usage = _ref.usage,
      usage = _ref$usage === void 0 ? 0 : _ref$usage;
    var dispatchSizes = device.createBuffer({
      label: label,
      usage: usage,
      size: data.length * 4,
      mappedAtCreation: true
    });
    var dispatchData = new Uint32Array(dispatchSizes.getMappedRange());
    dispatchData.set(data);
    dispatchSizes.unmap();
    return dispatchSizes;
  }

  var PrefixSumKernel = /*#__PURE__*/function () {
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
    function PrefixSumKernel(_ref) {
      var device = _ref.device,
        data = _ref.data,
        count = _ref.count,
        _ref$workgroup_size = _ref.workgroup_size,
        workgroup_size = _ref$workgroup_size === void 0 ? {
          x: 16,
          y: 16
        } : _ref$workgroup_size,
        _ref$avoid_bank_confl = _ref.avoid_bank_conflicts,
        avoid_bank_conflicts = _ref$avoid_bank_confl === void 0 ? false : _ref$avoid_bank_confl;
      _classCallCheck(this, PrefixSumKernel);
      this.device = device;
      this.workgroup_size = workgroup_size;
      this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;
      this.items_per_workgroup = 2 * this.threads_per_workgroup; // 2 items are processed per thread

      if (Math.log2(this.threads_per_workgroup) % 1 !== 0) throw new Error("workgroup_size.x * workgroup_size.y must be a power of two. (current: ".concat(this.threads_per_workgroup, ")"));
      this.pipelines = [];
      this.shaderModule = this.device.createShaderModule({
        label: 'prefix-sum',
        code: avoid_bank_conflicts ? prefixSumNoBankConflictSource : prefixSumSource
      });
      this.create_pass_recursive(data, count);
    }
    return _createClass(PrefixSumKernel, [{
      key: "create_pass_recursive",
      value: function create_pass_recursive(data, count) {
        // Find best dispatch x and y dimensions to minimize unused threads
        var workgroup_count = Math.ceil(count / this.items_per_workgroup);
        var dispatchSize = find_optimal_dispatch_size(this.device, workgroup_count);

        // Create buffer for block sums        
        var blockSumBuffer = this.device.createBuffer({
          label: 'prefix-sum-block-sum',
          size: workgroup_count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Create bind group and pipeline layout
        var bindGroupLayout = this.device.createBindGroupLayout({
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }]
        });
        var bindGroup = this.device.createBindGroup({
          label: 'prefix-sum-bind-group',
          layout: bindGroupLayout,
          entries: [{
            binding: 0,
            resource: {
              buffer: data
            }
          }, {
            binding: 1,
            resource: {
              buffer: blockSumBuffer
            }
          }]
        });
        var pipelineLayout = this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        });

        // Per-workgroup (block) prefix sum
        var scanPipeline = this.device.createComputePipeline({
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
              'ELEMENT_COUNT': count
            }
          }
        });
        this.pipelines.push({
          pipeline: scanPipeline,
          bindGroup: bindGroup,
          dispatchSize: dispatchSize
        });
        if (workgroup_count > 1) {
          // Prefix sum on block sums
          this.create_pass_recursive(blockSumBuffer, workgroup_count);

          // Add block sums to local prefix sums
          var blockSumPipeline = this.device.createComputePipeline({
            label: 'prefix-sum-add-block-pipeline',
            layout: pipelineLayout,
            compute: {
              module: this.shaderModule,
              entryPoint: 'add_block_sums',
              constants: {
                'WORKGROUP_SIZE_X': this.workgroup_size.x,
                'WORKGROUP_SIZE_Y': this.workgroup_size.y,
                'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
                'ELEMENT_COUNT': count
              }
            }
          });
          this.pipelines.push({
            pipeline: blockSumPipeline,
            bindGroup: bindGroup,
            dispatchSize: dispatchSize
          });
        }
      }
    }, {
      key: "get_dispatch_chain",
      value: function get_dispatch_chain() {
        return this.pipelines.flatMap(function (p) {
          return [p.dispatchSize.x, p.dispatchSize.y, 1];
        });
      }

      /**
       * Encode the prefix sum pipeline into the current pass.
       * If dispatchSizeBuffer is provided, the dispatch will be indirect (dispatchWorkgroupsIndirect)
       * 
       * @param {GPUComputePassEncoder} pass 
       * @param {GPUBuffer} dispatchSizeBuffer - (optional) Indirect dispatch buffer
       * @param {int} offset - (optional) Offset in bytes in the dispatch buffer. Default: 0
       */
    }, {
      key: "dispatch",
      value: function dispatch(pass, dispatchSizeBuffer) {
        var offset = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;
        for (var i = 0; i < this.pipelines.length; i++) {
          var _this$pipelines$i = this.pipelines[i],
            pipeline = _this$pipelines$i.pipeline,
            bindGroup = _this$pipelines$i.bindGroup,
            dispatchSize = _this$pipelines$i.dispatchSize;
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          if (dispatchSizeBuffer == null) pass.dispatchWorkgroups(dispatchSize.x, dispatchSize.y, 1);else pass.dispatchWorkgroupsIndirect(dispatchSizeBuffer, offset + i * 3 * 4);
        }
      }
    }]);
  }();

  var radixSortSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read> input: array<u32>;\n@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;\n@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;\n\noverride WORKGROUP_COUNT: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride CURRENT_BIT: u32;\noverride ELEMENT_COUNT: u32;\n\nvar<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn radix_sort(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    // Extract 2 bits from the input\n    let elm = select(input[GID], 0, GID >= ELEMENT_COUNT);\n    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;\n\n    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);\n\n    // If the workgroup is inactive, prevent block_sums buffer update\n    var LAST_THREAD: u32 = 0xffffffff; \n\n    if (WORKGROUP_ID < WORKGROUP_COUNT) {\n        // Otherwise store the index of the last active thread in the workgroup\n        LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;\n    }\n\n    // Initialize parameters for double-buffering\n    let TPW = THREADS_PER_WORKGROUP + 1;\n    var swapOffset: u32 = 0;\n    var inOffset:  u32 = TID;\n    var outOffset: u32 = TID + TPW;\n\n    // 4-way prefix sum\n    for (var b: u32 = 0; b < 4; b++) {\n        // Initialize local prefix with bitmask\n        let bitmask = select(0u, 1u, extract_bits == b);\n        s_prefix_sum[inOffset + 1] = bitmask;\n        workgroupBarrier();\n\n        var prefix_sum: u32 = 0;\n\n        // Prefix sum\n        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {\n            if (TID >= offset) {\n                prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];\n            } else {\n                prefix_sum = s_prefix_sum[inOffset];\n            }\n\n            s_prefix_sum[outOffset] = prefix_sum;\n            \n            // Swap buffers\n            outOffset = inOffset;\n            swapOffset = TPW - swapOffset;\n            inOffset = TID + swapOffset;\n            \n            workgroupBarrier();\n        }\n\n        // Store prefix sum for current bit\n        bit_prefix_sums[b] = prefix_sum;\n\n        if (TID == LAST_THREAD) {\n            // Store block sum to global memory\n            let total_sum: u32 = prefix_sum + bitmask;\n            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;\n        }\n\n        // Swap buffers\n        outOffset = inOffset;\n        swapOffset = TPW - swapOffset;\n        inOffset = TID + swapOffset;\n    }\n\n    if (GID < ELEMENT_COUNT) {\n        // Store local prefix sum to global memory\n        local_prefix_sums[GID] = bit_prefix_sums[extract_bits];\n    }\n}";

  /**
   * Radix sort with "local shuffle and coalesced mapping" optimization
   * 
   * (see Implementation section in README for details)
   */
  var radixSortCoalescedSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read_write> input: array<u32>;\n@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;\n@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;\n@group(0) @binding(3) var<storage, read_write> values: array<u32>;\n\noverride WORKGROUP_COUNT: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride CURRENT_BIT: u32;\noverride ELEMENT_COUNT: u32;\n\nvar<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;\nvar<workgroup> s_prefix_sum_scan: array<u32, 4>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn radix_sort(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    // Extract 2 bits from the input\n    var elm: u32 = 0;\n    var val: u32 = 0;\n    if (GID < ELEMENT_COUNT) {\n        elm = input[GID];\n        val = values[GID];\n    }\n    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;\n\n    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);\n\n    // If the workgroup is inactive, prevent block_sums buffer update\n    var LAST_THREAD: u32 = 0xffffffff; \n\n    if (WORKGROUP_ID < WORKGROUP_COUNT) {\n        // Otherwise store the index of the last active thread in the workgroup\n        LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;\n    }\n\n    // Initialize parameters for double-buffering\n    let TPW = THREADS_PER_WORKGROUP + 1;\n    var swapOffset: u32 = 0;\n    var inOffset:  u32 = TID;\n    var outOffset: u32 = TID + TPW;\n\n    // 4-way prefix sum\n    for (var b: u32 = 0; b < 4; b++) {\n        // Initialize local prefix with bitmask\n        let bitmask = select(0u, 1u, extract_bits == b);\n        s_prefix_sum[inOffset + 1] = bitmask;\n        workgroupBarrier();\n\n        var prefix_sum: u32 = 0;\n\n        // Prefix sum\n        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {\n            if (TID >= offset) {\n                prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];\n            } else {\n                prefix_sum = s_prefix_sum[inOffset];\n            }\n\n            s_prefix_sum[outOffset] = prefix_sum;\n\n            // Swap buffers\n            outOffset = inOffset;\n            swapOffset = TPW - swapOffset;\n            inOffset = TID + swapOffset;\n            \n            workgroupBarrier();\n        }\n\n        // Store prefix sum for current bit\n        bit_prefix_sums[b] = prefix_sum;\n\n        if (TID == LAST_THREAD) {\n            // Store block sum to global memory\n            let total_sum: u32 = prefix_sum + bitmask;\n            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;\n        }\n\n        // Swap buffers\n        outOffset = inOffset;\n        swapOffset = TPW - swapOffset;\n        inOffset = TID + swapOffset;\n    }\n\n    let prefix_sum = bit_prefix_sums[extract_bits];   \n\n    // Scan bit prefix sums\n    if (TID == LAST_THREAD) {\n        var sum: u32 = 0;\n        bit_prefix_sums[extract_bits] += 1;\n        for (var i: u32 = 0; i < 4; i++) {\n            s_prefix_sum_scan[i] = sum;\n            sum += bit_prefix_sums[i];\n        }\n    }\n    workgroupBarrier();\n\n    if (GID < ELEMENT_COUNT) {\n        // Compute new position\n        let new_pos: u32 = prefix_sum + s_prefix_sum_scan[extract_bits];\n\n        // Shuffle elements locally\n        input[WID + new_pos] = elm;\n        values[WID + new_pos] = val;\n        local_prefix_sums[WID + new_pos] = prefix_sum;\n    }\n}";

  var radixSortReorderSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;\n@group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;\n@group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;\n@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;\n@group(0) @binding(4) var<storage, read> inputValues: array<u32>;\n@group(0) @binding(5) var<storage, read_write> outputValues: array<u32>;\n\noverride WORKGROUP_COUNT: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride CURRENT_BIT: u32;\noverride ELEMENT_COUNT: u32;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn radix_sort_reorder(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) { \n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    if (GID >= ELEMENT_COUNT) {\n        return;\n    }\n\n    let k = inputKeys[GID];\n    let v = inputValues[GID];\n\n    let local_prefix = local_prefix_sum[GID];\n\n    // Calculate new position\n    let extract_bits = (k >> CURRENT_BIT) & 0x3;\n    let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;\n    let sorted_position = prefix_block_sum[pid] + local_prefix;\n    \n    outputKeys[sorted_position] = k;\n    outputValues[sorted_position] = v;\n}";

  var checkSortSource = function checkSortSource() {
    var isFirstPass = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : false;
    var isLastPass = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    var kernelMode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 'full';
    return /* wgsl */"\n\n@group(0) @binding(0) var<storage, read> input: array<u32>;\n@group(0) @binding(1) var<storage, read_write> output: array<u32>;\n@group(0) @binding(2) var<storage, read> original: array<u32>;\n@group(0) @binding(3) var<storage, read_write> is_sorted: u32;\n\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride ELEMENT_COUNT: u32;\noverride START_ELEMENT: u32;\n\nvar<workgroup> s_data: array<u32, THREADS_PER_WORKGROUP>;\n\n// Reset dispatch buffer and is_sorted flag\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn reset(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    if (TID >= ELEMENT_COUNT) {\n        return;\n    }\n\n    if (TID == 0) {\n        is_sorted = 0u;\n    }\n\n    let ELM_ID = TID * 3;\n\n    output[ELM_ID] = original[ELM_ID];\n}\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn check_sort(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP + START_ELEMENT;\n    let GID = TID + WID; // Global thread ID\n\n    // Load data into shared memory\n    ".concat(isFirstPass ? first_pass_load_data : "s_data[TID] = select(0u, input[GID], GID < ELEMENT_COUNT);", "\n\n    // Perform parallel reduction\n    for (var d = 1u; d < THREADS_PER_WORKGROUP; d *= 2u) {      \n        workgroupBarrier();  \n        if (TID % (2u * d) == 0u) {\n            s_data[TID] += s_data[TID + d];\n        }\n    }\n    workgroupBarrier();\n\n    // Write reduction result\n    ").concat(isLastPass ? last_pass(kernelMode) : write_reduction_result, "\n}");
  };
  var write_reduction_result = /* wgsl */"\n    if (TID == 0) {\n        output[WORKGROUP_ID] = s_data[0];\n    }\n";
  var first_pass_load_data = /* wgsl */"\n    let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;\n\n    // Load current element into shared memory\n    // Also load next element for comparison\n    let elm = select(0u, input[GID], GID < ELEMENT_COUNT);\n    let next = select(0u, input[GID + 1], GID < ELEMENT_COUNT-1);\n    s_data[TID] = elm;\n    workgroupBarrier();\n\n    s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);\n";
  var last_pass = function last_pass(kernelMode) {
    return /* wgsl */"\n    let fullDispatchLength = arrayLength(&output);\n    let dispatchIndex = TID * 3;\n\n    if (dispatchIndex >= fullDispatchLength) {\n        return;\n    }\n\n    ".concat(kernelMode == 'full' ? last_pass_full : last_pass_fast, "\n");
  };

  // If the fast check kernel is sorted and the data isn't already sorted, run the full check
  var last_pass_fast = /* wgsl */"\n    output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u);\n";

  // If the full check kernel is sorted, set the flag to 1 and skip radix sort passes
  var last_pass_full = /* wgsl */"\n    if (TID == 0 && s_data[0] == 0) {\n        is_sorted = 1u;\n    }\n\n    output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] != 0);\n";

  var CheckSortKernel = /*#__PURE__*/function () {
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
    function CheckSortKernel(_ref) {
      var device = _ref.device,
        data = _ref.data,
        result = _ref.result,
        original = _ref.original,
        is_sorted = _ref.is_sorted,
        count = _ref.count,
        _ref$start = _ref.start,
        start = _ref$start === void 0 ? 0 : _ref$start,
        _ref$mode = _ref.mode,
        mode = _ref$mode === void 0 ? 'full' : _ref$mode,
        _ref$workgroup_size = _ref.workgroup_size,
        workgroup_size = _ref$workgroup_size === void 0 ? {
          x: 16,
          y: 16
        } : _ref$workgroup_size;
      _classCallCheck(this, CheckSortKernel);
      this.device = device;
      this.count = count;
      this.start = start;
      this.mode = mode;
      this.workgroup_size = workgroup_size;
      this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;
      this.pipelines = [];
      this.buffers = {
        data: data,
        result: result,
        original: original,
        is_sorted: is_sorted,
        outputs: []
      };
      this.create_passes_recursive(data, count);
    }

    // Find the best dispatch size for each pass to minimize unused workgroups
    return _createClass(CheckSortKernel, [{
      key: "create_passes_recursive",
      value: function create_passes_recursive(buffer, count) {
        var passIndex = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;
        var workgroup_count = Math.ceil(count / this.threads_per_workgroup);
        var isFirstPass = passIndex === 0;
        var isLastPass = workgroup_count <= 1;
        var label = "check-sort-".concat(this.mode, "-").concat(passIndex);
        var outputBuffer = isLastPass ? this.buffers.result : this.device.createBuffer({
          label: label,
          size: workgroup_count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        var bindGroupLayout = this.device.createBindGroupLayout({
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'read-only-storage'
            }
          }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }].concat(_toConsumableArray(isLastPass ? [{
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'read-only-storage'
            }
          }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }] : []))
        });
        var bindGroup = this.device.createBindGroup({
          layout: bindGroupLayout,
          entries: [{
            binding: 0,
            resource: {
              buffer: buffer
            }
          }, {
            binding: 1,
            resource: {
              buffer: outputBuffer
            }
          }].concat(_toConsumableArray(isLastPass ? [{
            binding: 2,
            resource: {
              buffer: this.buffers.original
            }
          }, {
            binding: 3,
            resource: {
              buffer: this.buffers.is_sorted
            }
          }] : []))
        });
        var pipelineLayout = this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        });
        var element_count = isFirstPass ? this.start + count : count;
        var start_element = isFirstPass ? this.start : 0;
        var checkSortPipeline = this.device.createComputePipeline({
          layout: pipelineLayout,
          compute: {
            module: this.device.createShaderModule({
              label: label,
              code: checkSortSource(isFirstPass, isLastPass, this.mode)
            }),
            entryPoint: this.mode == 'reset' ? 'reset' : 'check_sort',
            constants: _objectSpread2({
              'ELEMENT_COUNT': element_count,
              'WORKGROUP_SIZE_X': this.workgroup_size.x,
              'WORKGROUP_SIZE_Y': this.workgroup_size.y
            }, this.mode != 'reset' && {
              'THREADS_PER_WORKGROUP': this.threads_per_workgroup,
              'START_ELEMENT': start_element
            })
          }
        });
        this.buffers.outputs.push(outputBuffer);
        this.pipelines.push({
          pipeline: checkSortPipeline,
          bindGroup: bindGroup
        });
        if (!isLastPass) {
          this.create_passes_recursive(outputBuffer, workgroup_count, passIndex + 1);
        }
      }
    }, {
      key: "dispatch",
      value: function dispatch(pass, dispatchSize) {
        var offset = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;
        for (var i = 0; i < this.pipelines.length; i++) {
          var _this$pipelines$i = this.pipelines[i],
            pipeline = _this$pipelines$i.pipeline,
            bindGroup = _this$pipelines$i.bindGroup;
          var dispatchIndirect = this.mode != 'reset' && (this.mode == 'full' || i < this.pipelines.length - 1);
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          if (dispatchIndirect) pass.dispatchWorkgroupsIndirect(dispatchSize, offset + i * 3 * 4);else
            // Only the reset kernel and the last dispatch of the fast check kernel are constant to (1, 1, 1)
            pass.dispatchWorkgroups(1, 1, 1);
        }
      }
    }], [{
      key: "find_optimal_dispatch_chain",
      value: function find_optimal_dispatch_chain(device, item_count, workgroup_size) {
        var threads_per_workgroup = workgroup_size.x * workgroup_size.y;
        var sizes = [];
        do {
          // Number of workgroups required to process all items
          var target_workgroup_count = Math.ceil(item_count / threads_per_workgroup);

          // Optimal dispatch size and updated workgroup count
          var dispatchSize = find_optimal_dispatch_size(device, target_workgroup_count);
          sizes.push(dispatchSize.x, dispatchSize.y, 1);
          item_count = target_workgroup_count;
        } while (item_count > 1);
        return sizes;
      }
    }]);
  }();

  var _RadixSortKernel_brand = /*#__PURE__*/new WeakSet();
  var RadixSortKernel = /*#__PURE__*/function () {
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
    function RadixSortKernel() {
      var _ref = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {},
        device = _ref.device,
        keys = _ref.keys,
        values = _ref.values,
        count = _ref.count,
        _ref$bit_count = _ref.bit_count,
        bit_count = _ref$bit_count === void 0 ? 32 : _ref$bit_count,
        _ref$workgroup_size = _ref.workgroup_size,
        workgroup_size = _ref$workgroup_size === void 0 ? {
          x: 16,
          y: 16
        } : _ref$workgroup_size,
        _ref$check_order = _ref.check_order,
        check_order = _ref$check_order === void 0 ? false : _ref$check_order,
        _ref$local_shuffle = _ref.local_shuffle,
        local_shuffle = _ref$local_shuffle === void 0 ? false : _ref$local_shuffle,
        _ref$avoid_bank_confl = _ref.avoid_bank_conflicts,
        avoid_bank_conflicts = _ref$avoid_bank_confl === void 0 ? false : _ref$avoid_bank_confl;
      _classCallCheck(this, RadixSortKernel);
      /**
       * Dispatch workgroups from CPU args
       */
      _classPrivateMethodInitSpec(this, _RadixSortKernel_brand);
      if (device == null) throw new Error('No device provided');
      if (keys == null) throw new Error('No keys buffer provided');
      if (!Number.isInteger(count) || count <= 0) throw new Error('Invalid count parameter');
      if (!Number.isInteger(bit_count) || bit_count <= 0 || bit_count > 32) throw new Error("Invalid bit_count parameter: ".concat(bit_count));
      if (!Number.isInteger(workgroup_size.x) || !Number.isInteger(workgroup_size.y)) throw new Error('Invalid workgroup_size parameter');
      if (bit_count % 4 != 0) throw new Error('bit_count must be a multiple of 4');
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
      this.has_values = values != null; // Is the values buffer provided ?

      this.dispatchSize = {}; // Dispatch dimension x and y
      this.shaderModules = {}; // GPUShaderModules
      this.kernels = {}; // PrefixSumKernel & CheckSortKernels
      this.pipelines = []; // List of passes
      this.buffers = {
        // GPUBuffers
        keys: keys,
        values: values
      };

      // Create shader modules from wgsl code
      this.create_shader_modules();

      // Create multi-pass pipelines
      this.create_pipelines();
    }
    return _createClass(RadixSortKernel, [{
      key: "create_shader_modules",
      value: function create_shader_modules() {
        // Remove every occurence of "values" in the shader code if values buffer is not provided
        var remove_values = function remove_values(source) {
          return source.split('\n').filter(function (line) {
            return !line.toLowerCase().includes('values');
          }).join('\n');
        };
        var blockSumSource = this.local_shuffle ? radixSortCoalescedSource : radixSortSource;
        this.shaderModules = {
          blockSum: this.device.createShaderModule({
            label: 'radix-sort-block-sum',
            code: this.has_values ? blockSumSource : remove_values(blockSumSource)
          }),
          reorder: this.device.createShaderModule({
            label: 'radix-sort-reorder',
            code: this.has_values ? radixSortReorderSource : remove_values(radixSortReorderSource)
          })
        };
      }
    }, {
      key: "create_pipelines",
      value: function create_pipelines() {
        // Block prefix sum kernel    
        this.create_prefix_sum_kernel();

        // Indirect dispatch buffers
        var dispatchData = this.calculate_dispatch_sizes();

        // GPU buffers
        this.create_buffers(dispatchData);

        // Check sort kernels
        this.create_check_sort_kernels(dispatchData);

        // Radix sort passes for every 2 bits
        for (var bit = 0; bit < this.bit_count; bit += 2) {
          // Swap buffers every pass
          var even = bit % 4 == 0;
          var inKeys = even ? this.buffers.keys : this.buffers.tmpKeys;
          var inValues = even ? this.buffers.values : this.buffers.tmpValues;
          var outKeys = even ? this.buffers.tmpKeys : this.buffers.keys;
          var outValues = even ? this.buffers.tmpValues : this.buffers.values;

          // Compute local prefix sums and block sums
          var blockSumPipeline = this.create_block_sum_pipeline(inKeys, inValues, bit);

          // Reorder keys and values
          var reorderPipeline = this.create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit);
          this.pipelines.push({
            blockSumPipeline: blockSumPipeline,
            reorderPipeline: reorderPipeline
          });
        }
      }
    }, {
      key: "create_prefix_sum_kernel",
      value: function create_prefix_sum_kernel() {
        // Prefix Block Sum buffer (4 element per workgroup)
        var prefixBlockSumBuffer = this.device.createBuffer({
          label: 'radix-sort-prefix-block-sum',
          size: this.prefix_block_workgroup_count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Create block prefix sum kernel
        var prefixSumKernel = new PrefixSumKernel({
          device: this.device,
          data: prefixBlockSumBuffer,
          count: this.prefix_block_workgroup_count,
          workgroup_size: this.workgroup_size,
          avoid_bank_conflicts: this.avoid_bank_conflicts
        });
        this.kernels.prefixSum = prefixSumKernel;
        this.buffers.prefixBlockSum = prefixBlockSumBuffer;
      }
    }, {
      key: "calculate_dispatch_sizes",
      value: function calculate_dispatch_sizes() {
        // Radix sort dispatch size
        var dispatchSize = find_optimal_dispatch_size(this.device, this.workgroup_count);

        // Prefix sum dispatch sizes
        var prefixSumDispatchSize = this.kernels.prefixSum.get_dispatch_chain();

        // Check sort element count (fast/full)
        var check_sort_fast_count = Math.min(this.count, this.threads_per_workgroup * 4);
        var check_sort_full_count = this.count - check_sort_fast_count;
        var start_full = check_sort_fast_count - 1;

        // Check sort dispatch sizes
        var dispatchSizesFast = CheckSortKernel.find_optimal_dispatch_chain(this.device, check_sort_fast_count, this.workgroup_size);
        var dispatchSizesFull = CheckSortKernel.find_optimal_dispatch_chain(this.device, check_sort_full_count, this.workgroup_size);

        // Initial dispatch sizes
        var initialDispatch = [dispatchSize.x, dispatchSize.y, 1].concat(_toConsumableArray(dispatchSizesFast.slice(0, 3)), _toConsumableArray(prefixSumDispatchSize));

        // Dispatch offsets in main buffer
        this.dispatchOffsets = {
          radix_sort: 0,
          check_sort_fast: 3 * 4,
          prefix_sum: 6 * 4
        };
        this.dispatchSize = dispatchSize;
        this.initialDispatch = initialDispatch;
        return {
          initialDispatch: initialDispatch,
          dispatchSizesFull: dispatchSizesFull,
          check_sort_fast_count: check_sort_fast_count,
          check_sort_full_count: check_sort_full_count,
          start_full: start_full
        };
      }
    }, {
      key: "create_buffers",
      value: function create_buffers(dispatchData) {
        // Keys and values double buffering
        var tmpKeysBuffer = this.device.createBuffer({
          label: 'radix-sort-tmp-keys',
          size: this.count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        var tmpValuesBuffer = !this.has_values ? null : this.device.createBuffer({
          label: 'radix-sort-tmp-values',
          size: this.count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Local Prefix Sum buffer (1 element per item)
        var localPrefixSumBuffer = this.device.createBuffer({
          label: 'radix-sort-local-prefix-sum',
          size: this.count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.buffers.tmpKeys = tmpKeysBuffer;
        this.buffers.tmpValues = tmpValuesBuffer;
        this.buffers.localPrefixSum = localPrefixSumBuffer;

        // Only create indirect dispatch buffers when check_order optimization is enabled
        if (!this.check_order) {
          return;
        }

        // Dispatch sizes (radix sort, check sort, prefix sum)
        var dispatchBuffer = create_buffer_from_data({
          device: this.device,
          label: 'radix-sort-dispatch-size',
          data: dispatchData.initialDispatch,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT
        });
        var originalDispatchBuffer = create_buffer_from_data({
          device: this.device,
          label: 'radix-sort-dispatch-size-original',
          data: dispatchData.initialDispatch,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Dispatch sizes (full sort)
        var checkSortFullDispatchBuffer = create_buffer_from_data({
          label: 'check-sort-full-dispatch-size',
          device: this.device,
          data: dispatchData.dispatchSizesFull,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT
        });
        var checkSortFullOriginalDispatchBuffer = create_buffer_from_data({
          label: 'check-sort-full-dispatch-size-original',
          device: this.device,
          data: dispatchData.dispatchSizesFull,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Flag to tell if the data is sorted
        var isSortedBuffer = create_buffer_from_data({
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
    }, {
      key: "create_check_sort_kernels",
      value: function create_check_sort_kernels(checkSortPartitionData) {
        if (!this.check_order) {
          return;
        }
        var check_sort_fast_count = checkSortPartitionData.check_sort_fast_count,
          check_sort_full_count = checkSortPartitionData.check_sort_full_count,
          start_full = checkSortPartitionData.start_full;

        // Create the full pass
        var checkSortFull = new CheckSortKernel({
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
        var checkSortFast = new CheckSortKernel({
          mode: 'fast',
          device: this.device,
          data: this.buffers.keys,
          result: this.buffers.checkSortFullDispatchSize,
          original: this.buffers.originalCheckSortFullDispatchSize,
          is_sorted: this.buffers.isSorted,
          count: check_sort_fast_count,
          workgroup_size: this.workgroup_size
        });
        var initialDispatchElementCount = this.initialDispatch.length / 3;
        if (checkSortFast.threads_per_workgroup < checkSortFull.pipelines.length || checkSortFull.threads_per_workgroup < initialDispatchElementCount) {
          console.warn("Warning: workgroup size is too small to enable check sort optimization, disabling...");
          this.check_order = false;
          return;
        }

        // Create the reset pass
        var checkSortReset = new CheckSortKernel({
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
          full: checkSortFull
        };
      }
    }, {
      key: "create_block_sum_pipeline",
      value: function create_block_sum_pipeline(inKeys, inValues, bit) {
        var bindGroupLayout = this.device.createBindGroupLayout({
          label: 'radix-sort-block-sum',
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: this.local_shuffle ? 'storage' : 'read-only-storage'
            }
          }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }, {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }].concat(_toConsumableArray(this.local_shuffle && this.has_values ? [{
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }] : []))
        });
        var bindGroup = this.device.createBindGroup({
          layout: bindGroupLayout,
          entries: [{
            binding: 0,
            resource: {
              buffer: inKeys
            }
          }, {
            binding: 1,
            resource: {
              buffer: this.buffers.localPrefixSum
            }
          }, {
            binding: 2,
            resource: {
              buffer: this.buffers.prefixBlockSum
            }
          }].concat(_toConsumableArray(this.local_shuffle && this.has_values ? [{
            binding: 3,
            resource: {
              buffer: inValues
            }
          }] : []))
        });
        var pipelineLayout = this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        });
        var blockSumPipeline = this.device.createComputePipeline({
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
              'CURRENT_BIT': bit
            }
          }
        });
        return {
          pipeline: blockSumPipeline,
          bindGroup: bindGroup
        };
      }
    }, {
      key: "create_reorder_pipeline",
      value: function create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit) {
        var bindGroupLayout = this.device.createBindGroupLayout({
          label: 'radix-sort-reorder',
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'read-only-storage'
            }
          }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }, {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'read-only-storage'
            }
          }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'read-only-storage'
            }
          }].concat(_toConsumableArray(this.has_values ? [{
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'read-only-storage'
            }
          }, {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: 'storage'
            }
          }] : []))
        });
        var bindGroup = this.device.createBindGroup({
          layout: bindGroupLayout,
          entries: [{
            binding: 0,
            resource: {
              buffer: inKeys
            }
          }, {
            binding: 1,
            resource: {
              buffer: outKeys
            }
          }, {
            binding: 2,
            resource: {
              buffer: this.buffers.localPrefixSum
            }
          }, {
            binding: 3,
            resource: {
              buffer: this.buffers.prefixBlockSum
            }
          }].concat(_toConsumableArray(this.has_values ? [{
            binding: 4,
            resource: {
              buffer: inValues
            }
          }, {
            binding: 5,
            resource: {
              buffer: outValues
            }
          }] : []))
        });
        var pipelineLayout = this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        });
        var reorderPipeline = this.device.createComputePipeline({
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
              'CURRENT_BIT': bit
            }
          }
        });
        return {
          pipeline: reorderPipeline,
          bindGroup: bindGroup
        };
      }

      /**
       * Encode all pipelines into the current pass
       * 
       * @param {GPUComputePassEncoder} pass 
       */
    }, {
      key: "dispatch",
      value: function dispatch(pass) {
        if (!this.check_order) {
          _assertClassBrand(_RadixSortKernel_brand, this, _dispatchPipelines).call(this, pass);
        } else {
          _assertClassBrand(_RadixSortKernel_brand, this, _dispatchPipelinesIndirect).call(this, pass);
        }
      }
    }]);
  }();
  function _dispatchPipelines(pass) {
    for (var i = 0; i < this.bit_count / 2; i++) {
      var _this$pipelines$i = this.pipelines[i],
        blockSumPipeline = _this$pipelines$i.blockSumPipeline,
        reorderPipeline = _this$pipelines$i.reorderPipeline;

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
  function _dispatchPipelinesIndirect(pass) {
    // Reset the `dispatch` and `is_sorted` buffers
    this.kernels.checkSort.reset.dispatch(pass);
    for (var i = 0; i < this.bit_count / 2; i++) {
      var _this$pipelines$i2 = this.pipelines[i],
        blockSumPipeline = _this$pipelines$i2.blockSumPipeline,
        reorderPipeline = _this$pipelines$i2.reorderPipeline;
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

  exports.PrefixSumKernel = PrefixSumKernel;
  exports.RadixSortKernel = RadixSortKernel;

}));
//# sourceMappingURL=radix-sort-umd.js.map
