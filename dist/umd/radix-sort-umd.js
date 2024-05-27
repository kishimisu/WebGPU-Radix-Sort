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
  function _classCallCheck(a, n) {
    if (!(a instanceof n)) throw new TypeError("Cannot call a class as a function");
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
  function _createForOfIteratorHelper(r, e) {
    var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"];
    if (!t) {
      if (Array.isArray(r) || (t = _unsupportedIterableToArray(r)) || e && r && "number" == typeof r.length) {
        t && (r = t);
        var n = 0,
          F = function () {};
        return {
          s: F,
          n: function () {
            return n >= r.length ? {
              done: !0
            } : {
              done: !1,
              value: r[n++]
            };
          },
          e: function (r) {
            throw r;
          },
          f: F
        };
      }
      throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
    }
    var o,
      a = !0,
      u = !1;
    return {
      s: function () {
        t = t.call(r);
      },
      n: function () {
        var r = t.next();
        return a = r.done, r;
      },
      e: function (r) {
        u = !0, o = r;
      },
      f: function () {
        try {
          a || null == t.return || t.return();
        } finally {
          if (u) throw o;
        }
      }
    };
  }
  function _iterableToArray(r) {
    if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
  }
  function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
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

  var prefixSumSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read_write> items: array<u32>;\n@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;\n\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride ITEMS_PER_WORKGROUP: u32;\n\nvar<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn reduce_downsweep(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n    \n    let ELM_TID = TID * 2; // Element pair local ID\n    let ELM_GID = GID * 2; // Element pair global ID\n    \n    // Load input to shared memory\n    temp[ELM_TID]     = items[ELM_GID];\n    temp[ELM_TID + 1] = items[ELM_GID + 1];\n\n    var offset: u32 = 1;\n\n    // Up-sweep (reduce) phase\n    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n            temp[bi] += temp[ai];\n        }\n\n        offset *= 2;\n    }\n\n    // Save workgroup sum and clear last element\n    if (TID == 0) {\n        let last_offset = ITEMS_PER_WORKGROUP - 1;\n\n        blockSums[WORKGROUP_ID] = temp[last_offset];\n        temp[last_offset] = 0;\n    }\n\n    // Down-sweep phase\n    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {\n        offset >>= 1;\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n\n            let t: u32 = temp[ai];\n            temp[ai] = temp[bi];\n            temp[bi] += t;\n        }\n    }\n    workgroupBarrier();\n\n    // Copy result from shared memory to global memory\n    items[ELM_GID]     = temp[ELM_TID];\n    items[ELM_GID + 1] = temp[ELM_TID + 1];\n}\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn add_block_sums(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n    \n\n    let ELM_ID = GID * 2;\n    let blockSum = blockSums[WORKGROUP_ID];\n\n    items[ELM_ID] += blockSum;\n    items[ELM_ID + 1] += blockSum;\n}";

  /**
   * Prefix sum with optimization to avoid bank conflicts
   * 
   * (see Implementation section in README for details)
   */
  var prefixSumNoBankConflictSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read_write> items: array<u32>;\n@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;\n\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride ITEMS_PER_WORKGROUP: u32;\n\nconst NUM_BANKS: u32 = 32;\nconst LOG_NUM_BANKS: u32 = 5;\n\nfn get_offset(offset: u32) -> u32 {\n    // return offset >> LOG_NUM_BANKS; // Conflict-free\n    return (offset >> NUM_BANKS) + (offset >> (2 * LOG_NUM_BANKS)); // Zero bank conflict\n}\n\nvar<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn reduce_downsweep(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n    \n    let ELM_TID = TID * 2; // Element pair local ID\n    let ELM_GID = GID * 2; // Element pair global ID\n    \n    // Load input to shared memory\n    let ai: u32 = TID;\n    let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1);\n    let s_ai = ai + get_offset(ai);\n    let s_bi = bi + get_offset(bi);\n    let g_ai = ai + WID * 2;\n    let g_bi = bi + WID * 2;\n    temp[s_ai] = items[g_ai];\n    temp[s_bi] = items[g_bi];\n\n    var offset: u32 = 1;\n\n    // Up-sweep (reduce) phase\n    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n            ai += get_offset(ai);\n            bi += get_offset(bi);\n            temp[bi] += temp[ai];\n        }\n\n        offset *= 2;\n    }\n\n    // Save workgroup sum and clear last element\n    if (TID == 0) {\n        var last_offset = ITEMS_PER_WORKGROUP - 1;\n        last_offset += get_offset(last_offset);\n\n        blockSums[WORKGROUP_ID] = temp[last_offset];\n        temp[last_offset] = 0;\n    }\n\n    // Down-sweep phase\n    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {\n        offset >>= 1;\n        workgroupBarrier();\n\n        if (TID < d) {\n            var ai: u32 = offset * (ELM_TID + 1) - 1;\n            var bi: u32 = offset * (ELM_TID + 2) - 1;\n            ai += get_offset(ai);\n            bi += get_offset(bi);\n\n            let t: u32 = temp[ai];\n            temp[ai] = temp[bi];\n            temp[bi] += t;\n        }\n    }\n    workgroupBarrier();\n\n    // Copy result from shared memory to global memory\n    items[g_ai] = temp[s_ai];\n    items[g_bi] = temp[s_bi];\n}\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn add_block_sums(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    let ELM_ID = GID * 2;\n    let blockSum = blockSums[WORKGROUP_ID];\n\n    items[ELM_ID] += blockSum;\n    items[ELM_ID + 1] += blockSum;\n}";

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
      key: "find_optimal_dispatch_size",
      value: function find_optimal_dispatch_size(item_count) {
        var maxComputeWorkgroupsPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        var workgroup_count = Math.ceil(item_count / this.items_per_workgroup);
        var x = workgroup_count;
        var y = 1;
        if (workgroup_count > maxComputeWorkgroupsPerDimension) {
          x = Math.floor(Math.sqrt(workgroup_count));
          y = Math.ceil(workgroup_count / x);
          workgroup_count = x * y;
        }
        return {
          workgroup_count: workgroup_count,
          dispatchSize: {
            x: x,
            y: y
          }
        };
      }
    }, {
      key: "create_pass_recursive",
      value: function create_pass_recursive(data, count) {
        // Find best dispatch x and y dimensions to minimize unused threads
        var _this$find_optimal_di = this.find_optimal_dispatch_size(count),
          workgroup_count = _this$find_optimal_di.workgroup_count,
          dispatchSize = _this$find_optimal_di.dispatchSize;

        // Create buffer for block sums        
        var blockSumBuffer = this.device.createBuffer({
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
              'ITEMS_PER_WORKGROUP': this.items_per_workgroup
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
                'THREADS_PER_WORKGROUP': this.threads_per_workgroup
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
      key: "dispatch",
      value: function dispatch(pass) {
        var _iterator = _createForOfIteratorHelper(this.pipelines),
          _step;
        try {
          for (_iterator.s(); !(_step = _iterator.n()).done;) {
            var _step$value = _step.value,
              pipeline = _step$value.pipeline,
              bindGroup = _step$value.bindGroup,
              dispatchSize = _step$value.dispatchSize;
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(dispatchSize.x, dispatchSize.y, 1);
          }
        } catch (err) {
          _iterator.e(err);
        } finally {
          _iterator.f();
        }
      }
    }]);
  }();

  var radixSortSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read> input: array<u32>;\n@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;\n@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;\n\noverride WORKGROUP_COUNT: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride CURRENT_BIT: u32;\noverride ELEMENT_COUNT: u32;\n\nvar<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn radix_sort(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    // Extract 2 bits from the input\n    let elm = input[GID];\n    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;\n\n    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);\n\n    // If the workgroup is inactive, prevent block_sums buffer update\n    var LAST_THREAD: u32 = 0xffffffff; \n\n    if (WORKGROUP_ID < WORKGROUP_COUNT) {\n        // Otherwise store the index of the last active thread in the workgroup\n        LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;\n    }\n\n    // Initialize parameters for double-buffering\n    let TPW = THREADS_PER_WORKGROUP + 1;\n    var swapOffset: u32 = 0;\n    var inOffset:  u32 = TID;\n    var outOffset: u32 = TID + TPW;\n\n    // 4-way prefix sum\n    for (var b: u32 = 0; b < 4; b++) {\n        // Initialize local prefix with bitmask\n        let bitmask = select(0u, 1u, extract_bits == b);\n        s_prefix_sum[inOffset + 1] = bitmask;\n        workgroupBarrier();\n\n        // Prefix sum\n        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {\n            if (TID >= offset) {\n                s_prefix_sum[outOffset] = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];\n            } else {\n                s_prefix_sum[outOffset] = s_prefix_sum[inOffset];\n            }\n\n            // Swap buffers\n            outOffset = inOffset;\n            swapOffset = TPW - swapOffset;\n            inOffset = TID + swapOffset;\n            \n            workgroupBarrier();\n        }\n\n        // Store prefix sum for current bit\n        let prefix_sum = s_prefix_sum[inOffset];\n        bit_prefix_sums[b] = prefix_sum;\n\n        if (TID == LAST_THREAD) {\n            // Store block sum to global memory\n            let total_sum: u32 = prefix_sum + bitmask;\n            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;\n        }\n\n        // Swap buffers\n        outOffset = inOffset;\n        swapOffset = TPW - swapOffset;\n        inOffset = TID + swapOffset;\n    }\n\n    // Store local prefix sum to global memory\n    local_prefix_sums[GID] = bit_prefix_sums[extract_bits];\n}";

  /**
   * Radix sort with "local shuffle and coalesced mapping" optimization
   * 
   * (see Implementation section in README for details)
   */
  var radixSortCoalescedSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read_write> input: array<u32>;\n@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;\n@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;\n@group(0) @binding(3) var<storage, read_write> values: array<u32>;\n\noverride WORKGROUP_COUNT: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride CURRENT_BIT: u32;\noverride ELEMENT_COUNT: u32;\n\nvar<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;\nvar<workgroup> s_prefix_sum_scan: array<u32, 4>;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn radix_sort(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) {\n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    // Extract 2 bits from the input\n    let elm = input[GID];\n    let val = values[GID];\n    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;\n\n    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);\n\n    // If the workgroup is inactive, prevent block_sums buffer update\n    var LAST_THREAD: u32 = 0xffffffff; \n\n    if (WORKGROUP_ID < WORKGROUP_COUNT) {\n        // Otherwise store the index of the last active thread in the workgroup\n        LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;\n    }\n\n    // Initialize parameters for double-buffering\n    let TPW = THREADS_PER_WORKGROUP + 1;\n    var swapOffset: u32 = 0;\n    var inOffset:  u32 = TID;\n    var outOffset: u32 = TID + TPW;\n\n    // 4-way prefix sum\n    for (var b: u32 = 0; b < 4; b++) {\n        // Initialize local prefix with bitmask\n        let bitmask = select(0u, 1u, extract_bits == b);\n        s_prefix_sum[inOffset + 1] = bitmask;\n        workgroupBarrier();\n\n        // Prefix sum\n        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {\n            if (TID >= offset) {\n                s_prefix_sum[outOffset] = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];\n            } else {\n                s_prefix_sum[outOffset] = s_prefix_sum[inOffset];\n            }\n\n            // Swap buffers\n            outOffset = inOffset;\n            swapOffset = TPW - swapOffset;\n            inOffset = TID + swapOffset;\n            \n            workgroupBarrier();\n        }\n\n        // Store prefix sum for current bit\n        let prefix_sum = s_prefix_sum[inOffset];\n        bit_prefix_sums[b] = prefix_sum;\n\n        if (TID == LAST_THREAD) {\n            // Store block sum to global memory\n            let total_sum: u32 = prefix_sum + bitmask;\n            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;\n        }\n\n        // Swap buffers\n        outOffset = inOffset;\n        swapOffset = TPW - swapOffset;\n        inOffset = TID + swapOffset;\n    }\n\n    let prefix_sum = bit_prefix_sums[extract_bits];   \n\n    // Scan bit prefix sums\n    if (TID == LAST_THREAD) {\n        var sum: u32 = 0;\n        bit_prefix_sums[extract_bits] += 1;\n        for (var i: u32 = 0; i < 4; i++) {\n            s_prefix_sum_scan[i] = sum;\n            sum += bit_prefix_sums[i];\n        }\n    }\n    workgroupBarrier();\n\n    if (GID < ELEMENT_COUNT) {\n        // Compute new position\n        let new_pos: u32 = prefix_sum + s_prefix_sum_scan[extract_bits];\n\n        // Shuffle elements locally\n        input[WID + new_pos] = elm;\n        values[WID + new_pos] = val;\n        local_prefix_sums[WID + new_pos] = prefix_sum;\n    }\n}";

  var radixSortReorderSource = /* wgsl */"\n\n@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;\n@group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;\n@group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;\n@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;\n@group(0) @binding(4) var<storage, read> inputValues: array<u32>;\n@group(0) @binding(5) var<storage, read_write> outputValues: array<u32>;\n\noverride WORKGROUP_COUNT: u32;\noverride THREADS_PER_WORKGROUP: u32;\noverride WORKGROUP_SIZE_X: u32;\noverride WORKGROUP_SIZE_Y: u32;\noverride CURRENT_BIT: u32;\noverride ELEMENT_COUNT: u32;\n\n@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)\nfn radix_sort_reorder(\n    @builtin(workgroup_id) w_id: vec3<u32>,\n    @builtin(num_workgroups) w_dim: vec3<u32>,\n    @builtin(local_invocation_index) TID: u32, // Local thread ID\n) { \n    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;\n    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;\n    let GID = WID + TID; // Global thread ID\n\n    if (GID >= ELEMENT_COUNT) {\n        return;\n    }\n\n    let k = inputKeys[GID];\n    let v = inputValues[GID];\n\n    let local_prefix = local_prefix_sum[GID];\n\n    // Calculate new position\n    let extract_bits = (k >> CURRENT_BIT) & 0x3;\n    let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;\n    let sorted_position = prefix_block_sum[pid] + local_prefix;\n    \n    outputKeys[sorted_position] = k;\n    outputValues[sorted_position] = v;\n}";

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
        _ref$local_shuffle = _ref.local_shuffle,
        local_shuffle = _ref$local_shuffle === void 0 ? false : _ref$local_shuffle,
        _ref$avoid_bank_confl = _ref.avoid_bank_conflicts,
        avoid_bank_conflicts = _ref$avoid_bank_confl === void 0 ? false : _ref$avoid_bank_confl;
      _classCallCheck(this, RadixSortKernel);
      if (device == null) throw new Error('No device provided');
      if (keys == null) throw new Error('No keys buffer provided');
      if (!Number.isInteger(count) || count <= 0) throw new Error('Invalid count parameter');
      if (!Number.isInteger(bit_count) || bit_count <= 0) throw new Error('Invalid bit_count parameter');
      if (!Number.isInteger(workgroup_size.x) || !Number.isInteger(workgroup_size.y)) throw new Error('Invalid workgroup_size parameter');
      this.device = device;
      this.count = count;
      this.bit_count = bit_count;
      this.workgroup_size = workgroup_size;
      this.local_shuffle = local_shuffle;
      this.avoid_bank_conflicts = avoid_bank_conflicts;
      this.threads_per_workgroup = workgroup_size.x * workgroup_size.y;
      this.workgroup_count = Math.ceil(count / this.threads_per_workgroup);
      this.prefix_block_workgroup_count = 4 * this.workgroup_count;
      this.has_values = values != null; // Is the values buffer provided ?

      this.dispatchSize = {}; // Dispatch dimension x and y
      this.shaderModules = {}; // GPUShaderModules
      this.buffers = {}; // GPUBuffers
      this.pipelines = []; // List of passes

      // Find best dispatch x and y dimensions to minimize unused threads
      this.find_optimal_dispatch_size();

      // Create shader modules from wgsl code
      this.create_shader_modules();

      // Create GPU buffers
      this.create_buffers(keys, values);

      // Create multi-pass pipelines
      this.create_pipelines();
    }
    return _createClass(RadixSortKernel, [{
      key: "find_optimal_dispatch_size",
      value: function find_optimal_dispatch_size() {
        var maxComputeWorkgroupsPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        this.dispatchSize = {
          x: this.workgroup_count,
          y: 1
        };
        if (this.workgroup_count > maxComputeWorkgroupsPerDimension) {
          var x = Math.floor(Math.sqrt(this.workgroup_count));
          var y = Math.ceil(this.workgroup_count / x);
          this.dispatchSize = {
            x: x,
            y: y
          };
        }
      }
    }, {
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
      key: "create_buffers",
      value: function create_buffers(keys, values) {
        // Keys and values double buffering
        var tmpKeysBuffer = this.device.createBuffer({
          size: this.count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        var tmpValuesBuffer = !this.has_values ? null : this.device.createBuffer({
          size: this.count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Local Prefix Sum buffer (1 element per item)
        var localPrefixSumBuffer = this.device.createBuffer({
          size: this.count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Prefix Block Sum buffer (4 element per workgroup)
        var prefixBlockSumBuffer = this.device.createBuffer({
          size: this.prefix_block_workgroup_count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.buffers = {
          keys: keys,
          values: values,
          tmpKeys: tmpKeysBuffer,
          tmpValues: tmpValuesBuffer,
          localPrefixSum: localPrefixSumBuffer,
          prefixBlockSum: prefixBlockSumBuffer
        };
      }

      // Create radix sort passes for every 2 bits
    }, {
      key: "create_pipelines",
      value: function create_pipelines() {
        for (var bit = 0; bit < this.bit_count; bit += 2) {
          // Swap buffers every pass
          var even = bit % 4 == 0;
          var inKeys = even ? this.buffers.keys : this.buffers.tmpKeys;
          var inValues = even ? this.buffers.values : this.buffers.tmpValues;
          var outKeys = even ? this.buffers.tmpKeys : this.buffers.keys;
          var outValues = even ? this.buffers.tmpValues : this.buffers.values;

          // Compute local prefix sums and block sums
          var blockSumPipeline = this.create_block_sum_pipeline(inKeys, inValues, bit);

          // Compute block sums prefix sums
          var prefixSumKernel = new PrefixSumKernel({
            device: this.device,
            data: this.buffers.prefixBlockSum,
            count: this.prefix_block_workgroup_count,
            workgroup_size: this.workgroup_size,
            avoid_bank_conflicts: this.avoid_bank_conflicts
          });

          // Reorder keys and values
          var reorderPipeline = this.create_reorder_pipeline(inKeys, inValues, outKeys, outValues, bit);
          this.pipelines.push({
            blockSumPipeline: blockSumPipeline,
            prefixSumKernel: prefixSumKernel,
            reorderPipeline: reorderPipeline
          });
        }
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
        var _iterator = _createForOfIteratorHelper(this.pipelines),
          _step;
        try {
          for (_iterator.s(); !(_step = _iterator.n()).done;) {
            var _step$value = _step.value,
              blockSumPipeline = _step$value.blockSumPipeline,
              prefixSumKernel = _step$value.prefixSumKernel,
              reorderPipeline = _step$value.reorderPipeline;
            pass.setPipeline(blockSumPipeline.pipeline);
            pass.setBindGroup(0, blockSumPipeline.bindGroup);
            pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1);
            prefixSumKernel.dispatch(pass);
            pass.setPipeline(reorderPipeline.pipeline);
            pass.setBindGroup(0, reorderPipeline.bindGroup);
            pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1);
          }
        } catch (err) {
          _iterator.e(err);
        } finally {
          _iterator.f();
        }
      }
    }]);
  }();

  exports.PrefixSumKernel = PrefixSumKernel;
  exports.RadixSortKernel = RadixSortKernel;

}));
//# sourceMappingURL=radix-sort-umd.js.map
