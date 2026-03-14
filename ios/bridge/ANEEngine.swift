// ANEEngine.swift — Swift wrapper around ane_bridge C API
// Provides safe, idiomatic Swift access to ANE inference

import Foundation

enum ANEError: Error, LocalizedError {
    case initFailed
    case compileFailed
    case evalFailed
    case invalidKernel

    var errorDescription: String? {
        switch self {
        case .initFailed: return "Failed to initialize ANE runtime"
        case .compileFailed: return "Failed to compile ANE kernel"
        case .evalFailed: return "Failed to evaluate ANE kernel"
        case .invalidKernel: return "Invalid or freed ANE kernel handle"
        }
    }
}

/// Compiled ANE kernel ready for evaluation.
class ANEKernel {
    let handle: OpaquePointer

    init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        ane_bridge_free(handle)
    }

    /// Write fp16 data to an input tensor.
    func writeInput(index: Int, data: UnsafeRawPointer, bytes: Int) {
        ane_bridge_write_input(handle, Int32(index), data, bytes)
    }

    /// Read fp16 data from an output tensor.
    func readOutput(index: Int, into buffer: UnsafeMutableRawPointer, bytes: Int) {
        ane_bridge_read_output(handle, Int32(index), buffer, bytes)
    }

    /// Run the kernel on ANE hardware.
    func evaluate() throws {
        guard ane_bridge_eval(handle) else {
            throw ANEError.evalFailed
        }
    }
}

/// ANE runtime engine — manages initialization and kernel compilation.
class ANEEngine {
    static let shared = ANEEngine()

    private var initialized = false

    private init() {}

    /// Initialize the ANE runtime. Safe to call multiple times.
    func initialize() throws {
        guard !initialized else { return }
        let result = ane_bridge_init()
        guard result == 0 else {
            throw ANEError.initFailed
        }
        initialized = true
    }

    /// Compile a MIL program with optional weights into an ANE kernel.
    func compile(
        mil: String,
        weights: [(name: String, data: Data)] = [],
        inputSizes: [Int],
        outputSizes: [Int]
    ) throws -> ANEKernel {
        try initialize()

        let milData = mil.data(using: .utf8)!

        let handle: OpaquePointer?

        if weights.isEmpty {
            handle = milData.withUnsafeBytes { milPtr in
                var inSizes = inputSizes.map { $0 }
                var outSizes = outputSizes.map { $0 }
                return ane_bridge_compile(
                    milPtr.baseAddress?.assumingMemoryBound(to: CChar.self),
                    milData.count,
                    nil, 0,
                    Int32(inputSizes.count), &inSizes,
                    Int32(outputSizes.count), &outSizes
                )
            }
        } else {
            // Single weight support via ane_bridge_compile
            // (multi-weight via ane_bridge_compile_multi_weights is used from ObjC/C side)
            let combined = weights[0].data
            handle = milData.withUnsafeBytes { milPtr in
                combined.withUnsafeBytes { wPtr in
                    var inSizes = inputSizes.map { $0 }
                    var outSizes = outputSizes.map { $0 }
                    return ane_bridge_compile(
                        milPtr.baseAddress?.assumingMemoryBound(to: CChar.self),
                        milData.count,
                        wPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        combined.count,
                        Int32(inputSizes.count), &inSizes,
                        Int32(outputSizes.count), &outSizes
                    )
                }
            }
        }

        guard let kernel = handle else {
            throw ANEError.compileFailed
        }

        return ANEKernel(handle: kernel)
    }

    /// Build an ANE weight blob from fp32 data.
    func buildWeightBlob(from data: [Float], rows: Int, cols: Int) -> Data {
        var outLen: Int = 0
        let ptr = ane_bridge_build_weight_blob(data, Int32(rows), Int32(cols), &outLen)!
        let result = Data(bytes: ptr, count: outLen)
        ane_bridge_free_blob(ptr)
        return result
    }

    /// Current compile count (for budgeting against the ~239 compile limit).
    var compileCount: Int {
        Int(ane_bridge_get_compile_count())
    }
}
