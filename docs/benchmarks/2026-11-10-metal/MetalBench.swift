import Foundation
import Metal
import Accelerate

enum Failure: Error { case message(String) }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure.message(message) }
}
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-6 }
func statistics(_ input: [Double]) -> [String: Any] {
    let s = input.sorted()
    return ["median_ms": s[s.count / 2], "min_ms": s[0],
            "p95_ms": s[Int(ceil(Double(s.count - 1) * 0.95))], "samples_ms": input]
}
func readValues<T>(_ url: URL, _: T.Type) throws -> [T] {
    let d = try Data(contentsOf: url)
    try require(d.count % MemoryLayout<T>.stride == 0, "bad binary length: \(url)")
    return d.withUnsafeBytes { raw in
        (0..<(d.count / MemoryLayout<T>.stride)).map {
            raw.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.stride, as: T.self)
        }
    }
}

func main() throws {
    let args = CommandLine.arguments
    try require(args.count == 5, "MetalBench FIXTURE_DIR SHADER OUTPUT_JSON SAMPLES")
    let root = URL(fileURLWithPath: args[1])
    let samples = Int(args[4]) ?? 0
    try require(samples > 0, "samples must be positive")
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw Failure.message("no Metal device/queue")
    }
    let setupStart = now()
    let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("fixture.json"))) as! [String: Any]
    let bins = metadata["bins"] as! Int
    let count = (metadata["height"] as! Int) * (metadata["width"] as! Int)
    let rows = try readValues(root.appendingPathComponent("rows.u32"), UInt32.self)
    let pixels = try readValues(root.appendingPathComponent("pixels.u32"), UInt32.self)
    let weights = try readValues(root.appendingPathComponent("weights.f32"), Float.self)
    let source32 = try readValues(root.appendingPathComponent("signal.f32"), Float.self)
    let source64 = try readValues(root.appendingPathComponent("signal.f64"), Double.self)
    let normalization = try readValues(root.appendingPathComponent("normalization.f64"), Double.self)
    try require(rows.count == bins + 1 && rows.first == 0 && Int(rows.last!) == pixels.count,
                "invalid CSR row pointers")
    try require(zip(rows, rows.dropFirst()).allSatisfy { $0 <= $1 }, "unordered CSR rows")
    try require(pixels.allSatisfy { Int($0) < count } && weights.count == pixels.count,
                "invalid CSR indices/weights")
    try require(source32.count == count && source64.count == count && normalization.count == bins,
                "invalid signal/normalization lengths")
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        // Metal disallows zero-length buffers (all-masked tables are valid).
        if values.isEmpty {
            guard let b = device.makeBuffer(length: MemoryLayout<T>.stride, options: .storageModeShared) else {
                throw Failure.message("empty buffer allocation failed")
            }
            return b
        }
        guard let b = values.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }) else { throw Failure.message("buffer allocation failed") }
        return b
    }
    let rowBuffer = try buffer(rows), pixelBuffer = try buffer(pixels), weightBuffer = try buffer(weights)
    let signalBuffer = try buffer(source32)
    guard let resultBuffer = device.makeBuffer(length: bins * MemoryLayout<SIMD4<Float>>.stride,
                                               options: .storageModeShared) else {
        throw Failure.message("output allocation failed")
    }
    let source = try String(contentsOfFile: args[2], encoding: .utf8)
    let options = MTLCompileOptions()
    options.mathMode = .safe
    options.mathFloatingPointFunctions = .precise
    let compileStart = now()
    let library = try device.makeLibrary(source: source, options: options)
    var pipelines: [String: MTLComputePipelineState] = [:]
    for accurate in [false, true] {
      for errors in [false, true] {
        let constants = MTLFunctionConstantValues()
        var e = errors, w = metadata["weighted"] as! Bool, a = accurate
        constants.setConstantValue(&e, type: .bool, index: 0)
        constants.setConstantValue(&w, type: .bool, index: 1)
        constants.setConstantValue(&a, type: .bool, index: 2)
        let function = try library.makeFunction(name: "integrate_csr", constantValues: constants)
        pipelines["\(errors)-\(accurate)"] = try device.makeComputePipelineState(function: function)
      }
    }
    let compileMS = now() - compileStart
    let setupMS = now() - setupStart
    var intensity = [Double](repeating: 0, count: bins)
    var std = [Double](repeating: 0, count: bins)
    var checksum = 0.0
    func run(_ pipeline: MTLComputePipelineState, _ threads: Int, _ errors: Bool, _ input: String) throws -> (Double, Double, Double, Double) {
        let start = now()
        if input == "f32_copy" {
            source32.withUnsafeBytes { raw in
                signalBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
            }
        } else if input == "f64_convert" {
            source64.withUnsafeBufferPointer { p in
                vDSP_vdpsp(p.baseAddress!, 1, signalBuffer.contents().assumingMemoryBound(to: Float.self),
                          1, vDSP_Length(count))
            }
        }
        let uploadEnd = now()
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw Failure.message("command allocation failed")
        }
        encoder.setComputePipelineState(pipeline)
        for (i, b) in [rowBuffer, pixelBuffer, weightBuffer, signalBuffer, resultBuffer].enumerated() {
            encoder.setBuffer(b, offset: 0, index: i)
        }
        encoder.dispatchThreadgroups(MTLSize(width: bins, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        try require(command.status == .completed, "GPU command did not complete")
        let readStart = now()
        let raw = resultBuffer.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        for b in 0..<bins {
            let n = normalization[b]
            intensity[b] = n == 0 ? 0 : (Double(raw[b].x) + Double(raw[b].y)) / n
            if errors { std[b] = n == 0 ? 0 : sqrt(max((Double(raw[b].z) + Double(raw[b].w)) / (n * n), 0)) }
        }
        checksum += intensity[bins / 2]
        return (now() - start, (command.gpuEndTime - command.gpuStartTime) * 1000,
                uploadEnd - start, now() - readStart)
    }
    var results: [[String: Any]] = []
    for accurate in [false, true] {
      for errors in [false, true] {
        let pipeline = pipelines["\(errors)-\(accurate)"]!
        for threads in [64, 128, 256] {
            try require(threads <= pipeline.maxTotalThreadsPerThreadgroup &&
                        threads % pipeline.threadExecutionWidth == 0 &&
                        threads / pipeline.threadExecutionWidth <= 32,
                        "unsupported threadgroup shape")
            var expectedIntensity: [Double] = [], expectedStd: [Double] = []
            for input in ["resident_f32", "f32_copy", "f64_convert"] {
                if input == "resident_f32" {
                    source32.withUnsafeBytes { raw in
                        signalBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
                    }
                } else {
                    // Poison previous input with zeros. A missing/stale upload
                    // must fail validation, even when benchmarking one frame.
                    signalBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0,
                                                             count: signalBuffer.length)
                }
                let cold = try run(pipeline, threads, errors, input).0
                if input == "resident_f32" { expectedIntensity = intensity; expectedStd = std }
                else {
                    try require(intensity == expectedIntensity && (!errors || std == expectedStd),
                                "input upload/conversion changed the expected pattern")
                }
                let warmStart = now()
                repeat { _ = try run(pipeline, threads, errors, input) } while now() - warmStart < 100
                var wall: [Double] = [], gpu: [Double] = [], upload: [Double] = [], read: [Double] = []
                for _ in 0..<samples {
                    let (w, g, u, r) = try run(pipeline, threads, errors, input)
                    wall.append(w); gpu.append(g); upload.append(u); read.append(r)
                }
                var result: [String: Any] = ["errors": errors, "compensated": accurate,
                    "threads": threads, "input": input,
                    "wall": statistics(wall), "gpu": statistics(gpu), "first_call_ms": cold,
                    "input_preparation": statistics(upload), "readback_normalization": statistics(read),
                    "intensity": intensity]
                if errors { result["std"] = std }
                results.append(result)
            }
        }
      }
    }
    let output: [String: Any] = ["device": device.name, "unified_memory": device.hasUnifiedMemory,
        "setup_including_file_io_ms": setupMS, "shader_compile_ms": compileMS,
        "shared_buffer_bytes": rowBuffer.length + pixelBuffer.length + weightBuffer.length +
            signalBuffer.length + resultBuffer.length,
        "fast_math": false, "input_upload_validated": true, "checksum": checksum, "results": results]
    let json = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: URL(fileURLWithPath: args[3]))
    print("Metal: \(device.name), \(bins) bins, \(samples) samples per variant")
}
do { try main() } catch { fputs("\(error)\n", stderr); exit(1) }
