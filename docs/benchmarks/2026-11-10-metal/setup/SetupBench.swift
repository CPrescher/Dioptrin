import Foundation
import Metal
import Accelerate

@_silgen_name("setup_new") func create(_ image: UnsafePointer<CChar>, _ poni: UnsafePointer<CChar>, _ mask: UnsafePointer<CChar>, _ weighted: Bool, _ bins: Int, _ threads: Int, _ backend: UInt32) -> UnsafeMutableRawPointer?
@_silgen_name("setup_ptr") func pointer(_ context: UnsafeRawPointer, _ kind: UInt32) -> UnsafeRawPointer
@_silgen_name("setup_len") func length(_ context: UnsafeRawPointer, _ kind: UInt32) -> Int
@_silgen_name("setup_integrate") func cpuIntegrate(_ context: UnsafeRawPointer, _ intensity: UnsafeMutablePointer<Double>, _ std: UnsafeMutablePointer<Double>)
@_silgen_name("setup_reference") func reference(_ context: UnsafeRawPointer, _ intensity: UnsafeMutablePointer<Double>, _ std: UnsafeMutablePointer<Double>)
@_silgen_name("setup_free") func releaseContext(_ context: UnsafeMutableRawPointer)

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds)*1e-6 }
enum Failure: Error { case message(String) }
func require(_ value: Bool, _ message: String) throws { if !value { throw Failure.message(message) } }
func emit(_ value: [String:Any]) throws {
    var data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    data.append(10)
    FileHandle.standardOutput.write(data)
}
func stats(_ times: [Double]) -> [String:Any] {
    let s = times.sorted()
    return ["median_ms":s[s.count/2],"min_ms":s[0],"p95_ms":s[Int(ceil(Double(s.count-1)*0.95))],"samples_ms":times]
}

func main() throws {
    let start = now()
    let a = CommandLine.arguments
    try require(a.count == 13, "SetupBench BACKEND IMAGE PONI MASK METHOD BINS CPU_THREADS GPU_THREADS SAMPLES BATCH_N IDLE_SECONDS SHADER")
    let backend = UInt32(a[1])!, bins = Int(a[6])!, threads = Int(a[7])!, gpuThreads = Int(a[8])!
    let samples = Int(a[9])!, batchN = Int(a[10])!, idle = Double(a[11])!
    try require(bins > 0 && samples > 0 && threads > 0 && batchN >= 0, "invalid dimensions/counts")
    let weighted = a[5] == "pixel_split", gpu = backend == 3
    let nativeStart = now()
    let maybe = a[2].withCString { image in a[3].withCString { poni in a[4].withCString { mask in
        create(image,poni,mask,weighted,bins,threads,backend)
    } } }
    guard let context = maybe else { throw Failure.message("native setup failed") }
    defer { releaseContext(context) }
    let nativeEnd = now()
    let count = length(context,0)
    var y = [Double](repeating:0,count:bins), e = y
    var device: MTLDevice?, queue: MTLCommandQueue?, pipeline: MTLComputePipelineState?
    var buffers: [MTLBuffer] = []
    var gpuSetupMS = 0.0
    if gpu {
        let t = now()
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { throw Failure.message("no Metal device") }
        device = d; queue = q
        for kind: UInt32 in [1,2,3,5] {
            let bytes = length(context,kind)*4
            guard let b = d.makeBuffer(bytes:pointer(context,kind),length:bytes,options:.storageModeShared) else { throw Failure.message("buffer allocation failed") }
            buffers.append(b)
        }
        guard let out = d.makeBuffer(length:bins*16,options:.storageModeShared) else { throw Failure.message("result allocation failed") }
        buffers.append(out)
        let options = MTLCompileOptions(); options.mathMode = .safe; options.mathFloatingPointFunctions = .precise
        let library = try d.makeLibrary(source:String(contentsOfFile:a[12],encoding:.utf8),options:options)
        let constants = MTLFunctionConstantValues()
        var yes = true, w = weighted
        constants.setConstantValue(&yes,type:.bool,index:0)
        constants.setConstantValue(&w,type:.bool,index:1)
        constants.setConstantValue(&yes,type:.bool,index:2)
        pipeline = try d.makeComputePipelineState(function:library.makeFunction(name:"integrate_csr",constantValues:constants))
        try require(gpuThreads <= pipeline!.maxTotalThreadsPerThreadgroup && gpuThreads % pipeline!.threadExecutionWidth == 0,"invalid GPU threadgroup")
        gpuSetupMS = now()-t
    }
    let source64 = pointer(context,0).assumingMemoryBound(to:Double.self)
    let source32 = gpu ? pointer(context,5) : nil
    let norm = gpu ? pointer(context,4).assumingMemoryBound(to:Double.self) : nil
    var checksum = 0.0
    func run(_ input: String = "f64_convert") throws -> Double {
        let t = now()
        if !gpu {
            y.withUnsafeMutableBufferPointer { yy in e.withUnsafeMutableBufferPointer { ee in
                cpuIntegrate(context,yy.baseAddress!,ee.baseAddress!)
            } }
        } else {
            if input == "f64_convert" { vDSP_vdpsp(source64,1,buffers[3].contents().assumingMemoryBound(to:Float.self),1,vDSP_Length(count)) }
            else if input == "f32_copy" { buffers[3].contents().copyMemory(from:source32!,byteCount:count*4) }
            guard let c = queue!.makeCommandBuffer(), let enc = c.makeComputeCommandEncoder() else { throw Failure.message("command allocation failed") }
            enc.setComputePipelineState(pipeline!)
            for (i,b) in buffers.enumerated() { enc.setBuffer(b,offset:0,index:i) }
            enc.dispatchThreadgroups(MTLSize(width:bins,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:gpuThreads,height:1,depth:1))
            enc.endEncoding(); c.commit(); c.waitUntilCompleted()
            if let error = c.error { throw error }
            try require(c.status == .completed,"command failed")
            let raw = buffers[4].contents().assumingMemoryBound(to:SIMD4<Float>.self)
            for b in 0..<bins {
                let n = norm![b]
                y[b] = n == 0 ? 0 : (Double(raw[b].x)+Double(raw[b].y))/n
                e[b] = n == 0 ? 0 : sqrt(max((Double(raw[b].z)+Double(raw[b].w))/(n*n),0))
            }
        }
        checksum += y[bins/2]
        return now()-t
    }
    let preparationMS = now()-start
    let firstFrameMS = try run()
    let firstInternalMS = now()-start
    let firstY = y, firstE = e
    try emit(["kind":"first","internal_ms":firstInternalMS,"first_frame_call_ms":firstFrameMS,
              "preparation_before_first_call_ms":preparationMS,"native_setup_ms":nativeEnd-nativeStart,"metal_setup_ms":gpuSetupMS])
    var results: [[String:Any]] = []
    if batchN > 0 {
        if batchN > 1 { for _ in 1..<batchN { _ = try run() } }
        try emit(["kind":"batch","n":batchN,"internal_ms":now()-start])
    } else {
        for mode in gpu ? ["f64_convert","f32_copy","resident_f32"] : ["f64_convert"] {
            let warm = now()
            repeat { _ = try run(mode) } while now()-warm < 150
            var times: [Double] = []
            for _ in 0..<samples { times.append(try run(mode)) }
            results.append(["input":mode,"timing":stats(times)])
        }
    }
    var idleResults: [[String:Any]] = []
    if idle > 0 {
        for cycle in 0..<3 {
            let warm = now()
            repeat { _ = try run() } while now()-warm < 150
            try emit(["kind":"idle_start","cycle":cycle,"seconds":idle])
            let idleStart = now()
            Thread.sleep(forTimeInterval:idle)
            let slept = now()-idleStart
            let firstAfter = try run()
            var next: [Double] = []
            for _ in 0..<10 { next.append(try run()) }
            idleResults.append(["sleep_ms":slept,"first_after_idle_ms":firstAfter,"next_10":stats(next)])
            try emit(["kind":"idle_end","cycle":cycle,"first_after_idle_ms":firstAfter])
        }
    }
    var maxRel = 0.0, maxInStd = 0.0, maxStdRel = 0.0
    if gpu {
        var refY = y, refE = e
        refY.withUnsafeMutableBufferPointer { yy in refE.withUnsafeMutableBufferPointer { ee in reference(context,yy.baseAddress!,ee.baseAddress!) } }
        for (yy,ee) in [(firstY,firstE),(y,e)] {
            for b in 0..<bins {
                let delta = abs(yy[b]-refY[b]), de = abs(ee[b]-refE[b])
                try require(yy[b].isFinite && ee[b].isFinite && delta <= 1e-5+1e-5*abs(refY[b]) && de <= 1e-5+1e-5*abs(refE[b]),"accuracy failed")
                if refY[b] != 0 { maxRel = max(maxRel,delta/abs(refY[b])) }
                if refE[b] > 0 { maxInStd = max(maxInStd,delta/refE[b]); maxStdRel = max(maxStdRel,de/refE[b]) }
            }
        }
        try require(maxInStd < 0.01,"accuracy exceeds 0.01 std")
    }
    try emit(["kind":"result","backend":backend,"device":device?.name ?? "CPU","first_internal_ms":firstInternalMS,
              "steady":results,"idle":idleResults,"checksum":checksum,"accuracy_passed":true,
              "max_relative_intensity_error":maxRel,"max_intensity_error_in_std":maxInStd,"max_relative_std_error":maxStdRel])
}
do { try main() } catch { fputs("\(error)\n",stderr); exit(1) }
