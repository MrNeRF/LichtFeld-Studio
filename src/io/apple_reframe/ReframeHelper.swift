/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
import Foundation
import CoreVideo
import CoreImage
import AlchemistBase

private func failure(_ message: String) -> NSError {
    NSError(domain: "LichtFeld.Reframe", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

private final class Cancellation: Cancellable {
    var isCancelled = false
    func cancel() { isCancelled = true }
    func checkCancellation() throws { if isCancelled { throw failure("Reconstruction cancelled") } }
}

private func imageBuffer(_ url: URL) throws -> CVPixelBuffer {
    guard let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true, .expandToHDR: false, .toneMapHDRtoSDR: true]),
          let srgb = CGColorSpace(name: CGColorSpace.sRGB),
          let working = CGColorSpace(name: CGColorSpace.extendedLinearSRGB) else {
        throw failure("Cannot decode this photo")
    }
    let extent = image.extent.integral
    guard extent.width.isFinite, extent.height.isFinite, extent.width > 0, extent.height > 0,
          extent.width <= 32768, extent.height <= 32768 else { throw failure("Invalid photo dimensions") }
    // Bound decoded input memory; the model still selects its own output size.
    let factor = min(1.0, 4096.0 / max(extent.width, extent.height))
    let width = max(1, Int(extent.width * factor)), height = max(1, Int(extent.height * factor))
    var pixelBuffer: CVPixelBuffer?
    let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &pixelBuffer) == kCVReturnSuccess,
          let pixelBuffer else { throw failure("Cannot allocate photo buffer") }
    let context = CIContext(options: [.workingColorSpace: working, .workingFormat: CIFormat.RGBAh])
    let transformed = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
        .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / extent.width, y: CGFloat(height) / extent.height))
    context.render(transformed, to: pixelBuffer, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: srgb)
    CVBufferSetAttachment(pixelBuffer, kCVImageBufferCGColorSpaceKey, srgb, .shouldPropagate)
    return pixelBuffer
}

private func packedHalfBuffer(_ buffer: CVPixelBuffer) throws -> Data {
    guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_OneComponent16Half,
          !CVPixelBufferIsPlanar(buffer), CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
        throw failure("Unsupported Reframe buffer layout")
    }
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw failure("Missing Reframe buffer") }
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    guard width > 0, height > 0, width <= 100_000_000 / height, stride >= width * 2 else {
        throw failure("Invalid Reframe buffer extent")
    }
    var data = Data(capacity: width * height * 2)
    for row in 0..<height { data.append(base.advanced(by: row * stride).assumingMemoryBound(to: UInt8.self), count: width * 2) }
    return data
}

@main
struct ReframeHelper {
    static func main() {
        do {
            guard #available(macOS 27.0, *), ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 27 else {
                throw failure("Reframe requires macOS 27 or later")
            }
            let models = LFSReframeModels()
            guard let joint = models["joint"], let fov = models["fov"] else {
                throw failure("Apple Reframe models are missing or ambiguous. Open Photos, use its spatial-photo feature and let the model download finish.")
            }
            let checking = CommandLine.arguments == [CommandLine.arguments[0], "--check"]
            guard checking || CommandLine.arguments.count == 3 else { throw failure("Usage: lfs-reframe photo output-file") }
            let pipeline = ALCBasePipeline()
            try pipeline.createAndLoadJointPredictor(URL(fileURLWithPath: joint))
            try pipeline.createAndLoadFoVPredictor(URL(fileURLWithPath: fov))
            if checking { print("ready"); return }
            let input = try imageBuffer(URL(fileURLWithPath: CommandLine.arguments[1]))
            let (buffers, _) = try pipeline.generateSplats(input, focalLengthPx: nil, canceller: Cancellation()) { _ in }
            let arrays = try [buffers.positions, buffers.rotations, buffers.scales, buffers.colors, buffers.alphas].map(packedHalfBuffer)
            let count = arrays[4].count / 2
            guard count > 0, count <= 10_000_000, arrays[4].count == count * 2,
                  arrays[0].count == count * 6, arrays[1].count == count * 8,
                  arrays[2].count == count * 6, arrays[3].count == count * 6 else {
                throw failure("Incompatible Reframe Gaussian buffers")
            }
            // Versioned half-float interchange. This is an IPC payload, never a
            // user-facing splat format. SplatData conversion happens in the app.
            var output = Data("LFSRFR02".utf8)
            var littleCount = UInt64(count).littleEndian
            withUnsafeBytes(of: &littleCount) { output.append(contentsOf: $0) }
            var width = UInt32(CVPixelBufferGetWidth(input)).littleEndian
            var height = UInt32(CVPixelBufferGetHeight(input)).littleEndian
            withUnsafeBytes(of: &width) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &height) { output.append(contentsOf: $0) }
            for array in arrays { output.append(array) }
            try output.write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
        } catch {
            fputs("\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
