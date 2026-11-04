import ARKit
import CoreImage
import UIKit

/// Runs ARKit with LiDAR depth and sends about 10 frames a second to the Mac over a WebSocket.
///
/// Each packet is: 4-byte header length (little endian), a JSON header, the colour JPEG,
/// depth as float16 metres, then confidence as one byte per depth pixel (0, 1 or 2).
/// src/stream.py reads exactly this layout.
final class Streamer: NSObject, ObservableObject, ARSessionDelegate {
    @Published var running = false
    @Published var status = "Not started"

    private let session = ARSession()
    private let ciContext = CIContext()
    private var socket: URLSessionWebSocketTask?
    private var lastSent: TimeInterval = 0
    private var sending = false
    private var sentCount = 0
    private var countStart = Date()

    func start(host: String) {
        socket = URLSession.shared.webSocketTask(with: URL(string: "ws://\(host):8765")!)
        socket?.resume()
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = .sceneDepth
        // A video format that also allows full-resolution stills, for reading labels.
        if let format = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = format
        }
        session.delegate = self
        session.run(config)
        UIApplication.shared.isIdleTimerDisabled = true  // keep the screen awake while streaming
        running = true
        status = "Connecting to \(host)"
        listen()
    }

    /// Wait for messages from the Mac. "photo" asks for one full-resolution still.
    private func listen() {
        socket?.receive { [weak self] result in
            DispatchQueue.main.async {
                guard let self, self.running, case .success(let message) = result else { return }
                if case .string("photo") = message {
                    self.capturePhoto()
                }
                self.listen()
            }
        }
    }

    private func capturePhoto() {
        session.captureHighResolutionFrame { [weak self] frame, error in
            DispatchQueue.main.async {
                guard let self else { return }
                guard let frame else {
                    self.status = "Photo failed: \(error?.localizedDescription ?? "unknown")"
                    return
                }
                let packet = self.makePacket(frame: frame, depth: nil, kind: "photo", quality: 0.9)
                self.socket?.send(.data(packet)) { _ in }
            }
        }
    }

    func stop() {
        session.pause()
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        UIApplication.shared.isIdleTimerDisabled = false
        running = false
        status = "Stopped"
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // ARKit delivers 60 frames a second. Send at most 10, and never while the last one is still going out.
        guard running, !sending, frame.timestamp - lastSent >= 0.1, let depth = frame.sceneDepth else { return }
        lastSent = frame.timestamp
        sending = true
        let packet = makePacket(frame: frame, depth: depth, kind: "stream", quality: 0.7)
        socket?.send(.data(packet)) { [weak self] error in
            DispatchQueue.main.async { self?.sent(error: error) }
        }
    }

    private func sent(error: Error?) {
        sending = false
        if let error {
            status = "Error: \(error.localizedDescription)"
            return
        }
        sentCount += 1
        let elapsed = Date().timeIntervalSince(countStart)
        if elapsed >= 1 {
            status = String(format: "Sending %.1f frames a second", Double(sentCount) / elapsed)
            sentCount = 0
            countStart = Date()
        }
    }

    /// A "stream" packet carries depth. A "photo" packet is a full-resolution still with no depth.
    private func makePacket(frame: ARFrame, depth: ARDepthData?, kind: String, quality: Double) -> Data {
        let colour = CIImage(cvPixelBuffer: frame.capturedImage)
        let jpeg = ciContext.jpegRepresentation(
            of: colour,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality]
        )!
        let depthBytes = depth.map { rows(of: $0.depthMap, as: Float32.self).map { Float16($0) } } ?? []
        let confidenceBytes = depth.map { rows(of: $0.confidenceMap!, as: UInt8.self) } ?? []

        // intrinsics is column major: column 2 holds the principal point cx, cy.
        let k = frame.camera.intrinsics
        let header: [String: Any] = [
            "kind": kind,
            "time": frame.timestamp,
            "intrinsics": [
                "fx": Double(k[0][0]), "fy": Double(k[1][1]), "cx": Double(k[2][0]), "cy": Double(k[2][1]),
                "width": CVPixelBufferGetWidth(frame.capturedImage),
                "height": CVPixelBufferGetHeight(frame.capturedImage),
            ],
            "jpeg_bytes": jpeg.count,
            "depth_width": depth.map { CVPixelBufferGetWidth($0.depthMap) } ?? 0,
            "depth_height": depth.map { CVPixelBufferGetHeight($0.depthMap) } ?? 0,
        ]
        let headerData = try! JSONSerialization.data(withJSONObject: header)

        var packet = Data()
        var headerLength = UInt32(headerData.count).littleEndian
        packet.append(Data(bytes: &headerLength, count: 4))
        packet.append(headerData)
        packet.append(jpeg)
        depthBytes.withUnsafeBufferPointer { packet.append($0) }
        confidenceBytes.withUnsafeBufferPointer { packet.append($0) }
        return packet
    }

    /// Copies a pixel buffer into a flat array, row by row. Rows can have padding at the end, so we skip it.
    private func rows<T>(of buffer: CVPixelBuffer, as type: T.Type) -> [T] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!
        var values = [T]()
        values.reserveCapacity(width * height)
        for y in 0..<height {
            let row = (base + y * rowBytes).assumingMemoryBound(to: T.self)
            values.append(contentsOf: UnsafeBufferPointer(start: row, count: width))
        }
        return values
    }
}
