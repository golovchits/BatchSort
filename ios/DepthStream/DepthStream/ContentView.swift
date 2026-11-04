import SwiftUI

/// One screen: the Mac's address, a start button, and how fast frames are going out.
struct ContentView: View {
    @StateObject private var streamer = Streamer()
    @AppStorage("macAddress") private var macAddress = "192.168.18.110"

    var body: some View {
        VStack(spacing: 24) {
            Text("DepthStream").font(.largeTitle)
            TextField("Mac address", text: $macAddress)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numbersAndPunctuation)
                .autocorrectionDisabled()
                .disabled(streamer.running)
            Button(streamer.running ? "Stop" : "Start") {
                if streamer.running {
                    streamer.stop()
                } else {
                    streamer.start(host: macAddress)
                }
            }
            .buttonStyle(.borderedProminent)
            Text(streamer.status)
        }
        .padding(32)
    }
}
