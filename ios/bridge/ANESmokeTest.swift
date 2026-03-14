// ANESmokeTest.swift — Minimal test: init ANE, compile trivial MIL, evaluate
// Add this view to the app temporarily to verify ANE private API access on iOS

import SwiftUI

struct ANESmokeTestView: View {
    @State private var status = "Not started"
    @State private var details = ""

    var body: some View {
        VStack(spacing: 20) {
            Text("ANE Bridge Smoke Test")
                .font(.title2.bold())

            Text(status)
                .font(.headline)
                .foregroundColor(status.contains("PASS") ? .green : status.contains("FAIL") ? .red : .secondary)

            if !details.isEmpty {
                Text(details)
                    .font(.system(.caption, design: .monospaced))
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            }

            Button("Run Test") {
                runSmokeTest()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    func runSmokeTest() {
        status = "Running..."
        details = ""

        DispatchQueue.global(qos: .userInitiated).async {
            var log = ""

            // Step 1: Initialize ANE bridge
            log += "1. ane_bridge_init()... "
            let initResult = ane_bridge_init()
            if initResult != 0 {
                DispatchQueue.main.async {
                    status = "FAIL: init"
                    details = log + "FAILED (returned \(initResult))"
                }
                return
            }
            log += "OK\n"

            // Step 2: Compile a trivial MIL program (identity: output = input)
            // MIL: single function that passes input through
            let mil = """
            program(1.0)
            [buildInfo = dict<string, string>()]
            {
                func main<ios18>(tensor<fp16, [1, 4, 1, 4]> x) {
                    fp16 one = const()[name=string("one"), val=fp16(1)];
                    tensor<fp16, [1, 4, 1, 4]> y = mul(x=x, y=one)[name=string("y")];
                } -> (y);
            }
            """

            log += "2. ane_bridge_compile()... "
            let inputSize = 4 * 4 * 2  // [1,4,1,4] fp16 = 32 bytes
            let outputSize = inputSize
            var inSizes = [inputSize]
            var outSizes = [outputSize]

            let kernel = mil.withCString { milPtr in
                ane_bridge_compile(
                    milPtr, strlen(milPtr),
                    nil, 0,
                    1, &inSizes,
                    1, &outSizes
                )
            }

            if kernel == nil {
                DispatchQueue.main.async {
                    status = "FAIL: compile"
                    details = log + "FAILED (returned NULL)\nThis may mean ANE private APIs are blocked on iOS."
                }
                return
            }
            log += "OK\n"

            // Step 3: Write test input, evaluate, read output
            log += "3. ane_bridge_eval()... "

            // Input: [1.0, 2.0, 3.0, ..., 16.0] as fp16
            var inputFp16 = (0..<16).map { Float16(Float($0 + 1)) }
            ane_bridge_write_input(kernel, 0, &inputFp16, inputSize)

            let evalOk = ane_bridge_eval(kernel)
            if !evalOk {
                ane_bridge_free(kernel)
                DispatchQueue.main.async {
                    status = "FAIL: eval"
                    details = log + "FAILED (returned false)"
                }
                return
            }
            log += "OK\n"

            // Step 4: Read output and verify
            var outputFp16 = [Float16](repeating: 0, count: 16)
            ane_bridge_read_output(kernel, 0, &outputFp16, outputSize)

            log += "4. Verify output... "
            let match = zip(inputFp16, outputFp16).allSatisfy { abs(Float($0) - Float($1)) < 0.01 }

            ane_bridge_free(kernel)

            log += match ? "OK\n" : "MISMATCH\n"
            log += "\nInput:  \(inputFp16.prefix(8).map { String(format: "%.1f", Float($0)) }.joined(separator: ", "))...\n"
            log += "Output: \(outputFp16.prefix(8).map { String(format: "%.1f", Float($0)) }.joined(separator: ", "))...\n"
            log += "\nCompile count: \(ane_bridge_get_compile_count())"

            DispatchQueue.main.async {
                status = match ? "PASS ✓" : "FAIL: output mismatch"
                details = log
            }
        }
    }
}
