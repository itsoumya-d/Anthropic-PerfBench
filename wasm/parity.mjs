import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.join(
  here,
  "target/wasm32-unknown-unknown/release/perfbench_wasm.wasm",
);
const fixture = JSON.parse(
  fs.readFileSync(path.join(here, "fixtures/parity.json"), "utf8"),
);

const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), {});
const e = instance.exports;

function allocBytes(length) {
  const ptr = e.pb_alloc(Math.max(1, length));
  if (!ptr) throw new Error(`pb_alloc failed for ${length} bytes`);
  return ptr;
}

const weights = Uint8Array.from(Buffer.from(fixture.weights_hex, "hex"));
const scales = Float32Array.from(fixture.scales);
const x = Float32Array.from(fixture.x);

const wPtr = allocBytes(weights.length);
const sPtr = allocBytes(scales.byteLength);
const xPtr = allocBytes(x.byteLength);
const yScalarPtr = allocBytes(fixture.rows * 4);
const ySimdPtr = allocBytes(fixture.rows * 4);

new Uint8Array(e.memory.buffer, wPtr, weights.length).set(weights);
new Float32Array(e.memory.buffer, sPtr, scales.length).set(scales);
new Float32Array(e.memory.buffer, xPtr, x.length).set(x);

e.pb_int4_gemv_scalar(
  wPtr,
  sPtr,
  xPtr,
  yScalarPtr,
  fixture.rows,
  fixture.cols,
  fixture.group,
);
e.pb_int4_gemv_simd(
  wPtr,
  sPtr,
  xPtr,
  ySimdPtr,
  fixture.rows,
  fixture.cols,
  fixture.group,
);

const scalar = Array.from(
  new Float32Array(e.memory.buffer, yScalarPtr, fixture.rows),
);
const simd = Array.from(new Float32Array(e.memory.buffer, ySimdPtr, fixture.rows));
const xAfter = Array.from(new Float32Array(e.memory.buffer, xPtr, x.length));

const tolerance = 2e-3;
let failures = 0;
for (let r = 0; r < fixture.rows; r++) {
  for (const [name, actual] of [
    ["scalar", scalar[r]],
    ["simd", simd[r]],
  ]) {
    const expected = fixture.expected[r];
    const error = Math.abs(actual - expected);
    if (error > tolerance * (1 + Math.abs(expected))) {
      console.error(
        `FAIL ${name} row ${r}: ${actual} vs ${expected} (err ${error})`,
      );
      failures++;
    }
  }
  if (Math.abs(scalar[r] - simd[r]) > tolerance * (1 + Math.abs(scalar[r]))) {
    console.error(
      `FAIL scalar/simd divergence row ${r}: ${scalar[r]} vs ${simd[r]}`,
    );
    failures++;
  }
}
for (let i = 0; i < x.length; i++) {
  if (xAfter[i] !== x[i]) {
    console.error(`FAIL activation buffer mutated at ${i}`);
    failures++;
    break;
  }
}

if (failures > 0) {
  console.error(`${failures} parity failure(s)`);
  process.exit(1);
}

const maxError = Math.max(
  ...scalar.map((value, r) => Math.abs(value - fixture.expected[r])),
  ...simd.map((value, r) => Math.abs(value - fixture.expected[r])),
);
console.log(
  `PASS WASM parity: ${fixture.rows} rows, max |error| ${maxError.toExponential(3)}`,
);
