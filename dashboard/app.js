// Dashboard glue: renders recorded results from results.json, and drives
// the live in-browser verification against kernels/reference/matmul_ref.c
// (compiled to wasm/matmul_ref.{js,wasm} by kernels/reference/build_wasm.sh).

const MAX_DIM = 768;

function fmtMs(ms) {
  return `${ms.toFixed(2)} ms`;
}

function badge(verdict) {
  const cls = verdict === "PASS" ? "pass" : verdict === "pending" ? "pending" : "fail";
  return `<span class="badge ${cls}">${verdict}</span>`;
}

async function loadRecordedResults() {
  const res = await fetch("results.json");
  const data = await res.json();

  const tbody = document.querySelector("#recorded-table tbody");
  for (const run of data.recorded_runs) {
    const tr = document.createElement("tr");
    const { M, N, K } = run.size;
    tr.innerHTML = `
      <td>${run.kernel_label}</td>
      <td>${run.device}${run.device_is_gpu ? "" : " <span class=\"subtle\">(not a discrete GPU)</span>"}</td>
      <td>${M}×${N}×${K}</td>
      <td>${fmtMs(run.kernel_time_ms)}</td>
      <td>${fmtMs(run.cpu_reference_time_ms)}</td>
      <td>${run.max_abs_diff}</td>
      <td>${badge(run.verdict)}</td>
    `;
    tbody.appendChild(tr);
    if (run.note) {
      const noteTr = document.createElement("tr");
      const noteTd = document.createElement("td");
      noteTd.colSpan = 7;
      noteTd.className = "note";
      noteTd.textContent = run.note;
      noteTr.appendChild(noteTd);
      tbody.appendChild(noteTr);
    }
  }

  const unavailableEl = document.getElementById("unavailable-list");
  for (const item of data.not_available) {
    const div = document.createElement("div");
    div.className = "note";
    div.innerHTML = `${badge("pending")} <strong>${item.kernel_label}</strong> — ${item.reason}`;
    unavailableEl.appendChild(div);
  }

  return data;
}

// Deterministic seeded PRNG (mulberry32) so "seed" is reproducible across
// runs/browsers - not the same generator as the C programs' rand(), just a
// stand-in that makes this page's own runs repeatable.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomMatrix(rows, cols, rng) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < out.length; i++) out[i] = rng() * 10;
  return out;
}

// Independent plain-JS reimplementation of the identical naive algorithm in
// kernels/reference/matmul_ref.c, used only to cross-check the WASM output.
// matmul_ref.c accumulates in `float` (32-bit), not double - Math.fround
// after every addition mirrors that, otherwise this "cross-check" would
// just be measuring double-vs-float32 rounding drift instead of an actual
// algorithm mismatch.
function matmulJs(A, B, M, N, K) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let k = 0; k < K; k++) {
        acc = Math.fround(acc + Math.fround(A[i * K + k] * B[k * N + j]));
      }
      C[i * N + j] = acc;
    }
  }
  return C;
}

function matmulWasm(Module, A, B, M, N, K) {
  const bytesA = A.length * 4;
  const bytesB = B.length * 4;
  const bytesC = M * N * 4;
  const pA = Module._malloc(bytesA);
  const pB = Module._malloc(bytesB);
  const pC = Module._malloc(bytesC);
  try {
    Module.HEAPF32.set(A, pA / 4);
    Module.HEAPF32.set(B, pB / 4);
    Module.ccall(
      "matmul_cpu_reference",
      null,
      ["number", "number", "number", "number", "number", "number"],
      [pA, pB, pC, M, N, K]
    );
    return Module.HEAPF32.slice(pC / 4, pC / 4 + M * N);
  } finally {
    Module._free(pA);
    Module._free(pB);
    Module._free(pC);
  }
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

async function main() {
  const statusEl = document.getElementById("verify-status");
  statusEl.textContent = "Loading WASM module…";

  const [recorded, Module] = await Promise.all([
    loadRecordedResults(),
    createMatmulRefModule(),
  ]);

  statusEl.textContent = "WASM module ready.";

  const runBtn = document.getElementById("run-btn");
  const resultGrid = document.getElementById("result-grid");
  const comparisonEl = document.getElementById("recorded-comparison");

  runBtn.addEventListener("click", () => {
    const M = clampDim(document.getElementById("size-m").value);
    const N = clampDim(document.getElementById("size-n").value);
    const K = clampDim(document.getElementById("size-k").value);
    const seed = Number(document.getElementById("seed").value) || 42;

    runBtn.disabled = true;
    statusEl.textContent = `Running ${M}×${N}×${K} matmul (WASM + JS cross-check)…`;

    // Defer to next paint so the "Running…" status actually shows before
    // the (synchronous, main-thread) O(n^3) loops block the tab.
    setTimeout(() => {
      try {
        const rng = mulberry32(seed);
        const A = randomMatrix(M, K, rng);
        const B = randomMatrix(K, N, rng);

        const t0 = performance.now();
        const wasmResult = matmulWasm(Module, A, B, M, N, K);
        const wasmMs = performance.now() - t0;

        const t1 = performance.now();
        const jsResult = matmulJs(A, B, M, N, K);
        const jsMs = performance.now() - t1;

        const diff = maxAbsDiff(wasmResult, jsResult);
        const pass = diff < 1e-2;

        document.getElementById("stat-wasm-time").textContent = fmtMs(wasmMs);
        document.getElementById("stat-js-time").textContent = fmtMs(jsMs);
        document.getElementById("stat-diff").textContent = diff.toExponential(3);
        const verdictEl = document.getElementById("stat-verdict");
        verdictEl.textContent = pass ? "PASS" : "FAIL";
        verdictEl.className = `value ${pass ? "good" : "bad"}`;
        resultGrid.hidden = false;

        statusEl.textContent = `Done. Compared ${(M * N).toLocaleString()} output elements.`;

        const matchingRecorded = recorded.recorded_runs.find(
          (r) => r.size.M === M && r.size.N === N && r.size.K === K
        );
        if (matchingRecorded) {
          comparisonEl.textContent =
            `This machine's live WASM run took ${fmtMs(wasmMs)} vs. the recorded native ` +
            `CPU reference time of ${fmtMs(matchingRecorded.cpu_reference_time_ms)} on ` +
            `"${matchingRecorded.device}" at this same size. Different machine, different day, ` +
            `and WASM has overhead a native binary doesn't - treat this as a plausibility check, not a controlled comparison.`;
        } else {
          comparisonEl.textContent =
            "No recorded run at this exact size to compare against - see the table above for sizes that were actually benchmarked.";
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        console.error(err);
      } finally {
        runBtn.disabled = false;
      }
    }, 10);
  });
}

function clampDim(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n) || n < 2) return 2;
  if (n > MAX_DIM) return MAX_DIM;
  return n;
}

main().catch((err) => {
  document.getElementById("verify-status").textContent = `Failed to initialize: ${err.message}`;
  console.error(err);
});
