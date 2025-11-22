// app.js - polished UI controller
// Backend endpoints:
// GET  /preview_dataset?dataset=NAME
// POST /start_training  {dataset, algo, hyperparams, epochs, interval_ms, mode} -> {session_id}
// GET  /stream_updates?session_id=ID  (SSE streaming per-iteration frames)
// GET  /stop_training?session_id=ID  (optional)

const API = "";  // relative for local dev


// DOM refs
const datasetSelect = document.getElementById('datasetSelect');
const algoSelect = document.getElementById('algoSelect');
const hpPills = document.getElementById('hpPills');
const plotOverlay = document.getElementById('plotOverlay');

const trainBtn = document.getElementById('trainBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');

const rightPanel = document.getElementById('rightPanel');

const accuracyVal = document.getElementById('accuracyVal');
const lossVal = document.getElementById('lossVal');

const confCard = document.getElementById('confCard');

const epochsInput = document.getElementById('epochs');
const intervalInput = document.getElementById('interval');

// theme toggle
const themeBtn = document.getElementById('themeBtn');
themeBtn.addEventListener('click', () => {
  const root = document.documentElement;
  if (root.classList.contains('light')) {
    root.classList.remove('light');
    themeBtn.innerText = '🌙';
  } else {
    root.classList.add('light');
    themeBtn.innerText = '☀️';
  }
});

// Hyperparameter schema (kept compact for pill UI)
const SCHEMA = {
  knn: [{key:'n_neighbors', type:'range', min:1, max:25, step:1, value:5, label:'k'}],
  logistic: [{key:'C', type:'number', min:0.001, max:100, step:0.001, value:1, label:'C'}],
  svm: [
    {key:'C', type:'number', min:0.01, max:100, step:0.01, value:1, label:'C'},
    {key:'kernel', type:'select', options:['rbf','linear','poly'], value:'rbf', label:'Kernel'}
  ],
  decision_tree: [{key:'max_depth', type:'range', min:1, max:50, step:1, value:5, label:'Max depth'}],
  random_forest: [{key:'n_estimators', type:'number', min:1, max:500, step:1, value:100, label:'Trees'}],
  neural_network: [
    {key:'activation_function', type:'select', options:['relu','linear','tanh','logistic'], value:'relu', label:'activation_function'},
    {key:'hidden_layer_sizes', type:'text', value:'50,', label:'Layers'},
    {key:'learning_rate_init', type:'number', min:0.00001, max:1, step:0.00001, value:0.001, label:'LR'}
  ]
};

// render HP pills in nav
function renderHpPills(algo) {
  hpPills.innerHTML = '';
  const schema = SCHEMA[algo] || [];
  schema.forEach(s => {
    const pill = document.createElement('div');
    pill.className = 'hp-pill';
    const lbl = document.createElement('label');
    lbl.innerText = s.label;
    lbl.className = 'small muted';
    pill.appendChild(lbl);

    let input;
    if (s.type === 'select') {
      input = document.createElement('select');
      s.options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt;
        o.innerText = opt;
        if (opt === s.value) o.selected = true;
        input.appendChild(o);
      });
    } else {
      input = document.createElement('input');
      input.type = s.type === 'range' ? 'range' : s.type;
      input.value = s.value;
      if (s.min !== undefined) input.min = s.min;
      if (s.max !== undefined) input.max = s.max;
      if (s.step !== undefined) input.step = s.step;
    }
    input.id = `hp_${s.key}`;
    pill.appendChild(input);
    hpPills.appendChild(pill);

    // if range, show dynamic value
    if (s.type === 'range') {
      const valSpan = document.createElement('div');
      valSpan.className = 'small muted';
      valSpan.style.marginTop = '6px';
      valSpan.innerText = input.value;
      pill.appendChild(valSpan);
      input.addEventListener('input', () => (valSpan.innerText = input.value));
    }
  });
}
algoSelect.addEventListener('change', () => renderHpPills(algoSelect.value));
renderHpPills(algoSelect.value);

// collect HPs
function collectHPs() {
  const schema = SCHEMA[algoSelect.value] || [];
  const out = {};
  schema.forEach(s => {
    const el = document.getElementById(`hp_${s.key}`);
    if (!el) return;
    out[s.key] = (s.type === 'number' || s.type === 'range') ? Number(el.value) : el.value;
  });
  return out;
}

// Plot preview
async function loadPreview() {
  plotOverlay.style.display = 'block';
  plotOverlay.innerText = 'Loading preview…';
  try {
    const resp = await fetch(`${API}/preview_dataset?dataset=${encodeURIComponent(datasetSelect.value)}`);
    if (!resp.ok) throw new Error('no preview');
    const data = await resp.json();
    plotOverlay.style.display = 'none';
    renderPreview(data);
  } catch (e) {
    plotOverlay.innerText = 'Preview unavailable — start training to see live frames';
  }
}
datasetSelect.addEventListener('change', loadPreview);
window.addEventListener('load', loadPreview);

// Render dataset preview (scatter)
function renderPreview(payload) {
  const traces = [];
  const classes = Array.from(new Set(payload.train.labels.concat(payload.test.labels))).sort();
  classes.forEach(cls => {
    const xs = [], ys = [];
    payload.train.labels.forEach((l, i) => {
      if (String(l) === String(cls)) {
        xs.push(payload.train.x[i]);
        ys.push(payload.train.y[i]);
      }
    });
    traces.push({x: xs, y: ys, mode: 'markers', marker: {size: 8}, name: `train:${cls}`});
  });

  const tx = [], ty = [];
  payload.test.labels.forEach((l, i) => {
    tx.push(payload.test.x[i]);
    ty.push(payload.test.y[i]);
  });
  traces.push({x: tx, y: ty, mode: 'markers', marker: {symbol: 'x', size: 11}, name: 'test'});

  const layout = {margin: {t: 10}, showlegend: true};
  window.renderBoundaryFrame({}, traces, layout);
}

// SSE streaming
let sse = null;
let sessionID = null;
let streaming = false;

trainBtn.addEventListener('click', startTraining);
stopBtn.addEventListener('click', stopTraining);
resetBtn.addEventListener('click', () => {
  loadPreview();

  // reset metric plots with fixed axes
  Plotly.react('accPlot',
    [{x: [], y: [], mode: 'lines+markers', name: 'Accuracy'}],
    {
      margin: {t: 10},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: {range: [0, 1]}
    }
  );
  Plotly.react('lossPlot',
    [{x: [], y: [], mode: 'lines+markers', name: 'Loss'}],
    {
      margin: {t: 10},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: {rangemode: 'tozero'}
    }
  );

  accuracyVal.innerText = '—';
  lossVal.innerText = '—';
  confCard.classList.add('hidden');
});

// ⭐⭐⭐ AUTO RESET + START TRAINING ⭐⭐⭐
async function startTraining() {
  if (streaming) return;

  // AUTO RESET BEFORE TRAINING
  if (sse) {
    sse.close();
    sse = null;
  }

  streaming = false;
  sessionID = null;

  plotOverlay.style.display = 'none';

  accuracyVal.innerText = '—';
  lossVal.innerText = '—';

  confCard.classList.add('hidden');

  Plotly.react('accPlot',
    [{x: [], y: [], mode: 'lines+markers', name: 'Accuracy'}],
    {
      margin: {t: 10},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: {range: [0, 1]}
    }
  );
  Plotly.react('lossPlot',
    [{x: [], y: [], mode: 'lines+markers', name: 'Loss'}],
    {
      margin: {t: 10},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: {rangemode: 'tozero'}
    }
  );

  Plotly.react(
    'boundaryPlot',
    [{x: [], y: [], mode: 'markers'}],
    {margin: {t: 20}},
  );

  loadPreview();
  stopBtn.disabled = true;

  // PAYLOAD
  const mode = document.getElementById("trainingMode").value;
  const payload = {
    dataset: datasetSelect.value,
    algo: algoSelect.value,
    hyperparams: {...collectHPs(), mode},
    epochs: Number(epochsInput.value),
    interval_ms: Number(intervalInput.value),
    mode
  };

  try {
    const resp = await fetch(`${API}/start_training`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error('start failed');
    const res = await resp.json();

    sessionID = res.session_id;
    sse = new EventSource(`${API}/stream_updates?session_id=${sessionID}`);
    streaming = true;
    trainBtn.disabled = true;
    stopBtn.disabled = false;
    plotOverlay.style.display = 'none';

    sse.onmessage = (e) => {
      if (!e.data) return;
      const msg = JSON.parse(e.data);
      if (msg.status === 'done') {
        streaming = false;
        trainBtn.disabled = false;
        stopBtn.disabled = true;
        if (msg.final_confusion) showConfusion(msg.final_confusion, msg.class_names || null);
        if (sse) {
          sse.close();
          sse = null;
        }
        return;
      }
      handleFrame(msg);
    };

    sse.onerror = (err) => {
      console.error('SSE error', err);
      stopTraining();
    };

  } catch (err) {
    alert('Could not start training: ' + err.message);
  }
}

function stopTraining() {
  if (!streaming) return;
  if (sessionID) fetch(`${API}/stop_training?session_id=${sessionID}`).catch(() => {});
  if (sse) {
    sse.close();
    sse = null;
  }
  streaming = false;
  trainBtn.disabled = false;
  stopBtn.disabled = true;
}


function handleFrame(frame) {

  /* ---- Bug-3 Fix: persistent full dataset for optimized mode ---- */
  if (frame.mode === "optimized") {
    // store full training set on the first iteration
    if (!window.fullTrainSet) {
      window.fullTrainSet = {
        x: frame.train.x.slice(),
        y: frame.train.y.slice(),
        labels: frame.train.labels.slice()
      };
    }
    // always reuse full training set for plotting
    frame.train = window.fullTrainSet;
  } else {
    // educational mode → reset cache
    window.fullTrainSet = null;
  }
  /* --------------------------------------------------------------- */

  const grid = frame.grid;
  let contour = null;
  if (grid && grid.preds) {
    const nx = grid.nx, ny = grid.ny;
    const z = [];
    for (let r = 0; r < ny; r++) {
      const row = [];
      for (let c = 0; c < nx; c++) row.push(grid.preds[r * nx + c]);
      z.push(row);
    }
    const xi = Array.from({length: nx}, (_, i) => grid.x_min + i * (grid.x_max - grid.x_min) / (nx - 1));
    const yi = Array.from({length: ny}, (_, i) => grid.y_min + i * (grid.y_max - grid.y_min) / (ny - 1));
    contour = {x: xi, y: yi, z: z, type: 'contour', colorscale: 'RdBu', opacity: 0.45, showscale: false};
  }

  const traces = [];
  const classes = Array.from(new Set(frame.train.labels.concat(frame.test.labels))).sort();
  classes.forEach(cls => {
    const xs = [], ys = [];
    frame.train.labels.forEach((lab, i) => {
      if (String(lab) === String(cls)) {
        xs.push(frame.train.x[i]);
        ys.push(frame.train.y[i]);
      }
    });
    traces.push({x: xs, y: ys, mode: 'markers', marker: {size: 8}, name: `train:${cls}`});
  });

  const goodX = [], goodY = [], badX = [], badY = [];
  frame.test.preds.forEach((p, i) => {
    if (String(p) === String(frame.test.labels[i])) {
      goodX.push(frame.test.x[i]); goodY.push(frame.test.y[i]);
    } else {
      badX.push(frame.test.x[i]); badY.push(frame.test.y[i]);
    }
  });
  traces.push({x: goodX, y: goodY, mode: 'markers', marker: {symbol: 'x', color: 'green', size: 12}, name: 'Correct'});
  traces.push({x: badX, y: badY, mode: 'markers', marker: {symbol: 'x', color: 'red', size: 12}, name: 'Wrong'});

  const modeLabel = frame.mode === 'optimized' ? 'Optimized' : 'Educational';
  const fx = (frame.feature_names && frame.feature_names[0]) ? frame.feature_names[0] : 'Feature 1';
  const fy = (frame.feature_names && frame.feature_names[1]) ? frame.feature_names[1] : 'Feature 2';

  const layout = {
    margin: {t: 30},
    title: `${modeLabel} — Iter ${frame.iteration} — Acc ${(frame.acc * 100).toFixed(2)}%`,
    xaxis: {title: fx},
    yaxis: {title: fy}
  };
  window.renderBoundaryFrame(contour || {}, traces, layout);

  accuracyVal.innerText = (frame.acc * 100).toFixed(2) + "%";
  lossVal.innerText = frame.loss.toFixed(4);

  Plotly.extendTraces('accPlot', {x: [[frame.iteration]], y: [[frame.acc]]}, [0]);
  Plotly.extendTraces('lossPlot', {x: [[frame.iteration]], y: [[frame.loss]]}, [0]);

  if (frame.confusion && Array.isArray(frame.confusion)) {
    showConfusion(frame.confusion, frame.class_names || null);
  }
}


function showConfusion(matrix, classNames) {
  confCard.classList.remove("hidden");

  let labels = Array.isArray(classNames) && classNames.length === matrix.length
    ? classNames
    : matrix.map((_, i) => String(i));

  const rows = matrix.length;
  const cols = matrix[0].length;

  let maxVal = Math.max(...matrix.flat());
  if (maxVal === 0) maxVal = 1;

  // build table format for plotly: header + full rows
  const header = ["Actual / Pred", ...labels];
  const values = [];

  for (let r = 0; r < rows; r++) {
    const rowVals = [labels[r]];
    for (let c = 0; c < cols; c++) rowVals.push(matrix[r][c]);
    values.push(rowVals);
  }

  // generate color shading
  const colors = values.map((row, idx) =>
    row.map((v, j) => {
      if (j === 0) return "rgb(230,230,230)"; // label column
      const t = matrix[idx][j - 1] / maxVal;
      const r = Math.round(90 - 60 * t);
      const g = Math.round(115 - 70 * t);
      const b = Math.round(180 - 70 * t);
      return `rgb(${r},${g},${b})`;
    })
  );

  Plotly.react(
    "confMatrix",
    [{
      type: "table",
      header: {
        values: header,
        align: "center",
        fill: { color: "rgb(210,210,210)" },
        font: { size: 16, color: "black" }
      },
      cells: {
        values: values[0].map((_, colIdx) => values.map(row => row[colIdx])),
        fill: { color: colors },
        align: "center",
        font: { size: 14, color: "black" },
        height: 34
      }
    }],
    {
      margin: { t: 20, l: 0, r: 0, b: 0 },
      autosize: true
    },
    { responsive: true }
  );
}


