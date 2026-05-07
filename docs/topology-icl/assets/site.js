(function () {
  const fallbackReport = {
    pooled: {
      run_rows: 240,
      aggregate_groups: 48,
      retrain_groups: 96,
      run_level: {
        edge_count: { leave_one_out_r2: -0.008385707533131503 },
        edge_plus_drel: { leave_one_out_r2: 0.056613684968085054 },
        input_plus_masked_geometry: { leave_one_out_r2: 0.11606011072334954 },
        edge_plus_projection: { leave_one_out_r2: 0.7404015889947272 }
      },
      aggregate_target_mean: {
        edge_count: { leave_one_out_r2: -0.04300588501584346 },
        edge_plus_drel: { leave_one_out_r2: 0.14495126498743238 },
        input_plus_masked_geometry: { leave_one_out_r2: 0.1886838122574226 },
        edge_plus_projection: { leave_one_out_r2: 0.8092894491130684 }
      },
      retrain_target_mean: {
        layout_type: { leave_one_out_r2: 0.5097686119912274 },
        layout_plus_input_plus_drel: { leave_one_out_r2: 0.8209996829258707 },
        layout_plus_input_plus_masked_geometry: { leave_one_out_r2: 0.8493997176917111 }
      }
    },
    experiments: [
      { name: "random", run_summary: { n_runs: 80, target_mean: 76.7575, target_max: 94.6, target_std: 9.678 } },
      { name: "cycle", run_summary: { n_runs: 80, target_mean: 76.3775, target_max: 94.2, target_std: 9.787 } },
      { name: "hub", run_summary: { n_runs: 80, target_mean: 69.8725, target_max: 91.4, target_std: 10.461 } }
    ]
  };

  const fallbackInterpretation = {
    support_summary: { verdict: "strong_positive" },
    essential_retention: [
      { experiment: "random", layout: "physical subgraph", n_joined: 16, retention_mean_mean: 0.7331775257406948, retrain_mean_mean: 65.6875, retrain_max_best: 90.0 },
      { experiment: "random", layout: "input mask", n_joined: 16, retention_mean_mean: 0.6246560445474783, retrain_mean_mean: 55.3675, retrain_max_best: 72.2 },
      { experiment: "cycle", layout: "physical subgraph", n_joined: 16, retention_mean_mean: 0.7580217663110975, retrain_mean_mean: 67.4025, retrain_max_best: 88.6 },
      { experiment: "cycle", layout: "input mask", n_joined: 16, retention_mean_mean: 0.618140707093195, retrain_mean_mean: 54.59, retrain_max_best: 73.2 },
      { experiment: "hub", layout: "physical subgraph", n_joined: 16, retention_mean_mean: 0.8557884784151505, retrain_mean_mean: 72.09, retrain_max_best: 92.4 },
      { experiment: "hub", layout: "input mask", n_joined: 16, retention_mean_mean: 0.6372176351574668, retrain_mean_mean: 52.3975, retrain_max_best: 65.2 }
    ]
  };

  const palette = {
    blue: "#315f9c",
    teal: "#087d7a",
    rust: "#b6503b",
    gold: "#b17b1d",
    green: "#3c7f46",
    gray: "#7b8796",
    line: "#d9e1ea",
    ink: "#1c2633"
  };

  function pct(value) {
    return `${Number(value).toFixed(1)}%`;
  }

  function fixed(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }
    return Number(value).toFixed(digits);
  }

  function get(obj, path, fallback) {
    return path.reduce((node, key) => (node && node[key] !== undefined ? node[key] : undefined), obj) ?? fallback;
  }

  async function loadJson(path, fallback) {
    try {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (_error) {
      return fallback;
    }
  }

  function updateText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  }

  function chartDefaults() {
    if (!window.Chart) return;
    Chart.defaults.font.family =
      'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    Chart.defaults.color = palette.ink;
    Chart.defaults.borderColor = palette.line;
  }

  function renderCoverageChart(report) {
    if (!window.Chart) return;
    const canvas = document.getElementById("coverageChart");
    if (!canvas) return;

    const experiments = report.experiments || fallbackReport.experiments;
    const labels = experiments.map((item) => item.name);
    const means = experiments.map((item) => item.run_summary.target_mean);
    const maxes = experiments.map((item) => item.run_summary.target_max);
    const stds = experiments.map((item) => item.run_summary.target_std);

    new Chart(canvas, {
      data: {
        labels,
        datasets: [
          {
            type: "bar",
            label: "mean accuracy",
            data: means,
            backgroundColor: palette.blue,
            borderRadius: 6
          },
          {
            type: "bar",
            label: "best seed",
            data: maxes,
            backgroundColor: palette.teal,
            borderRadius: 6
          },
          {
            type: "line",
            label: "seed std.",
            data: stds,
            borderColor: palette.rust,
            backgroundColor: palette.rust,
            tension: 0.25,
            pointRadius: 4
          }
        ]
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: { callback: (value) => `${value}%` },
            title: { display: true, text: "novel-class ICL accuracy" }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (item) => `${item.dataset.label}: ${pct(item.raw)}`
            }
          }
        }
      }
    });
  }

  function renderModelChart(report) {
    if (!window.Chart) return;
    const canvas = document.getElementById("modelChart");
    if (!canvas) return;

    const rows = [
      ["run: edge count", ["pooled", "run_level", "edge_count", "leave_one_out_r2"], palette.gray],
      ["run: + d_rel", ["pooled", "run_level", "edge_plus_drel", "leave_one_out_r2"], palette.blue],
      ["run: + masked geometry", ["pooled", "run_level", "input_plus_masked_geometry", "leave_one_out_r2"], palette.teal],
      ["run: projection diagnostics", ["pooled", "run_level", "edge_plus_projection", "leave_one_out_r2"], palette.rust],
      ["topology: edge count", ["pooled", "aggregate_target_mean", "edge_count", "leave_one_out_r2"], palette.gray],
      ["topology: + d_rel", ["pooled", "aggregate_target_mean", "edge_plus_drel", "leave_one_out_r2"], palette.blue],
      ["topology: + masked geometry", ["pooled", "aggregate_target_mean", "input_plus_masked_geometry", "leave_one_out_r2"], palette.teal],
      ["topology: projection diagnostics", ["pooled", "aggregate_target_mean", "edge_plus_projection", "leave_one_out_r2"], palette.rust],
      ["retrain: layout", ["pooled", "retrain_target_mean", "layout_type", "leave_one_out_r2"], palette.gray],
      ["retrain: + d_rel", ["pooled", "retrain_target_mean", "layout_plus_input_plus_drel", "leave_one_out_r2"], palette.blue],
      ["retrain: + masked geometry", ["pooled", "retrain_target_mean", "layout_plus_input_plus_masked_geometry", "leave_one_out_r2"], palette.teal]
    ];

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: rows.map((row) => row[0]),
        datasets: [
          {
            label: "leave-one-out R2",
            data: rows.map((row) => get(report, row[1], 0)),
            backgroundColor: rows.map((row) => row[2]),
            borderRadius: 6
          }
        ]
      },
      options: {
        indexAxis: "y",
        maintainAspectRatio: false,
        scales: {
          x: {
            min: -0.1,
            max: 0.9,
            title: { display: true, text: "LOO R2" }
          },
          y: {
            ticks: { autoSkip: false }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => `LOO R2: ${fixed(item.raw, 3)}`
            }
          }
        }
      }
    });
  }

  function renderMotifChart(interpretation) {
    if (!window.Chart) return;
    const canvas = document.getElementById("motifChart");
    if (!canvas) return;

    const rows = interpretation.essential_retention || fallbackInterpretation.essential_retention;
    const experiments = [...new Set(rows.map((row) => row.experiment))];
    const physical = experiments.map((name) => rows.find((row) => row.experiment === name && row.layout === "physical subgraph")?.retrain_mean_mean || 0);
    const mask = experiments.map((name) => rows.find((row) => row.experiment === name && row.layout === "input mask")?.retrain_mean_mean || 0);

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: experiments,
        datasets: [
          {
            label: "physical subgraph",
            data: physical,
            backgroundColor: palette.green,
            borderRadius: 6
          },
          {
            label: "input mask",
            data: mask,
            backgroundColor: palette.gold,
            borderRadius: 6
          }
        ]
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: { callback: (value) => `${value}%` },
            title: { display: true, text: "retrained novel-class ICL" }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (item) => `${item.dataset.label}: ${pct(item.raw)}`
            }
          }
        }
      }
    });
  }

  function renderRetentionTable(interpretation) {
    const tableBody = document.querySelector("#retentionTable tbody");
    if (!tableBody) return;
    const rows = interpretation.essential_retention || fallbackInterpretation.essential_retention;
    tableBody.innerHTML = "";

    rows.forEach((row) => {
      const tr = document.createElement("tr");
      [
        row.experiment,
        row.layout,
        fixed(row.retention_mean_mean, 3),
        pct(row.retrain_mean_mean),
        pct(row.retrain_max_best)
      ].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      tableBody.appendChild(tr);
    });
  }

  function renderSummary(report, interpretation) {
    updateText("runRows", get(report, ["pooled", "run_rows"], 240));
    updateText("topologyGroups", get(report, ["pooled", "aggregate_groups"], 48));
    updateText("retrainGroups", get(report, ["pooled", "retrain_groups"], 96));
    updateText("verdictText", get(interpretation, ["support_summary", "verdict"], "strong_positive"));
  }

  function drawFlowCanvas() {
    const canvas = document.getElementById("flowCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const nodes = [
      { x: 0.12, y: 0.34 },
      { x: 0.28, y: 0.19 },
      { x: 0.45, y: 0.37 },
      { x: 0.26, y: 0.58 },
      { x: 0.62, y: 0.25 },
      { x: 0.79, y: 0.47 },
      { x: 0.61, y: 0.66 }
    ];
    const edges = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [0, 2],
      [2, 4],
      [4, 5],
      [5, 6],
      [6, 3],
      [3, 5],
      [1, 4]
    ];

    let width = 0;
    let height = 0;
    let startedAt = performance.now();

    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const box = canvas.getBoundingClientRect();
      width = Math.max(1, Math.round(box.width));
      height = Math.max(1, Math.round(box.height));
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function point(node) {
      return { x: node.x * width, y: node.y * height };
    }

    function draw(time) {
      if (!width || !height) resize();
      ctx.clearRect(0, 0, width, height);

      const tick = (time - startedAt) / 1000;
      ctx.lineCap = "round";
      edges.forEach(([from, to], index) => {
        const a = point(nodes[from]);
        const b = point(nodes[to]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = "rgba(49, 95, 156, 0.22)";
        ctx.lineWidth = 2;
        ctx.stroke();

        const phase = (tick * 0.2 + index * 0.087) % 1;
        const x = a.x + (b.x - a.x) * phase;
        const y = a.y + (b.y - a.y) * phase;
        ctx.beginPath();
        ctx.arc(x, y, 4.2, 0, Math.PI * 2);
        ctx.fillStyle = index % 3 === 0 ? "rgba(182, 80, 59, 0.82)" : "rgba(8, 125, 122, 0.75)";
        ctx.fill();
      });

      nodes.forEach((node, index) => {
        const p = point(node);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = index % 2 ? "rgba(8, 125, 122, 0.68)" : "rgba(49, 95, 156, 0.68)";
        ctx.stroke();
      });

      requestAnimationFrame(draw);
    }

    resize();
    window.addEventListener("resize", resize, { passive: true });
    requestAnimationFrame(draw);
  }

  async function main() {
    chartDefaults();
    drawFlowCanvas();
    const [report, interpretation] = await Promise.all([
      loadJson("data/topology_research_report.json", fallbackReport),
      loadJson("data/topology_research_report_interpretation.json", fallbackInterpretation)
    ]);
    renderSummary(report, interpretation);
    renderCoverageChart(report);
    renderModelChart(report);
    renderMotifChart(interpretation);
    renderRetentionTable(interpretation);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
