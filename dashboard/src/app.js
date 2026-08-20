(function () {
  "use strict";
  const D = window.DATA;
  const classes = D.meta.classes;
  const classColor = (i) => Charts.seriesColor(i);
  const classIdx = (c) => classes.indexOf(c);

  /* ============================ theme ============================ */
  function applyTheme(mode) {
    if (mode === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", mode);
    try { localStorage.setItem("omics-theme", mode); } catch (e) {}
    const btn = document.getElementById("themeToggle");
    if (btn) btn.textContent = mode === "dark" ? "🌙" : mode === "light" ? "☀️" : "🌓";
  }
  (function initTheme() {
    let saved = "system";
    try { saved = localStorage.getItem("omics-theme") || "system"; } catch (e) {}
    applyTheme(saved);
    document.getElementById("themeToggle").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "system";
      const next = cur === "system" ? "dark" : cur === "dark" ? "light" : "system";
      applyTheme(next);
    });
  })();

  /* ============================ tabs ============================ */
  const tabBtns = [...document.querySelectorAll(".tab-btn")];
  const panels = [...document.querySelectorAll(".panel")];
  function selectTab(name) {
    tabBtns.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    panels.forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    initPanel(name);
  }
  tabBtns.forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));

  const initialized = new Set();
  function initPanel(name) {
    if (initialized.has(name)) return;
    initialized.add(name);
    if (name === "overview") renderOverview();
    if (name === "explore") renderExplore();
    if (name === "performance") renderPerformance();
    if (name === "predict") renderPredict();
  }

  /* ============================ helpers ============================ */
  function pct(v) { return (v * 100).toFixed(1) + "%"; }
  function statTile(container, label, value, note) {
    const div = document.createElement("div");
    div.className = "card stat-tile";
    div.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div>${note ? `<div class="stat-note">${note}</div>` : ""}`;
    container.appendChild(div);
  }

  /* ============================ Overview ============================ */
  function renderOverview() {
    const stats = document.getElementById("overviewStats");
    const bestModel = Object.values(D.results).sort((a, b) => b.accuracy - a.accuracy)[0];
    statTile(stats, "Patients / samples", D.meta.n_samples, "TCGA RNA-Seq cohort");
    statTile(stats, "Genes profiled", D.meta.n_genes_total.toLocaleString(), `${D.meta.n_genes_selected} selected via ANOVA F-test`);
    statTile(stats, "Cancer types", classes.length, classes.join(" · "));
    statTile(stats, "Best test accuracy", pct(bestModel.accuracy), bestModel.name);

    const chartEl = document.getElementById("classDistChart");
    const counts = classes.map((c) => D.meta.class_counts[c]);
    Charts.barChart(chartEl, {
      data: classes.map((c, i) => ({ label: c, values: [D.meta.class_counts[c]] })),
      seriesNames: [],
      showLegend: false,
      height: 240,
    });
    // recolor bars per-class (barChart default colors by series index, we want by category)
    chartEl.querySelectorAll("rect.mark-bar").forEach((r, i) => r.setAttribute("fill", classColor(i)));

    const tbody = document.getElementById("classTableBody");
    classes.forEach((c, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><span class="badge dot" style="color:${classColor(i)}">${c}</span></td><td>${D.meta.class_info[c]}</td><td>${D.meta.class_counts[c]}</td><td>${pct(D.meta.class_counts[c] / D.meta.n_samples)}</td>`;
      tbody.appendChild(tr);
    });
  }

  /* ============================ Explore ============================ */
  function renderExplore() {
    let mode = "pca";
    const noteEl = document.getElementById("exploreNote");
    function draw() {
      const proj = D.projections[mode];
      const series = classes.map((c, i) => ({
        name: c, color: classColor(i),
        points: proj.points.filter((p) => p.label === c),
      }));
      Charts.scatterChart(document.getElementById("exploreChart"), {
        series,
        xLabel: mode === "pca" ? `PC1 (${(D.projections.pca.var_explained[0] * 100).toFixed(1)}% var)` : "t-SNE 1",
        yLabel: mode === "pca" ? `PC2 (${(D.projections.pca.var_explained[1] * 100).toFixed(1)}% var)` : "t-SNE 2",
        height: 420,
      });
      noteEl.textContent = mode === "pca"
        ? `Principal-component projection of the ${D.meta.n_genes_selected} selected genes. Together PC1+PC2 explain ${((D.projections.pca.var_explained[0] + D.projections.pca.var_explained[1]) * 100).toFixed(1)}% of variance. Filled markers = held-out test samples; faint markers = training samples.`
        : "t-SNE projection (perplexity 30) of the same gene set, better at revealing local cluster structure than PCA, at the cost of not preserving global distances.";
    }
    document.querySelectorAll("#exploreSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll("#exploreSeg button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        mode = b.dataset.mode;
        draw();
      });
    });
    draw();
  }

  /* ============================ Performance ============================ */
  function renderPerformance() {
    const order = ["logreg", "rf", "mlp"];
    // comparison table
    const tbody = document.getElementById("modelCompareBody");
    order.forEach((k) => {
      const r = D.results[k];
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><strong>${r.name}</strong></td><td>${pct(r.accuracy)}</td><td>${pct(r.f1_macro)}</td><td>${r.train_time_sec}s</td>`;
      tbody.appendChild(tr);
    });

    let current = "logreg";
    function draw() {
      const r = D.results[current];
      document.getElementById("perfModelLabel").textContent = r.name;

      Charts.heatmap(document.getElementById("confusionChart"), {
        matrix: r.confusion_matrix, rowLabels: classes, colLabels: classes, height: 320,
      });

      Charts.lineChart(document.getElementById("rocChart"), {
        series: classes.map((c, i) => ({
          name: `${c} (AUC ${r.roc[c].auc.toFixed(3)})`, color: classColor(i),
          points: r.roc[c].fpr.map((f, j) => ({ x: f, y: r.roc[c].tpr[j] })),
        })),
        xDomain: [0, 1], yDomain: [0, 1], height: 300, diagonal: true,
        xLabel: "False positive rate", yLabel: "True positive rate",
        xTickFmt: (v) => v.toFixed(1), valueFmt: (v) => v.toFixed(1),
      });

      Charts.barChart(document.getElementById("perClassChart"), {
        data: r.per_class.map((pc) => ({ label: pc.class, values: [pc.precision, pc.recall, pc.f1] })),
        seriesNames: ["Precision", "Recall", "F1"],
        yDomain: [0, 1], height: 260, valueFmt: (v) => v.toFixed(1),
      });
    }
    document.querySelectorAll("#perfSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll("#perfSeg button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        current = b.dataset.model;
        draw();
      });
    });
    draw();

    // MLP training curves
    const h = D.mlp_history;
    Charts.lineChart(document.getElementById("lossChart"), {
      series: [
        { name: "Train", color: classColor(0), points: h.epoch.map((e, i) => ({ x: e, y: h.train_loss[i] })) },
        { name: "Test", color: classColor(1), points: h.epoch.map((e, i) => ({ x: e, y: h.test_loss[i] })) },
      ],
      height: 240, xLabel: "Epoch", yLabel: "Loss",
    });
    Charts.lineChart(document.getElementById("accChart"), {
      series: [
        { name: "Train", color: classColor(0), points: h.epoch.map((e, i) => ({ x: e, y: h.train_acc[i] })) },
        { name: "Test", color: classColor(1), points: h.epoch.map((e, i) => ({ x: e, y: h.test_acc[i] })) },
      ],
      height: 240, xLabel: "Epoch", yLabel: "Accuracy", yDomain: [0, 1], valueFmt: (v) => v.toFixed(1),
    });

    // feature importance
    const top = D.feature_importance.slice(0, 20);
    Charts.barChart(document.getElementById("importanceChart"), {
      data: top.map((f) => ({ label: f.gene.replace("gene_", "g"), values: [f.rf_importance] })),
      showLegend: false, height: 260, rotateLabels: true, valueFmt: (v) => v.toFixed(3),
    });
    document.getElementById("importanceChart").querySelectorAll("rect.mark-bar").forEach((r) => r.setAttribute("fill", "var(--series-1)"));
  }

  /* ============================ Predict (live, client-side inference) ============================ */
  function relu(v) { return v.map((x) => Math.max(0, x)); }
  function matVec(W, x, b) {
    return W.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + (b ? b[i] : 0));
  }
  function softmax(v) {
    const m = Math.max(...v);
    const ex = v.map((x) => Math.exp(x - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((x) => x / s);
  }
  function standardize(x) {
    return x.map((v, i) => (v - D.scaler.mean[i]) / D.scaler.scale[i]);
  }
  function predictLogReg(rawX) {
    const xs = standardize(rawX);
    const logits = matVec(D.weights.logreg.coef, xs, D.weights.logreg.intercept);
    return softmax(logits);
  }
  function predictMLP(rawX) {
    const xs = standardize(rawX);
    const w = D.weights.mlp;
    const h1 = relu(matVec(w.W1, xs, w.b1));
    const h2 = relu(matVec(w.W2, h1, w.b2));
    const out = matVec(w.W3, h2, w.b3);
    return softmax(out);
  }

  const TOP_N_SLIDERS = 16;
  function renderPredict() {
    const testSamples = D.samples.filter((s) => s.split === "test");
    const select = document.getElementById("sampleSelect");
    testSamples.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id} (true: ${s.label})`;
      select.appendChild(opt);
    });

    const sliderGenes = D.feature_importance.slice(0, TOP_N_SLIDERS).map((f) => f.gene);
    const geneIndex = {};
    D.gene_names.forEach((g, i) => (geneIndex[g] = i));

    let currentSample = testSamples[0];
    let currentValues = currentSample.values.slice();
    let currentModel = "logreg";

    const sliderContainer = document.getElementById("sliderList");
    sliderGenes.forEach((gene) => {
      const gi = geneIndex[gene];
      const row = document.createElement("div");
      row.className = "slider-row";
      const min = D.gene_ranges.min[gi], max = D.gene_ranges.max[gi];
      row.innerHTML = `
        <span class="slider-gene">${gene}</span>
        <input type="range" min="${min}" max="${max}" step="${((max - min) / 200).toFixed(4)}" data-gi="${gi}">
        <span class="slider-val" data-gi-val="${gi}"></span>`;
      sliderContainer.appendChild(row);
    });
    const sliderInputs = [...sliderContainer.querySelectorAll("input[type=range]")];

    function syncSlidersFromValues() {
      sliderInputs.forEach((inp) => {
        const gi = +inp.dataset.gi;
        const v = currentValues[gi];
        inp.value = v;
        const min = +inp.min, max = +inp.max;
        inp.style.setProperty("--pct", `${((v - min) / (max - min)) * 100}%`);
        sliderContainer.querySelector(`[data-gi-val="${gi}"]`).textContent = v.toFixed(2);
      });
    }

    function runPrediction() {
      const proba = currentModel === "logreg" ? predictLogReg(currentValues) : predictMLP(currentValues);
      const order = classes.map((c, i) => i).sort((a, b) => proba[b] - proba[a]);
      const predIdx = order[0];
      const predClass = classes[predIdx];
      const correct = predClass === currentSample.label;

      document.getElementById("predictClass").textContent = predClass;
      document.getElementById("predictClass").style.color = classColor(predIdx);
      document.getElementById("predictConf").textContent = `${pct(proba[predIdx])} confidence`;
      const matchBadge = document.getElementById("predictMatch");
      matchBadge.textContent = correct ? `✓ matches true label (${currentSample.label})` : `✗ true label: ${currentSample.label}`;
      matchBadge.className = "predict-match " + (correct ? "correct" : "incorrect");

      const barsEl = document.getElementById("probBars");
      barsEl.innerHTML = "";
      classes.forEach((c, i) => {
        const row = document.createElement("div");
        row.className = "prob-bar-row";
        row.innerHTML = `
          <span class="prob-bar-label" style="color:${classColor(i)}">${c}</span>
          <span class="prob-bar-track"><span class="prob-bar-fill" style="width:${(proba[i] * 100).toFixed(1)}%;background:${classColor(i)}"></span></span>
          <span class="prob-bar-val">${(proba[i] * 100).toFixed(1)}%</span>`;
        barsEl.appendChild(row);
      });
    }

    function loadSample(id) {
      currentSample = D.samples.find((s) => s.id === id);
      currentValues = currentSample.values.slice();
      select.value = id;
      syncSlidersFromValues();
      runPrediction();
    }

    select.addEventListener("change", () => loadSample(select.value));
    sliderInputs.forEach((inp) => {
      inp.addEventListener("input", () => {
        const gi = +inp.dataset.gi;
        currentValues[gi] = +inp.value;
        const min = +inp.min, max = +inp.max;
        inp.style.setProperty("--pct", `${((+inp.value - min) / (max - min)) * 100}%`);
        sliderContainer.querySelector(`[data-gi-val="${gi}"]`).textContent = (+inp.value).toFixed(2);
        runPrediction();
      });
    });
    document.getElementById("resetSampleBtn").addEventListener("click", () => loadSample(currentSample.id));
    document.getElementById("randomSampleBtn").addEventListener("click", () => {
      const s = testSamples[Math.floor(Math.random() * testSamples.length)];
      loadSample(s.id);
    });
    document.querySelectorAll("#predictModelSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll("#predictModelSeg button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        currentModel = b.dataset.model;
        runPrediction();
      });
    });

    loadSample(currentSample.id);
  }

  selectTab("overview");
})();
