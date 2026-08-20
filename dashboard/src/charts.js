/* Small dependency-free SVG chart toolkit. Vanilla JS + SVG only (no CDN deps),
   so the dashboard works fully offline and inside a locked-down sandbox. */
(function (global) {
  const SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    const node = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  const SERIES_VARS = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];
  function seriesColor(i) { return `var(${SERIES_VARS[i % SERIES_VARS.length]})`; }

  // ---------- tooltip singleton ----------
  let tooltipEl = null;
  function ensureTooltip() {
    if (tooltipEl) return tooltipEl;
    tooltipEl = document.createElement("div");
    tooltipEl.className = "tooltip";
    tooltipEl.setAttribute("role", "status");
    document.body.appendChild(tooltipEl);
    return tooltipEl;
  }
  function showTooltip(x, y, html) {
    const t = ensureTooltip();
    t.innerHTML = "";
    t.appendChild(html);
    t.classList.add("visible");
    const pad = 14;
    let left = x + pad, top = y + pad;
    const rect = t.getBoundingClientRect();
    if (left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
    if (top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
    t.style.transform = `translate(${left}px, ${top}px)`;
  }
  function hideTooltip() { if (tooltipEl) tooltipEl.classList.remove("visible"); }

  function ttRow(colorVar, label, value, muted) {
    const row = document.createElement("div");
    row.className = "tooltip-row";
    if (colorVar) {
      const key = document.createElement("span");
      key.className = "tooltip-key";
      key.style.background = colorVar;
      row.appendChild(key);
    }
    const lab = document.createElement("span");
    lab.textContent = label;
    if (muted) lab.className = "tooltip-muted";
    row.appendChild(lab);
    if (value !== undefined) {
      const val = document.createElement("span");
      val.className = "tooltip-val";
      val.style.marginLeft = "auto";
      val.textContent = value;
      row.appendChild(val);
    }
    return row;
  }
  function ttBlock(titleText, rows) {
    const wrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "tooltip-title";
    title.textContent = titleText;
    wrap.appendChild(title);
    rows.forEach((r) => wrap.appendChild(r));
    return wrap;
  }

  // ---------- number formatting / ticks ----------
  function niceTicks(min, max, count) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const step0 = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    let step;
    if (norm < 1.5) step = 1 * mag;
    else if (norm < 3) step = 2 * mag;
    else if (norm < 7) step = 5 * mag;
    else step = 10 * mag;
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step * 0.001; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
    return ticks;
  }
  function fmtNum(v) {
    if (Math.abs(v) >= 1000) return (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "k";
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(2).replace(/\.00$/, "");
  }

  function legend(container, items, onToggle) {
    const wrap = document.createElement("div");
    wrap.className = "legend";
    items.forEach((it, i) => {
      const btn = document.createElement("button");
      btn.className = "legend-item";
      btn.type = "button";
      btn.setAttribute("aria-pressed", "true");
      const sw = document.createElement("span");
      sw.className = "legend-swatch" + (it.line ? " line" : "");
      sw.style.background = it.color;
      btn.appendChild(sw);
      const txt = document.createElement("span");
      txt.textContent = it.label;
      btn.appendChild(txt);
      btn.addEventListener("click", () => {
        const on = btn.getAttribute("aria-pressed") === "true";
        btn.setAttribute("aria-pressed", on ? "false" : "true");
        onToggle && onToggle(i, !on);
      });
      wrap.appendChild(btn);
    });
    container.appendChild(wrap);
    return wrap;
  }

  // =====================================================================
  // Bar chart, supports single or grouped series. data: [{label, values:[n,...]}]
  // =====================================================================
  function barChart(container, opts) {
    const {
      data, seriesNames = [], height = 260, yLabel = "", valueFmt = fmtNum,
      colorOffset = 0, showLegend = true, yDomain = null, rotateLabels = false,
    } = opts;
    container.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    container.appendChild(wrap);

    const width = 640;
    const margin = { top: 14, right: 10, bottom: rotateLabels ? 46 : 28, left: 40 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    const nSeries = data[0] ? data[0].values.length : 1;
    let maxV = 0, minV = 0;
    data.forEach((d) => d.values.forEach((v) => { if (v > maxV) maxV = v; if (v < minV) minV = v; }));
    const ticks = yDomain ? niceTicks(yDomain[0], yDomain[1], 4) : niceTicks(Math.min(0, minV), maxV, 4);
    const yMin = ticks[0], yMax = ticks[ticks.length - 1];
    const yScale = (v) => plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "xMidYMid meet" });
    const g = el("g", { transform: `translate(${margin.left},${margin.top})` });
    svg.appendChild(g);

    ticks.forEach((t) => {
      const y = yScale(t);
      g.appendChild(el("line", { x1: 0, x2: plotW, y1: y, y2: y, class: t === 0 ? "baseline" : "tick-line" }));
      const lab = el("text", { x: -8, y: y + 3.5, class: "axis-label", "text-anchor": "end" });
      lab.textContent = valueFmt(t);
      g.appendChild(lab);
    });

    const bandW = plotW / data.length;
    const groupPad = bandW * 0.18;
    const barGap = 2;
    const barW = Math.min(24, (bandW - groupPad * 2 - barGap * (nSeries - 1)) / nSeries);
    const totalBarsW = barW * nSeries + barGap * (nSeries - 1);
    const zeroY = yScale(0);

    data.forEach((d, di) => {
      const groupX = di * bandW + (bandW - totalBarsW) / 2;
      d.values.forEach((v, si) => {
        const x = groupX + si * (barW + barGap);
        const y = yScale(Math.max(v, 0));
        const barH = Math.abs(yScale(v) - zeroY);
        const color = seriesColor(si + colorOffset);
        const rect = el("rect", {
          x, y: v >= 0 ? y : zeroY, width: barW, height: Math.max(barH, 0.5),
          rx: 4, fill: color, class: "mark-bar",
        });
        g.appendChild(rect);
        rect.addEventListener("pointermove", (e) => {
          const rows = [ttRow(color, seriesNames[si] || d.label, valueFmt(v))];
          showTooltip(e.clientX, e.clientY, ttBlock(seriesNames.length ? d.label : d.label, rows));
        });
        rect.addEventListener("pointerleave", hideTooltip);
      });
      const lx = di * bandW + bandW / 2;
      const lab = el("text", { x: lx, y: plotH + (rotateLabels ? 14 : 18), class: "axis-label", "text-anchor": rotateLabels ? "end" : "middle" });
      lab.textContent = d.label;
      if (rotateLabels) lab.setAttribute("transform", `rotate(-40 ${lx} ${plotH + 14})`);
      g.appendChild(lab);
    });

    wrap.appendChild(svg);
    if (showLegend && seriesNames.length > 1) {
      legend(wrap, seriesNames.map((n, i) => ({ label: n, color: seriesColor(i + colorOffset) })), (i, on) => {
        const rects = g.querySelectorAll("rect.mark-bar");
        rects.forEach((r, idx) => { if (idx % nSeries === i) r.style.display = on ? "" : "none"; });
      });
    }
    return svg;
  }

  // =====================================================================
  // Line chart. series: [{name, color, points:[{x,y}], dashed?}]
  // =====================================================================
  function lineChart(container, opts) {
    const {
      series, height = 260, xDomain, yDomain, xLabel = "", yLabel = "",
      valueFmt = fmtNum, showLegend = true, area = false, xTickFmt = fmtNum, diagonal = false,
    } = opts;
    container.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    container.appendChild(wrap);

    const width = 640;
    const margin = { top: 14, right: 16, bottom: 30, left: 42 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    let xs = [], ys = [];
    series.forEach((s) => s.points.forEach((p) => { xs.push(p.x); ys.push(p.y); }));
    const xTicks = niceTicks(xDomain ? xDomain[0] : Math.min(...xs), xDomain ? xDomain[1] : Math.max(...xs), 5);
    const yTicks = niceTicks(yDomain ? yDomain[0] : Math.min(0, ...ys), yDomain ? yDomain[1] : Math.max(...ys), 4);
    const xMin = xTicks[0], xMax = xTicks[xTicks.length - 1];
    const yMin = yTicks[0], yMax = yTicks[yTicks.length - 1];
    const xScale = (v) => ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (v) => plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "xMidYMid meet" });
    const g = el("g", { transform: `translate(${margin.left},${margin.top})` });
    svg.appendChild(g);

    yTicks.forEach((t) => {
      const y = yScale(t);
      g.appendChild(el("line", { x1: 0, x2: plotW, y1: y, y2: y, class: t === yTicks[0] ? "baseline" : "tick-line" }));
      const lab = el("text", { x: -8, y: y + 3.5, class: "axis-label", "text-anchor": "end" });
      lab.textContent = valueFmt(t);
      g.appendChild(lab);
    });
    xTicks.forEach((t) => {
      const x = xScale(t);
      const lab = el("text", { x, y: plotH + 18, class: "axis-label", "text-anchor": "middle" });
      lab.textContent = xTickFmt(t);
      g.appendChild(lab);
    });
    g.appendChild(el("line", { x1: 0, x2: plotW, y1: plotH, y2: plotH, class: "baseline" }));

    if (diagonal) {
      g.appendChild(el("line", { x1: xScale(0), y1: yScale(0), x2: xScale(1), y2: yScale(1), class: "tick-line", "stroke-dasharray": "3,3" }));
    }

    const hitLayer = el("g");

    series.forEach((s) => {
      const pts = s.points;
      const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${xScale(p.x).toFixed(2)},${yScale(p.y).toFixed(2)}`).join(" ");
      if (area) {
        const areaPath = `${path} L${xScale(pts[pts.length - 1].x).toFixed(2)},${yScale(yMin)} L${xScale(pts[0].x).toFixed(2)},${yScale(yMin)} Z`;
        g.appendChild(el("path", { d: areaPath, fill: s.color, class: "mark-area", stroke: "none" }));
      }
      g.appendChild(el("path", { d: path, class: "mark-line", stroke: s.color, "stroke-dasharray": s.dashed ? "5,4" : "" }));

      pts.forEach((p) => {
        const cx = xScale(p.x), cy = yScale(p.y);
        const hit = el("circle", { cx, cy, r: 12, class: "hit-area" });
        hit.addEventListener("pointermove", (e) => {
          const rows = [ttRow(s.color, s.name, valueFmt(p.y))];
          showTooltip(e.clientX, e.clientY, ttBlock(xTickFmt(p.x), rows));
        });
        hit.addEventListener("pointerleave", hideTooltip);
        hitLayer.appendChild(hit);
      });
      const last = pts[pts.length - 1];
      g.appendChild(el("circle", { cx: xScale(last.x), cy: yScale(last.y), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
    });
    g.appendChild(hitLayer);

    wrap.appendChild(svg);
    if (showLegend && series.length > 1) {
      legend(wrap, series.map((s) => ({ label: s.name, color: s.color, line: true })), (i, on) => {
        [...g.children].forEach(() => {});
      });
    }
    return svg;
  }

  // =====================================================================
  // Scatter chart with composite hue x shape encoding. series: [{name,color,shape,points:[{x,y,meta}]}]
  // =====================================================================
  const SHAPES = {
    circle: (cx, cy, r) => el("circle", { cx, cy, r }),
    square: (cx, cy, r) => el("rect", { x: cx - r * 0.85, y: cy - r * 0.85, width: r * 1.7, height: r * 1.7, rx: 1.5 }),
    triangle: (cx, cy, r) => {
      const h = r * 1.15;
      return el("polygon", { points: `${cx},${cy - h} ${cx - h * 0.98},${cy + h * 0.72} ${cx + h * 0.98},${cy + h * 0.72}` });
    },
    diamond: (cx, cy, r) => el("polygon", { points: `${cx},${cy - r * 1.2} ${cx + r * 1.2},${cy} ${cx},${cy + r * 1.2} ${cx - r * 1.2},${cy}` }),
    cross: (cx, cy, r) => {
      const w = r * 0.42;
      return el("path", { d: `M${cx - r},${cy - w} h${r - w} v${-(r - w)} h${w * 2} v${r - w} h${r - w} v${w * 2} h${-(r - w)} v${r - w} h${-w * 2} v${-(r - w)} h${-(r - w)} Z` });
    },
  };
  const SHAPE_ORDER = ["circle", "square", "triangle", "diamond", "cross"];

  function scatterChart(container, opts) {
    const { series, height = 380, xLabel = "", yLabel = "", onPointClick } = opts;
    container.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    container.appendChild(wrap);

    const width = 640;
    const margin = { top: 10, right: 10, bottom: 30, left: 36 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    let xs = [], ys = [];
    series.forEach((s) => s.points.forEach((p) => { xs.push(p.x); ys.push(p.y); }));
    const padX = (Math.max(...xs) - Math.min(...xs)) * 0.08 || 1;
    const padY = (Math.max(...ys) - Math.min(...ys)) * 0.08 || 1;
    const xTicks = niceTicks(Math.min(...xs) - padX, Math.max(...xs) + padX, 5);
    const yTicks = niceTicks(Math.min(...ys) - padY, Math.max(...ys) + padY, 5);
    const xMin = xTicks[0], xMax = xTicks[xTicks.length - 1];
    const yMin = yTicks[0], yMax = yTicks[yTicks.length - 1];
    const xScale = (v) => ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (v) => plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "xMidYMid meet" });
    const g = el("g", { transform: `translate(${margin.left},${margin.top})` });
    svg.appendChild(g);

    yTicks.forEach((t) => g.appendChild(el("line", { x1: 0, x2: plotW, y1: yScale(t), y2: yScale(t), class: "tick-line" })));
    xTicks.forEach((t) => g.appendChild(el("line", { x1: xScale(t), x2: xScale(t), y1: 0, y2: plotH, class: "tick-line" })));
    g.appendChild(el("rect", { x: 0, y: 0, width: plotW, height: plotH, fill: "none", stroke: "var(--axis)", "stroke-width": 1 }));

    const groups = {};
    series.forEach((s, si) => {
      const shape = SHAPE_ORDER[si % SHAPE_ORDER.length];
      const grp = el("g", { "data-series": s.name });
      groups[s.name] = grp;
      s.points.forEach((p) => {
        const cx = xScale(p.x), cy = yScale(p.y);
        const r = p.split === "test" ? 4.6 : 3.6;
        const dot = SHAPES[shape](cx, cy, r);
        dot.setAttribute("fill", p.split === "test" ? s.color : s.color);
        dot.setAttribute("fill-opacity", p.split === "test" ? "0.95" : "0.45");
        dot.setAttribute("stroke", "var(--surface-1)");
        dot.setAttribute("stroke-width", p.split === "test" ? 1.4 : 0.8);
        dot.setAttribute("class", "mark-dot");
        grp.appendChild(dot);
        const hit = el("circle", { cx, cy, r: 12, class: "hit-area" });
        hit.addEventListener("pointermove", (e) => {
          const rows = [
            ttRow(s.color, "Class", s.name),
            ttRow(null, "Sample", p.sample || ""),
            ttRow(null, "Split", p.split),
          ];
          showTooltip(e.clientX, e.clientY, ttBlock(p.sample || s.name, rows));
        });
        hit.addEventListener("pointerleave", hideTooltip);
        if (onPointClick) hit.addEventListener("click", () => onPointClick(p));
        grp.appendChild(hit);
      });
      g.appendChild(grp);
    });

    const xl = el("text", { x: plotW / 2, y: plotH + 24, class: "axis-label", "text-anchor": "middle" });
    xl.textContent = xLabel;
    g.appendChild(xl);
    const yl = el("text", { x: -plotH / 2, y: -24, class: "axis-label", "text-anchor": "middle", transform: `rotate(-90)` });
    yl.textContent = yLabel;
    g.appendChild(yl);

    wrap.appendChild(svg);
    const legendItems = series.map((s, i) => ({ label: s.name, color: s.color, shape: SHAPE_ORDER[i % SHAPE_ORDER.length] }));
    const legendEl = document.createElement("div");
    legendEl.className = "legend";
    legendItems.forEach((it, i) => {
      const btn = document.createElement("button");
      btn.className = "legend-item";
      btn.type = "button";
      btn.setAttribute("aria-pressed", "true");
      const miniSvg = document.createElementNS(SVGNS, "svg");
      miniSvg.setAttribute("width", "12");
      miniSvg.setAttribute("height", "12");
      miniSvg.setAttribute("viewBox", "0 0 12 12");
      const shapeNode = SHAPES[it.shape](6, 6, 4.4);
      shapeNode.setAttribute("fill", it.color);
      miniSvg.appendChild(shapeNode);
      btn.appendChild(miniSvg);
      const txt = document.createElement("span");
      txt.textContent = it.label;
      btn.appendChild(txt);
      btn.addEventListener("click", () => {
        const on = btn.getAttribute("aria-pressed") === "true";
        btn.setAttribute("aria-pressed", on ? "false" : "true");
        groups[it.label].style.display = on ? "none" : "";
      });
      legendEl.appendChild(btn);
    });
    wrap.appendChild(legendEl);
    return svg;
  }

  // =====================================================================
  // Heatmap (confusion matrix). matrix: rows[i][j], rowLabels, colLabels
  // =====================================================================
  function heatmap(container, opts) {
    const { matrix, rowLabels, colLabels, height = 320, rowTitle = "Actual", colTitle = "Predicted" } = opts;
    container.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    container.appendChild(wrap);

    const n = matrix.length;
    const width = 560;
    const margin = { top: 16, right: 10, bottom: 44, left: 78 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const cellW = plotW / n, cellH = plotH / n;
    const maxV = Math.max(...matrix.flat());

    const seqSteps = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];
    function colorFor(v) {
      const t = maxV === 0 ? 0 : v / maxV;
      const idx = Math.min(seqSteps.length - 1, Math.round(t * (seqSteps.length - 1)));
      return `var(${seqSteps[idx]})`;
    }

    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "xMidYMid meet" });
    const g = el("g", { transform: `translate(${margin.left},${margin.top})` });
    svg.appendChild(g);

    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const v = matrix[r][c];
        const x = c * cellW, y = r * cellH;
        const rect = el("rect", {
          x: x + 1, y: y + 1, width: cellW - 2, height: cellH - 2, rx: 4,
          fill: v === 0 ? "var(--gridline)" : colorFor(v), class: "mark-bar",
        });
        g.appendChild(rect);
        const light = maxV > 0 && v / maxV > 0.55;
        const txt = el("text", {
          x: x + cellW / 2, y: y + cellH / 2 + 4, "text-anchor": "middle",
          fill: v === 0 ? "var(--text-muted)" : (light ? "#ffffff" : "var(--text-primary)"),
          "font-size": "12", "font-weight": r === c ? "800" : "600",
        });
        txt.textContent = v;
        g.appendChild(txt);
        rect.addEventListener("pointermove", (e) => {
          const rows = [ttRow(null, "Actual", rowLabels[r]), ttRow(null, "Predicted", colLabels[c]), ttRow(null, "Count", String(v))];
          showTooltip(e.clientX, e.clientY, ttBlock(r === c ? "Correct" : "Misclassified", rows));
        });
        rect.addEventListener("pointerleave", hideTooltip);
      }
    }
    rowLabels.forEach((lab, r) => {
      const t = el("text", { x: -8, y: r * cellH + cellH / 2 + 4, class: "axis-label", "text-anchor": "end" });
      t.textContent = lab;
      g.appendChild(t);
    });
    colLabels.forEach((lab, c) => {
      const t = el("text", { x: c * cellW + cellW / 2, y: plotH + 16, class: "axis-label", "text-anchor": "middle" });
      t.textContent = lab;
      g.appendChild(t);
    });
    const rt = el("text", { x: -plotH / 2, y: -60, class: "axis-label", "text-anchor": "middle", transform: "rotate(-90)" });
    rt.textContent = rowTitle;
    g.appendChild(rt);
    const ct = el("text", { x: plotW / 2, y: plotH + 34, class: "axis-label", "text-anchor": "middle" });
    ct.textContent = colTitle;
    g.appendChild(ct);

    wrap.appendChild(svg);
    return svg;
  }

  global.Charts = { seriesColor, barChart, lineChart, scatterChart, heatmap, showTooltip, hideTooltip, fmtNum, niceTicks };
})(window);
