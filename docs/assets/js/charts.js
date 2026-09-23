/*
 * The Marvel Graph Diaries — shared interactive chart helpers (Plotly.js).
 * Renders network graphs and degree histograms themed to the site's
 * maroon/red comic palette, with hover tooltips and zoom/pan.
 */
(function (global) {
  "use strict";

  const THEME = {
    paper: "transparent",
    plot: "transparent",
    font: "#f7f0e6",
    grid: "rgba(247,240,230,0.08)",
    faint: "#b3897e",
    red: "#ff3b3b",
    redDeep: "#d4181f",
    gold: "#e0b45c",
    steel: "#7fa8d9",
    tan: "#d9b8ac",
    gray: "#9a8f8a",
  };

  const ROLE_COLORS = {
    giant: THEME.red,
    island: "#4a8a6a",
    isolate: THEME.gray,
  };

  const DIST_COLORS = {
    0: THEME.red,
    1: THEME.gold,
    2: THEME.steel,
    3: THEME.tan,
    "-1": THEME.gray,
  };

  function baseLayout(overrides) {
    return Object.assign(
      {
        paper_bgcolor: THEME.paper,
        plot_bgcolor: THEME.plot,
        font: { color: THEME.font, family: "Inter, sans-serif", size: 12 },
        margin: { l: 40, r: 20, t: 10, b: 40 },
        showlegend: false,
        hoverlabel: {
          bgcolor: "#3a0d10",
          bordercolor: THEME.red,
          font: { color: THEME.font, family: "Inter, sans-serif" },
        },
      },
      overrides || {}
    );
  }

  const baseConfig = {
    displaylogo: false,
    modeBarButtonsToRemove: [
      "select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
    ],
    responsive: true,
  };

  /**
   * Render an interactive network graph.
   * @param {string} elId - target div id
   * @param {object} data - {nodes: [{id,name,x,y,in,out,role|dist}], edges: [{source,target}]}
   * @param {object} opts - {colorBy: 'role'|'dist', sizeBy: fn(node)->size, title}
   */
  function renderNetwork(elId, data, opts) {
    opts = opts || {};
    const colorBy = opts.colorBy || "role";
    const nodeById = {};
    data.nodes.forEach(n => { nodeById[n.id] = n; });

    // edge traces: one thin gray line trace covering all edges (fast, single trace)
    const edgeX = [];
    const edgeY = [];
    data.edges.forEach(e => {
      const s = nodeById[e.source];
      const t = nodeById[e.target];
      if (!s || !t) return;
      edgeX.push(s.x, t.x, null);
      edgeY.push(s.y, t.y, null);
    });

    const edgeTrace = {
      x: edgeX, y: edgeY,
      mode: "lines",
      line: { color: "rgba(247,240,230,0.15)", width: 0.6 },
      hoverinfo: "skip",
      type: "scatter",
    };

    const colorMap = colorBy === "dist" ? DIST_COLORS : ROLE_COLORS;
    const nodeColors = data.nodes.map(n => colorMap[n[colorBy]] || THEME.gray);
    const nodeSizes = data.nodes.map(n => {
      if (opts.sizeBy) return opts.sizeBy(n);
      return 4 + Math.sqrt((n.in || 0) + (n.out || 0)) * 3;
    });
    const hoverText = data.nodes.map(n => {
      if (colorBy === "dist") {
        const distLabel = n.dist === -1 ? "unreachable" : n.dist + " hop" + (n.dist === 1 ? "" : "s") + " from Spider-Man";
        return `<b>${n.name}</b><br>${distLabel}<br>in: ${n.in} · out: ${n.out}`;
      }
      return `<b>${n.name}</b><br>in: ${n.in} · out: ${n.out}<br>role: ${n.role}`;
    });

    const nodeTrace = {
      x: data.nodes.map(n => n.x),
      y: data.nodes.map(n => n.y),
      mode: "markers",
      marker: {
        color: nodeColors,
        size: nodeSizes,
        line: { color: "#f7f0e6", width: 0.5 },
        opacity: 0.92,
      },
      text: hoverText,
      hoverinfo: "text",
      type: "scatter",
    };

    const layout = baseLayout({
      xaxis: { visible: false, fixedrange: false },
      yaxis: { visible: false, fixedrange: false, scaleanchor: "x" },
      margin: { l: 10, r: 10, t: 10, b: 10 },
      dragmode: "pan",
    });

    Plotly.newPlot(elId, [edgeTrace, nodeTrace], layout, Object.assign({}, baseConfig, { scrollZoom: true }));
  }

  /**
   * Render an interactive degree histogram (in vs out, side by side).
   * @param {string} elId
   * @param {object} data - {in_degree: [...], out_degree: [...]}
   */
  function renderDegreeHist(elId, data) {
    function histTrace(vals, color, name, xaxis, yaxis) {
      return {
        x: vals,
        type: "histogram",
        marker: { color: color, line: { color: "#3a0d10", width: 1 } },
        name: name,
        xaxis: xaxis,
        yaxis: yaxis,
        hovertemplate: name + " degree %{x}<br>%{y} characters<extra></extra>",
        xbins: { start: -0.5, size: 1 },
      };
    }

    const traces = [
      histTrace(data.in_degree, THEME.red, "In-degree", "x", "y"),
      histTrace(data.out_degree, THEME.steel, "Out-degree", "x2", "y2"),
    ];

    const layout = baseLayout({
      grid: { rows: 1, columns: 2, pattern: "independent" },
      xaxis: { title: "in-degree", gridcolor: THEME.grid, zeroline: false },
      yaxis: { title: "characters", gridcolor: THEME.grid, zeroline: false },
      xaxis2: { title: "out-degree", gridcolor: THEME.grid, zeroline: false },
      yaxis2: { gridcolor: THEME.grid, zeroline: false },
      margin: { l: 50, r: 20, t: 20, b: 45 },
      bargap: 0.1,
    });

    Plotly.newPlot(elId, traces, layout, baseConfig);
  }

  /**
   * Render a simple distance-bucket bar chart (clickable).
   * @param {string} elId
   * @param {object} data - {distances: [0,1,2,3], counts: [1,106,152,18]}
   * @param {function} onClick - called with distance value when a bar is clicked
   */
  function renderDistanceHist(elId, data, onClick) {
    const trace = {
      x: data.distances.map(String),
      y: data.counts,
      type: "bar",
      marker: { color: data.distances.map(d => DIST_COLORS[d] || THEME.gray), line: { color: "#3a0d10", width: 1 } },
      text: data.counts.map(String),
      textposition: "outside",
      textfont: { color: THEME.font },
      hovertemplate: "%{x} hops<br>%{y} characters<extra></extra>",
    };
    const layout = baseLayout({
      xaxis: { title: "hops from Spider-Man", gridcolor: THEME.grid, zeroline: false },
      yaxis: { title: "characters", gridcolor: THEME.grid, zeroline: false },
      margin: { l: 55, r: 20, t: 20, b: 45 },
    });
    Plotly.newPlot(elId, [trace], layout, baseConfig);
    if (onClick) {
      document.getElementById(elId).on("plotly_click", function (evt) {
        const d = parseInt(evt.points[0].x, 10);
        onClick(d);
      });
    }
  }

  /**
   * Render a fragmentation / robustness chart: giant-component size vs.
   * number of nodes removed, for several removal orders, with a marker
   * at the step where a named node (e.g. Spider-Man) was removed.
   * @param {string} elId
   * @param {object} data - {steps, degree_order, betweenness_order,
   *   random_avg, spiderman_step_degree, spiderman_step_betweenness}
   */
  function renderFragmentation(elId, data) {
    const traces = [
      {
        x: data.steps, y: data.degree_order,
        mode: "lines", name: "Remove by degree",
        line: { color: THEME.red, width: 2.5 },
        hovertemplate: "Removed %{x} (by degree)<br>Giant component: %{y}<extra></extra>",
      },
      {
        x: data.steps, y: data.betweenness_order,
        mode: "lines", name: "Remove by betweenness",
        line: { color: THEME.gold, width: 2.5 },
        hovertemplate: "Removed %{x} (by betweenness)<br>Giant component: %{y}<extra></extra>",
      },
      {
        x: data.steps, y: data.random_avg,
        mode: "lines", name: "Random removal (avg of 30)",
        line: { color: THEME.steel, width: 2, dash: "dot" },
        hovertemplate: "Removed %{x} (random)<br>Giant component: %{y:.1f}<extra></extra>",
      },
      {
        x: [data.spiderman_step_degree],
        y: [data.degree_order[data.spiderman_step_degree - 1]],
        mode: "markers", name: "Spider-Man removed (degree order)",
        marker: { color: THEME.red, size: 11, symbol: "star", line: { color: "#fff", width: 1 } },
        hovertemplate: "Spider-Man removed here<br>Giant component: %{y}<extra></extra>",
        showlegend: false,
      },
    ];

    const layout = baseLayout({
      xaxis: { title: "characters removed", gridcolor: THEME.grid, zeroline: false },
      yaxis: { title: "giant component size", gridcolor: THEME.grid, zeroline: false },
      margin: { l: 55, r: 20, t: 10, b: 45 },
      legend: { orientation: "h", y: -0.25, font: { size: 11 } },
    });

    Plotly.newPlot(elId, traces, layout, baseConfig);
  }

  global.MarvelCharts = {
    renderNetwork: renderNetwork,
    renderDegreeHist: renderDegreeHist,
    renderDistanceHist: renderDistanceHist,
    renderFragmentation: renderFragmentation,
    THEME: THEME,
    ROLE_COLORS: ROLE_COLORS,
    DIST_COLORS: DIST_COLORS,
  };
})(window);
