/* ChaseIQ frontend: vanilla JavaScript, hash routing, SVG charts. No external libraries. */
"use strict";

// ---------------------------------------------------------------- state
const state = {
  filters: { match_type: "", season: "", event: "", team: "" },
  meta: null,            // /api/filters
  status: null,          // /api/status
  matchId: null,
  matchOffset: 0,
  explorerTable: "matches",
  explorerOffset: 0,
  scenario: { target: 160, runs: 60, balls: 60, wickets: 6 },
};
let routeToken = 0;

// ---------------------------------------------------------------- helpers
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(d)}%`);
const probPct = (p, d = 1) => (p === null || p === undefined ? "—" : `${(Number(p) * 100).toFixed(d)}%`);

async function api(path, params = {}) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") qs.set(k, v); });
  const url = `/api/${path}${qs.toString() ? "?" + qs : ""}`;
  const res = await fetch(url);
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

function filterParams(extra = {}) { return { ...state.filters, ...extra }; }

function kpi(label, value, sub = "") {
  return `<div class="card kpi"><span class="label">${esc(label)}</span><span class="value">${value}</span>${sub ? `<span class="note">${sub}</span>` : ""}</div>`;
}

function table(columns, rows, opts = {}) {
  if (!rows || !rows.length) return `<p class="note">No rows for this selection.</p>`;
  const head = columns.map((c) => `<th class="${c.align === "left" ? "left" : ""}">${esc(c.label)}</th>`).join("");
  const body = rows.map((r) => {
    const cls = opts.rowClass ? opts.rowClass(r) : "";
    const attrs = opts.rowAttr ? opts.rowAttr(r) : "";
    return `<tr class="${cls}" ${attrs}>${columns.map((c) => `<td class="${c.align === "left" ? "left" : ""}">${c.html ? c.html(r) : esc(c.fmt ? c.fmt(r[c.key]) : r[c.key] ?? "—")}</td>`).join("")}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function barRows(items, { label, value, format = (v) => fmt(v, 1), max = null, neg = false }) {
  if (!items.length) return `<p class="note">No data.</p>`;
  const top = max ?? Math.max(...items.map((i) => Math.abs(i[value] ?? 0)), 1);
  return items.map((i) => {
    const v = i[value] ?? 0;
    const w = Math.min(100, (Math.abs(v) / top) * 100);
    return `<div class="bar-row"><span>${esc(i[label])}</span><div class="bar-track"><div class="bar-fill ${neg && v < 0 ? "neg" : ""}" style="width:${w}%"></div></div><span class="num" style="text-align:right">${format(v)}</span></div>`;
  }).join("");
}

// SVG bar chart
function barChart(items, { xKey, yKey, height = 220, color = "#20D6A1", yFormat = (v) => fmt(v, 0) }) {
  if (!items || !items.length) return `<p class="note">No data.</p>`;
  const W = 640, H = height, pad = { l: 52, r: 12, t: 12, b: 34 };
  const max = Math.max(...items.map((d) => d[yKey] || 0), 1);
  const bw = (W - pad.l - pad.r) / items.length;
  const innerH = H - pad.t - pad.b;
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = (max * i) / 4;
    const y = H - pad.b - (v / max) * innerH;
    grid += `<line class="grid-line" x1="${pad.l}" x2="${W - pad.r}" y1="${y}" y2="${y}"/><text x="${pad.l - 6}" y="${y + 4}" text-anchor="end">${yFormat(v)}</text>`;
  }
  const step = Math.ceil(items.length / 14);
  let bars = "", labels = "";
  items.forEach((d, i) => {
    const v = d[yKey] || 0;
    const h = (v / max) * innerH;
    const x = pad.l + i * bw + bw * 0.15;
    bars += `<rect x="${x.toFixed(1)}" y="${(H - pad.b - h).toFixed(1)}" width="${(bw * 0.7).toFixed(1)}" height="${h.toFixed(1)}" rx="3" fill="${color}"><title>${esc(d[xKey])}: ${yFormat(v)}</title></rect>`;
    if (i % step === 0) labels += `<text x="${(x + bw * 0.35).toFixed(1)}" y="${H - 12}" text-anchor="middle">${esc(d[xKey])}</text>`;
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${grid}${bars}${labels}</svg>`;
}

// SVG line chart with several series: [{name, color, points:[{x,y}]}]
function lineChart(series, { height = 260, yMin = null, yMax = null, yFormat = (v) => fmt(v, 0), xLabel = "", xLabels = null } = {}) {
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) return `<p class="note">No data.</p>`;
  const W = 640, H = height, pad = { l: 52, r: 14, t: 12, b: 34 };
  const x0 = Math.min(...pts.map((p) => p.x)), x1 = Math.max(...pts.map((p) => p.x));
  let y0 = yMin ?? Math.min(...pts.map((p) => p.y));
  let y1 = yMax ?? Math.max(...pts.map((p) => p.y));
  if (y1 === y0) y1 = y0 + 1;
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const sy = (y) => H - pad.b - ((y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = y0 + ((y1 - y0) * i) / 4;
    grid += `<line class="grid-line" x1="${pad.l}" x2="${W - pad.r}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${pad.l - 6}" y="${sy(v) + 4}" text-anchor="end">${yFormat(v)}</text>`;
  }
  let xt = "";
  const xl = xLabels;
  for (let i = 0; i <= 5; i++) {
    const v = x0 + ((x1 - x0) * i) / 5;
    const label = xl ? (xl[Math.round(v)] ?? "") : fmt(v, 0);
    xt += `<text x="${sx(v)}" y="${H - 12}" text-anchor="middle">${esc(label)}</text>`;
  }
  const paths = series.map((s) => {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
  }).join("");
  const axis = xLabel ? `<text x="${W - pad.r}" y="${H - 1}" text-anchor="end">${esc(xLabel)}</text>` : "";
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${grid}${xt}${paths}${axis}</svg>${legend(series)}`;
}

function legend(series) {
  if (series.length < 2) return "";
  return `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`;
}

// Radar chart: axes [{axis, a, b}] with values 0..1
function radarChart(axes, nameA, nameB) {
  const W = 360, H = 340, cx = 180, cy = 170, R = 120, n = axes.length;
  const ang = (i) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const pt = (i, v) => [cx + R * v * Math.cos(ang(i)), cy + R * v * Math.sin(ang(i))];
  let rings = "";
  [0.25, 0.5, 0.75, 1].forEach((r) => {
    rings += `<polygon points="${axes.map((_, i) => pt(i, r).join(",")).join(" ")}" fill="none" stroke="#22304D"/>`;
  });
  let spokes = "", labels = "";
  axes.forEach((a, i) => {
    const [x, y] = pt(i, 1);
    spokes += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#22304D"/>`;
    const [lx, ly] = pt(i, 1.22);
    labels += `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle">${esc(a.axis)}</text>`;
  });
  const poly = (key, color) => `<polygon points="${axes.map((a, i) => pt(i, a[key]).join(",")).join(" ")}" fill="${color}" fill-opacity="0.18" stroke="${color}" stroke-width="2"/>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${rings}${spokes}${poly("a", "#20D6A1")}${poly("b", "#48A9FF")}${labels}</svg>
    ${legend([{ name: nameA, color: "#20D6A1" }, { name: nameB, color: "#48A9FF" }])}`;
}

// Win-probability colour: red (0) -> amber (0.5) -> teal (1)
function probColor(p) {
  if (p === null || p === undefined) return "transparent";
  const stops = [[255, 107, 107], [245, 185, 74], [32, 214, 161]];
  const t = Math.max(0, Math.min(1, p));
  const [a, b, f] = t < 0.5 ? [stops[0], stops[1], t / 0.5] : [stops[1], stops[2], (t - 0.5) / 0.5];
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * f));
  return `rgba(${c[0]},${c[1]},${c[2]},0.45)`;
}

function heatGrid(rowLabels, colLabels, cell) {
  const head = `<tr><th></th>${colLabels.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  const body = rowLabels.map((r, i) => `<tr><td>${esc(r)}</td>${colLabels.map((_, j) => {
    const c = cell(i, j);
    return `<td style="background:${c.bg || "transparent"}" class="${c.empty ? "empty" : ""}" title="${esc(c.title || "")}">${c.text ?? "—"}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table class="heat"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

function sampleBadge(balls) {
  if (balls < 12) return `<span class="badge bad">Very small sample</span>`;
  if (balls < 30) return `<span class="badge warn">Small sample</span>`;
  return `<span class="badge good">Adequate sample</span>`;
}

function requireData() {
  if (!state.status || !state.status.data_loaded) {
    return `<div class="card"><h2>No data loaded yet</h2>
      <p>Upload the Cricsheet T20 JSON ZIP on the <a href="#/data">Data &amp; Upload</a> page, or place it in <code>data/raw/</code> and restart the app.</p></div>`;
  }
  return null;
}

// ---------------------------------------------------------------- shared inputs
function scenarioControls(onChange) {
  const s = state.scenario;
  return `<div class="controls" id="scenario-controls">
    <label class="field">Target<input type="number" id="sc-target" min="20" max="400" value="${s.target}"></label>
    <label class="field">Runs needed<input type="number" id="sc-runs" min="0" max="400" value="${s.runs}"></label>
    <label class="field">Balls left<input type="number" id="sc-balls" min="0" max="120" value="${s.balls}"></label>
    <label class="field">Wickets left<input type="number" id="sc-wkts" min="0" max="10" value="${s.wickets}"></label>
  </div>`;
}

function bindScenario(ctx, onChange) {
  let timer = null;
  const read = () => {
    state.scenario = {
      target: Number($("#sc-target").value), runs: Number($("#sc-runs").value),
      balls: Number($("#sc-balls").value), wickets: Number($("#sc-wkts").value),
    };
    return state.scenario;
  };
  ["sc-target", "sc-runs", "sc-balls", "sc-wkts"].forEach((id) => {
    $("#" + id).addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => { read(); if (ctx.token === routeToken) onChange(read()); }, 250);
    });
  });
}

function stateParams(s) {
  return { target: s.target, runs_needed: s.runs, balls_left: s.balls, wickets_left: s.wickets };
}

// ---------------------------------------------------------------- pages
async function pageOverview(ctx) {
  const d = await api("overview", filterParams());
  const seasons = d.matches_per_season;
  ctx.render(`
    <div class="kpi-row">
      ${kpi("Matches", fmt(d.matches, 0))}
      ${kpi("Decided", fmt(d.decided_matches, 0), "with a winner")}
      ${kpi("Deliveries", fmt(d.deliveries, 0))}
      ${kpi("Teams", fmt(d.teams, 0))}
      ${kpi("Chase win rate", pct(d.chase_win_rate), "chasing side won")}
      ${kpi("Avg 1st innings", fmt(d.avg_first_innings, 1), "runs")}
    </div>
    <div class="grid">
      <div class="card span-7"><h2>Matches per season</h2>${barChart(seasons, { xKey: "season", yKey: "matches" })}</div>
      <div class="card span-5"><h2>Top teams by wins</h2>${table([
        { key: "team", label: "Team", align: "left" }, { key: "wins", label: "Wins" }, { key: "win_pct", label: "Win %", fmt: (v) => pct(v) },
        { key: "chase_win_pct", label: "Chase %", fmt: (v) => pct(v) }], d.top_teams)}</div>
      <div class="card span-6"><h2>Top run scorers</h2>${table([
        { key: "player", label: "Batter" }, { key: "runs", label: "Runs", fmt: (v) => fmt(v, 0) },
        { key: "strike_rate", label: "SR", fmt: (v) => fmt(v, 1) }, { key: "average", label: "Avg", fmt: (v) => fmt(v, 1) }], d.top_batters)}</div>
      <div class="card span-6"><h2>Top wicket takers</h2>${table([
        { key: "player", label: "Bowler" }, { key: "wickets", label: "Wkts", fmt: (v) => fmt(v, 0) },
        { key: "economy", label: "Econ", fmt: (v) => fmt(v, 2) }, { key: "bowling_average", label: "Avg", fmt: (v) => fmt(v, 1) }], d.top_bowlers)}</div>
    </div>
    <p class="note" style="margin-top:12px">${esc(d.ranking_rule)} Formats in data: ${esc(Object.entries(d.match_types).map(([k, v]) => `${k} (${v})`).join(", "))}.</p>`);
}

async function pageMatch(ctx, params) {
  if (params.id) state.matchId = params.id;
  const list = await api("matches", { ...filterParams(), limit: 30, offset: state.matchOffset });
  if (!state.matchId && list.rows.length) state.matchId = list.rows[0].match_id;
  const detail = state.matchId ? await api("match", { id: state.matchId }) : null;
  const rows = list.rows;
  const listHtml = table([
    { key: "date", label: "Date" },
    { html: (r) => `${esc(r.team1)} v ${esc(r.team2)}<div class="note">${esc(r.event || "")}</div>`, label: "Match", align: "left" },
  ], rows, {
    rowClass: (r) => (r.match_id === state.matchId ? "clickable selected" : "clickable"),
    rowAttr: (r) => `data-id="${esc(r.match_id)}"`,
  });
  let detailHtml = `<p class="note">No match selected.</p>`;
  if (detail) {
    const inn = detail.innings;
    const result = detail.winner
      ? `${esc(detail.winner)} won ${detail.result.by_runs ? `by ${detail.result.by_runs} runs` : detail.result.by_wickets ? `by ${detail.result.by_wickets} wickets` : ""}${detail.result.method ? ` (${esc(detail.result.method)})` : ""}`
      : `No winner (${esc(detail.result.result || "no result")})`;
    const innCards = inn.map((i) => kpi(`Innings ${i.innings}: ${i.team}`, `${i.runs}/${i.wickets}`, `${Math.floor(i.legal_balls / 6)}.${i.legal_balls % 6} overs`)).join("");
    const wp = detail.winprob;
    const wpChart = wp.available
      ? lineChart([{ name: "Chasing side win probability", color: "#48A9FF", points: wp.points.map((p) => ({ x: p.over, y: p.probability })) }],
        { yMin: 0, yMax: 1, yFormat: (v) => `${Math.round(v * 100)}%`, xLabel: "overs" })
      : `<p class="note">${esc(wp.reason)}</p>`;
    const worm = lineChart(inn.map((i, idx) => ({ name: i.team, color: idx ? "#48A9FF" : "#20D6A1", points: i.worm.map((p) => ({ x: p.over, y: p.runs })) })), { xLabel: "overs" });
    const manhattan = inn.map((i) => `<div><h3>${esc(i.team)} runs per over</h3>${barChart(i.manhattan, { xKey: "over", yKey: "runs", height: 180, color: idx2color(i.innings) })}</div>`).join("");
    detailHtml = `
      <div class="card"><h2>${esc(detail.teams[0])} v ${esc(detail.teams[1])}</h2>
        <p class="sub">${esc(detail.date)} · ${esc(detail.event || "—")} · ${esc(detail.match_type)} · ${esc(detail.venue)}${detail.city ? ", " + esc(detail.city) : ""}</p>
        <p>${result}. Toss: ${esc(detail.toss.winner || "—")} chose to ${esc(detail.toss.decision || "—")}.</p>
      </div>
      <div class="kpi-row" style="margin-top:var(--gutter)">${innCards}</div>
      <div class="card"><h2>Worm (cumulative runs)</h2>${worm}</div>
      <div class="card"><h2>Manhattan (runs per over)</h2>${manhattan}</div>
      <div class="card"><h2>Win probability through the chase</h2>${wpChart}
        <p class="note">Model estimate after each legal ball of the chase. It is a historical-frequency model, not a live forecast.</p></div>`;
  }
  ctx.render(`
    <div class="controls">
      <label class="field">Match ID<input type="search" id="match-id" placeholder="e.g. ${esc(rows[0]?.match_id || "1001349")}" value="${esc(state.matchId || "")}"></label>
      <button type="button" id="match-go">Open</button>
      <div style="margin-left:auto" class="controls">
        <button type="button" class="ghost" id="match-prev" ${state.matchOffset ? "" : "disabled"}>Newer</button>
        <button type="button" class="ghost" id="match-next" ${rows.length < 30 ? "disabled" : ""}>Older</button>
      </div>
    </div>
    <div class="grid">
      <div class="card flat span-4"><h2>Recent matches (${fmt(list.total, 0)} in selection)</h2>${listHtml}</div>
      <div class="span-8" style="display:flex;flex-direction:column;gap:var(--gutter);min-width:0">${detailHtml}</div>
    </div>`);
  if (ctx.token !== routeToken) return;
  const go = () => { state.matchId = $("#match-id").value.trim() || state.matchId; location.hash = `#/match?id=${encodeURIComponent(state.matchId)}`; };
  $("#match-go").onclick = go;
  $("#match-id").onkeydown = (e) => { if (e.key === "Enter") go(); };
  $("#match-prev").onclick = () => { state.matchOffset = Math.max(0, state.matchOffset - 30); route(); };
  $("#match-next").onclick = () => { state.matchOffset += 30; route(); };
  document.querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", () => {
    location.hash = `#/match?id=${encodeURIComponent(tr.dataset.id)}`;
  }));
}

function idx2color(n) { return n === 2 ? "#48A9FF" : "#20D6A1"; }

async function pageWinProb(ctx) {
  ctx.render(`${requireData() || ""}${scenarioControls()}<div id="results"><p class="spinner">Loading…</p></div>`);
  const fill = async () => {
  const s = state.scenario;
  const grid = await api("winprob-grid", { wickets_left: s.wickets, target: s.target }).catch((e) => ({ error: e.message }));
  const strat = await api("strategy", stateParams(s)).catch((e) => ({ error: e.message }));
  const gridHtml = grid.error ? `<p class="warning-text">${esc(grid.error)}</p>` : heatGrid(
    grid.runs_needed.map((r) => `${r} needed`), grid.balls_left.map((b) => `${b} balls`),
    (i, j) => { const p = grid.probability[i][j]; return { bg: probColor(p), text: probPct(p, 0), title: `${grid.runs_needed[i]} needed, ${grid.balls_left[j]} balls left` }; });
  const big = strat.error ? `<p class="warning-text">${esc(strat.error)}</p>` : `
    <div class="prob ${strat.model_win_probability >= 0.5 ? "good" : "bad"}">${probPct(strat.model_win_probability)}</div>
    <p class="note">Chasing side's win probability (model estimate)</p>
    <p>Required ${fmt(s.runs, 0)} from ${fmt(s.balls, 0)} balls with ${s.wickets} wickets left. Current rate ${fmt(strat.crr, 2)}, required ${fmt(strat.rrr, 2)}.
    Pressure: <span class="badge ${strat.pressure_level === "Low" ? "good" : strat.pressure_level === "Medium" ? "warn" : "bad"}">${esc(strat.pressure_level)}</span></p>
    <p class="note">Historical analogue: ${strat.historical?.sample_size ? `${probPct(strat.historical.win_rate)} of ${fmt(strat.historical.sample_size, 0)} comparable states (${fmt(strat.historical.matches, 0)} matches) were won.` : esc(strat.historical?.note || "no comparable states")}</p>`;
  if (ctx.token === routeToken) $("#results").innerHTML = `
    <div class="grid">
      <div class="card span-4"><h2>Current state</h2>${big}</div>
      <div class="card span-8"><h2>Win probability by state</h2>
        <p class="note" style="margin-top:-6px">Target ${s.target} · ${s.wickets} wickets left. Rows: runs still needed. Columns: balls left.</p>
        ${gridHtml}
        <p class="note" style="margin-top:8px">Colours: red = low, amber = even, teal = high. Model estimates from historical chase states; see Methodology for validation.</p>
      </div>
    </div>`;
  };
  bindScenario(ctx, fill);
  await fill();
}

async function pageStrategy(ctx) {
  ctx.render(`${requireData() || ""}${scenarioControls()}<div id="results"><p class="spinner">Loading…</p></div>`);
  const fill = async () => {
  const s = state.scenario;
  const st = await api("strategy", stateParams(s));
  const sc = st.scenarios;
  const order = ["conservative", "balanced", "aggressive"];
  const scenRows = order.map((k) => ({ ...sc[k], key: k }));
  const cal = st.simulator_calibration;
  if (ctx.token === routeToken) $("#results").innerHTML = `
    <div class="kpi-row">
      ${kpi("Current run rate", fmt(st.crr, 2))}
      ${kpi("Required run rate", fmt(st.rrr, 2))}
      ${kpi("Gap (RRR − CRR)", fmt(st.rrr_minus_crr, 2))}
      ${kpi("Pressure", `<span class="${st.pressure_level === "Low" ? "good" : st.pressure_level === "Very high" ? "bad" : ""}">${esc(st.pressure_level)}</span>`, "heuristic band")}
      ${kpi("Wickets in hand", st.state.wickets_left, `${st.state.balls_left} balls left`)}
      ${kpi("Model win probability", probPct(st.model_win_probability), "trained model")}
    </div>
    <div class="grid">
      <div class="card span-6"><h2>Recommended approach</h2>
        <p>${esc(st.recommended_approach)}</p>
        <p class="note">Best scenario in simulation: <strong>${esc(sc[st.best_scenario].label)}</strong>.</p>
        <p class="note">${esc(st.assumption_note)}</p>
        ${cal ? `<p class="warning-text" style="margin-top:8px">Calibration check: across ${cal.states_compared} historical states, the simulator's absolute probabilities differ from realised win rates by ${(cal.simulator_mean_abs_error * 100).toFixed(1)} points on average (bias ${(cal.simulator_bias * 100).toFixed(1)} points). Use the differences between scenarios, not the absolute values.</p>` : ""}
      </div>
      <div class="card span-6"><h2>Historical analogue</h2>
        ${st.historical?.sample_size ? `<dl class="kv">
          <dt>Comparable states</dt><dd>${fmt(st.historical.sample_size, 0)}</dd>
          <dt>Matches</dt><dd>${fmt(st.historical.matches, 0)}</dd>
          <dt>Observed win rate</dt><dd>${probPct(st.historical.win_rate)}</dd>
          <dt>Criteria</dt><dd style="font-size:12px">${esc(st.historical.criteria)}</dd></dl>
          <p class="note" style="margin-top:8px">Observed frequency in the uploaded data. Sample size matters; check it before trusting a rate.</p>`
          : `<p class="note">${esc(st.historical?.note || "No historical comparison available.")}</p>`}
      </div>
      <div class="card span-12"><h2>Scenario comparison (simulated, assumption-based)</h2>
        <div class="table-wrap"><table><thead><tr><th>Scenario</th><th>Win chance</th><th>Δ vs balanced</th><th>Avg runs scored</th><th>Avg wickets lost</th><th>Assumptions</th></tr></thead><tbody>
        ${scenRows.map((r) => `<tr><td style="text-align:left"><strong>${esc(r.label)}</strong><div class="note">${esc(r.description)}</div></td>
          <td class="num">${probPct(r.win_probability)}</td>
          <td class="num ${r.delta_vs_balanced_pp > 0 ? "good" : r.delta_vs_balanced_pp < 0 ? "bad" : ""}">${r.delta_vs_balanced_pp > 0 ? "+" : ""}${fmt(r.delta_vs_balanced_pp, 1)} pts</td>
          <td class="num">${fmt(r.avg_runs_scored, 1)}</td><td class="num">${fmt(r.avg_wickets_lost, 2)}</td>
          <td class="note" style="white-space:normal">runs ×${r.assumptions.run_multiplier}, wickets ×${r.assumptions.wicket_multiplier}</td></tr>`).join("")}
        </tbody></table></div>
        <div style="margin-top:14px">${barRows(scenRows, { label: "label", value: "win_probability", format: (v) => probPct(v), max: 1 })}</div>
        <p class="note">Each ball is drawn from historical second-innings outcomes for its phase and wickets-fallen group. Outcomes are first scaled to this chase's current scoring pace (balanced = keep scoring at the current rate), then by the multipliers above. The multipliers are assumptions chosen for comparison, not estimates from data.</p>
      </div>
    </div>`;
  };
  bindScenario(ctx, fill);
  await fill();
}

async function pagePlayers(ctx, params) {
  const name = params.name || state.playerName || "";
  state.playerName = name;
  const meta = state.meta;
  let profile = null;
  if (name) profile = await api("player", { ...filterParams(), name });
  const b = profile?.batting, w = profile?.bowling, pi = profile?.pressure;
  const ctxRows = pi ? Object.entries(pi.contexts).map(([k, v]) => ({ context: k.replace(/_/g, " "), balls: v.balls, sr: v.strike_rate, used: v.used })) : [];
  ctx.render(`
    ${requireData() || ""}
    <div class="controls">
      <label class="field">Player<input type="text" id="player-name" list="player-list" value="${esc(name)}" placeholder="Start typing a name"></label>
      <datalist id="player-list">${(meta?.players || []).slice(0, 4000).map((p) => `<option value="${esc(p)}">`).join("")}</datalist>
    </div>
    ${!name ? `<div class="card"><p class="note">Choose a player to see batting, bowling and pressure metrics.</p></div>` :
      !profile.found ? `<div class="card"><p class="warning-text">No data for “${esc(name)}” in this selection.</p></div>` : `
    <div class="kpi-row">
      ${kpi("Runs", fmt(b.runs, 0), `${fmt(b.balls_faced, 0)} balls · ${b.dismissals} outs`)}
      ${kpi("Strike rate", fmt(b.strike_rate, 1))}
      ${kpi("Average", fmt(b.average, 1))}
      ${kpi("Boundary %", pct(b.boundary_pct))}
      ${kpi("Dot % (batting)", pct(b.dot_pct))}
      ${kpi("Wickets", fmt(w.wickets, 0), `${fmt(w.overs, 1)} overs`)}
      ${kpi("Economy", fmt(w.economy, 2))}
      ${kpi("Bowling average", fmt(w.bowling_average, 1))}
    </div>
    <div class="grid">
      <div class="card span-6"><h2>Batting</h2><dl class="kv">
        <dt>Fours / sixes</dt><dd>${b.fours} / ${b.sixes}</dd>
        <dt>Balls faced</dt><dd>${fmt(b.balls_faced, 0)}</dd>
        <dt>Dismissals</dt><dd>${b.dismissals}</dd>
        <dt>Dot % (batting)</dt><dd>${pct(b.dot_pct)}</dd></dl></div>
      <div class="card span-6"><h2>Bowling</h2><dl class="kv">
        <dt>Legal balls / overs</dt><dd>${fmt(w.legal_balls, 0)} / ${fmt(w.overs, 1)}</dd>
        <dt>Runs conceded</dt><dd>${fmt(w.runs_conceded, 0)}</dd>
        <dt>Bowling strike rate</dt><dd>${fmt(w.bowling_strike_rate, 1)}</dd>
        <dt>Dot % (bowling)</dt><dd>${pct(w.dot_pct)}</dd></dl></div>
      <div class="card span-6"><h2>Pressure Performance Index <span class="badge">custom index</span></h2>
        <p class="note">Not an official cricket statistic. 100 = same strike rate under pressure as overall; above 100 = scores faster under pressure. Formula: mean of (pressure SR ÷ career SR) over contexts with at least ${30} balls.</p>
        <p style="font-size:34px;font-weight:700;margin:8px 0">${pi.index === null ? "—" : fmt(pi.index, 1)}</p>
        ${pi.note ? `<p class="warning-text">${esc(pi.note)}</p>` : ""}
        ${table([{ key: "context", label: "Context", fmt: (v) => v }, { key: "balls", label: "Balls", fmt: (v) => fmt(v, 0) },
          { key: "sr", label: "SR", fmt: (v) => fmt(v, 1) }, { html: (r) => r.used ? `<span class="badge good">used</span>` : `<span class="badge">too few balls</span>`, label: "Used" }], ctxRows)}
      </div>
      <div class="card span-6"><h2>Strike rate by season</h2>${lineChart([{ name: "Strike rate", color: "#20D6A1", points: profile.by_season.map((s, i) => ({ x: i, y: s.strike_rate })) }],
        { yFormat: (v) => fmt(v, 0), xLabels: profile.by_season.map((s) => s.season) })}
        <p class="note">Seasons with batting in the selection (ball by ball, wides excluded).</p></div>
    </div>`}`);
  if (ctx.token !== routeToken) return;
  const inp = $("#player-name");
  inp.onchange = () => { location.hash = `#/players?name=${encodeURIComponent(inp.value.trim())}`; };
  inp.onkeydown = (e) => { if (e.key === "Enter") inp.onchange(); };
}

async function pageCompare(ctx, params) {
  const a = params.a || state.compareA || "";
  const b = params.b || state.compareB || "";
  state.compareA = a; state.compareB = b;
  let cmp = null;
  if (a && b) cmp = await api("compare", { ...filterParams(), a, b });
  const metricRows = cmp ? [
    ["Runs", cmp.a.batting?.runs, cmp.b.batting?.runs, 0],
    ["Strike rate", cmp.a.batting?.strike_rate, cmp.b.batting?.strike_rate, 1],
    ["Batting average", cmp.a.batting?.average, cmp.b.batting?.average, 1],
    ["Boundary %", cmp.a.batting?.boundary_pct, cmp.b.batting?.boundary_pct, 1],
    ["Wickets", cmp.a.bowling?.wickets, cmp.b.bowling?.wickets, 0],
    ["Economy (lower better)", cmp.a.bowling?.economy, cmp.b.bowling?.economy, 2],
    ["Bowling average", cmp.a.bowling?.bowling_average, cmp.b.bowling?.bowling_average, 1],
    ["Pressure index", cmp.a.pressure?.index, cmp.b.pressure?.index, 1],
  ] : [];
  ctx.render(`
    ${requireData() || ""}
    <div class="controls">
      <label class="field">Player A<input type="text" id="cmp-a" list="player-list" value="${esc(a)}"></label>
      <label class="field">Player B<input type="text" id="cmp-b" list="player-list" value="${esc(b)}"></label>
      <datalist id="player-list">${(state.meta?.players || []).slice(0, 4000).map((p) => `<option value="${esc(p)}">`).join("")}</datalist>
      <button type="button" id="cmp-go">Compare</button>
    </div>
    ${!cmp ? `<div class="card"><p class="note">Enter two players to compare them side by side.</p></div>` : `
    <div class="grid">
      <div class="card span-6"><h2>Side by side</h2><div class="table-wrap"><table><thead><tr><th>Metric</th><th>${esc(a)}</th><th>${esc(b)}</th></tr></thead><tbody>
        ${metricRows.map(([l, va, vb, d]) => `<tr><td>${esc(l)}</td><td class="num">${fmt(va, d)}</td><td class="num">${fmt(vb, d)}</td></tr>`).join("")}
      </tbody></table></div></div>
      <div class="card span-6"><h2>Profile radar (scaled to the better player)</h2>${radarChart(cmp.radar.map((r) => ({ axis: r.axis, a: r.a, b: r.b })), a, b)}
        <p class="note">Each axis is scaled so the better player on it reaches the outer edge. Economy is inverted so that lower is better.</p></div>
      <div class="card span-12"><p class="note">Samples: ${esc(a)} ${cmp.a.batting?.balls_faced ?? 0} balls faced, ${cmp.a.bowling?.legal_balls ?? 0} balls bowled; ${esc(b)} ${cmp.b.batting?.balls_faced ?? 0} balls faced, ${cmp.b.bowling?.legal_balls ?? 0} balls bowled. Compare only where both samples are large.</p></div>
    </div>`}`);
  if (ctx.token !== routeToken) return;
  const go = () => { location.hash = `#/compare?a=${encodeURIComponent($("#cmp-a").value.trim())}&b=${encodeURIComponent($("#cmp-b").value.trim())}`; };
  $("#cmp-go").onclick = go;
  [$("#cmp-a"), $("#cmp-b")].forEach((el) => el.onkeydown = (e) => { if (e.key === "Enter") go(); });
}

async function pageMatchup(ctx, params) {
  const batter = params.batter || state.batter || "";
  const bowler = params.bowler || "";
  state.batter = batter;
  let options = null, result = null;
  if (batter) {
    options = await api("matchup", { ...filterParams(), batter });
    if (bowler) result = await api("matchup", { ...filterParams(), batter, bowler });
  }
  ctx.render(`
    ${requireData() || ""}
    <div class="controls">
      <label class="field">Batter<input type="text" id="mu-batter" list="player-list" value="${esc(batter)}"></label>
      <label class="field">Bowler<select id="mu-bowler" ${options ? "" : "disabled"}>
        <option value="">${options ? "Choose a bowler…" : "Choose a batter first"}</option>
        ${(options?.bowlers || []).map((o) => `<option value="${esc(o.bowler)}" ${o.bowler === bowler ? "selected" : ""}>${esc(o.bowler)} (${o.balls} balls)</option>`).join("")}
      </select></label>
      <datalist id="player-list">${(state.meta?.players || []).slice(0, 4000).map((p) => `<option value="${esc(p)}">`).join("")}</datalist>
    </div>
    ${!batter ? `<div class="card"><p class="note">Choose a batter, then a bowler, to see their head-to-head record in the selected data.</p></div>` :
      !bowler ? `<div class="card"><p class="note">${options?.bowlers?.length ? `${options.bowlers.length} bowlers have faced ${esc(batter)} in this selection.` : `No balls found for ${esc(batter)} in this selection.`}</p></div>` : `
    <div class="grid">
      <div class="card span-8"><h2>${esc(batter)} vs ${esc(bowler)}</h2>
        <div class="kpi-row">
          ${kpi("Balls faced", fmt(result.balls, 0))}
          ${kpi("Runs", fmt(result.runs, 0))}
          ${kpi("Strike rate", fmt(result.strike_rate, 1))}
          ${kpi("Dismissals", fmt(result.dismissals, 0), "by this bowler")}
          ${kpi("Dot %", pct(result.dot_pct))}
          ${kpi("Boundary %", pct(result.boundary_pct))}
        </div>
        <p>${sampleBadge(result.balls)} <span class="note">${esc(result.warning)}</span></p>
      </div>
      <div class="card span-4"><h2>Sample size</h2><p class="note">Matches with this matchup: ${fmt(result.matches, 0)}.</p>
        <p class="note">Small samples swing widely. Use these numbers as context, not as predictions.</p></div>
    </div>`}`);
  if (ctx.token !== routeToken) return;
  const bat = $("#mu-batter");
  bat.onchange = () => { location.hash = `#/matchup?batter=${encodeURIComponent(bat.value.trim())}`; };
  bat.onkeydown = (e) => { if (e.key === "Enter") bat.onchange(); };
  $("#mu-bowler").onchange = (e) => { location.hash = `#/matchup?batter=${encodeURIComponent(batter)}&bowler=${encodeURIComponent(e.target.value)}`; };
}

async function pageTeams(ctx, params) {
  const team = params.team || state.teamPick || "";
  state.teamPick = team;
  const d = await api("teams", { ...filterParams(), team });
  const teamNames = d.table.map((t) => t.team);
  ctx.render(`
    ${requireData() || ""}
    <p class="note" style="margin-bottom:12px">The team filter above does not apply on this page: it compares all teams in the selected format, season and tournament.</p>
    <div class="controls">
      <label class="field">Focus team<select id="team-pick"><option value="">Choose a team…</option>
        ${teamNames.map((t) => `<option value="${esc(t)}" ${t === team ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></label>
    </div>
    <div class="grid">
      <div class="card flat span-12"><h2>Team table</h2>${table([
        { key: "team", label: "Team" }, { key: "played", label: "Played", fmt: (v) => fmt(v, 0) },
        { key: "wins", label: "Wins", fmt: (v) => fmt(v, 0) }, { key: "win_pct", label: "Win %", fmt: (v) => pct(v) },
        { key: "chases", label: "Chases", fmt: (v) => fmt(v, 0) }, { key: "chase_win_pct", label: "Chase win %", fmt: (v) => pct(v) },
        { key: "bat_first", label: "Bat first", fmt: (v) => fmt(v, 0) }, { key: "defend_win_pct", label: "Win % batting first", fmt: (v) => pct(v) },
        { key: "avg_first_innings", label: "Avg 1st inns", fmt: (v) => fmt(v, 1) }, { key: "toss_win_match_win_pct", label: "Toss→win %", fmt: (v) => pct(v) },
      ], d.table.slice(0, 60), { rowClass: (r) => (r.team === team ? "selected" : "") })}</div>
      ${d.selected ? `<div class="card span-4"><h2>${esc(team)}</h2><dl class="kv">
        <dt>Win rate</dt><dd>${pct(d.selected.win_pct)}</dd>
        <dt>Chase win rate</dt><dd>${pct(d.selected.chase_win_pct)}</dd>
        <dt>Win rate batting first</dt><dd>${pct(d.selected.defend_win_pct)}</dd>
        <dt>Average first innings</dt><dd>${fmt(d.selected.avg_first_innings, 1)}</dd></dl></div>
        <div class="card span-8"><h2>Win % by season</h2>${lineChart([{ name: team, color: "#20D6A1", points: d.season_trend.map((s, i) => ({ x: i, y: s.win_pct })) }], { yMin: 0, yMax: 100, yFormat: (v) => `${Math.round(v)}%`, xLabels: d.season_trend.map((s) => s.season) })}
          <p class="note">Win % per season, decided matches only.</p></div>` : ""}
      <div class="card span-12"><h2>Head-to-head win % (top ${d.head_to_head.teams.length} teams by matches)</h2>
        ${heatGrid(d.head_to_head.teams, d.head_to_head.teams, (i, j) => {
          const c = d.head_to_head.grid[i][j];
          if (!c) return { empty: true, text: "—" };
          if (!c.played) return { empty: true, text: "0" };
          const p = c.wins / c.played;
          return { bg: probColor(p), text: `${Math.round(p * 100)}%`, title: `${c.wins}-${c.played - c.wins} in ${c.played} matches` };
        })}
        <p class="note">Row team's win % against the column team. Only decided matches are counted.</p></div>
    </div>`);
  if (ctx.token !== routeToken) return;
  $("#team-pick").onchange = (e) => { location.hash = `#/teams?team=${encodeURIComponent(e.target.value)}`; };
}

async function pageExplorer(ctx, params) {
  const table_ = params.table || state.explorerTable;
  state.explorerTable = table_;
  if (params.match_id) state.explorerMatch = params.match_id;
  const offset = Number(params.offset ?? state.explorerOffset) || 0;
  state.explorerOffset = offset;
  const limit = 50;
  let body = "";
  if (table_ === "matches") {
    const d = await api("explorer", { table: "matches", limit, offset, ...filterParams() });
    body = table([
      { key: "match_id", label: "Match ID" }, { key: "date", label: "Date" }, { key: "match_type", label: "Format" },
      { key: "event", label: "Event" }, { html: (r) => `${esc(r.team1)} v ${esc(r.team2)}`, label: "Teams" },
      { key: "inn1_runs", label: "1st inns", fmt: (v) => fmt(v, 0) }, { key: "inn2_runs", label: "2nd inns", fmt: (v) => fmt(v, 0) },
      { key: "winner", label: "Winner" },
      { html: (r) => `<a href="#/match?id=${encodeURIComponent(r.match_id)}">Open</a>`, label: "" },
    ], d.rows);
    body += pager(offset, limit, d.total);
  } else {
    const mid = state.explorerMatch || "";
    let d = { rows: [], total: 0 };
    if (mid) d = await api("explorer", { table: "deliveries", match_id: mid, limit, offset });
    body = `<div class="controls"><label class="field">Match ID<input type="search" id="ball-match" value="${esc(mid)}"></label></div>` +
      (mid ? table([
        { key: "innings", label: "Inn" }, { key: "over", label: "Over" }, { key: "batter", label: "Batter" }, { key: "bowler", label: "Bowler" },
        { key: "runs_batter", label: "Bat" }, { key: "runs_extras", label: "Extras" }, { key: "runs_total", label: "Total" },
        { key: "extra_type", label: "Extra", fmt: (v) => v || "" }, { key: "wicket_kind", label: "Wicket", fmt: (v) => v || "" },
        { key: "legal", label: "Legal", fmt: (v) => (v ? "yes" : "no") }], d.rows) + pager(offset, limit, d.total)
        : `<p class="note">Enter a match ID (copy one from the Matches tab) to see its deliveries.</p>`);
  }
  ctx.render(`
    ${requireData() || ""}
    <div class="tabs">
      <button type="button" data-t="matches" class="${table_ === "matches" ? "on" : ""}">Matches</button>
      <button type="button" data-t="deliveries" class="${table_ === "deliveries" ? "on" : ""}">Deliveries (ball by ball)</button>
      <a class="btn" style="margin-left:auto;background:var(--bg3);color:var(--text);border:1px solid var(--line)" href="/api/export/matches.csv">Download matches CSV</a>
    </div>
    <div class="card flat">${body}</div>`);
  if (ctx.token !== routeToken) return;
  document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => { state.explorerOffset = 0; location.hash = `#/explorer?table=${b.dataset.t}`; });
  document.querySelectorAll("[data-page-go]").forEach((b) => b.onclick = () => { location.hash = `#/explorer?table=${table_}&offset=${b.dataset.pageGo}`; });
  const bm = $("#ball-match");
  if (bm) bm.onkeydown = (e) => { if (e.key === "Enter") location.hash = `#/explorer?table=deliveries&match_id=${encodeURIComponent(bm.value.trim())}`; };
}

function pager(offset, limit, total) {
  const prev = Math.max(0, offset - limit), next = offset + limit;
  return `<div class="controls" style="padding:12px 16px">
    <span class="note">${fmt(Math.min(total, offset + 1), 0)}–${fmt(Math.min(total, offset + limit), 0)} of ${fmt(total, 0)}</span>
    <button type="button" class="ghost" data-page-go="${prev}" ${offset ? "" : "disabled"}>Previous</button>
    <button type="button" class="ghost" data-page-go="${next}" ${next < total ? "" : "disabled"}>Next</button></div>`;
}

async function pageMethodology(ctx) {
  const m = await api("model").catch(() => null);
  const status = state.status;
  const cal = await api("strategy", stateParams(state.scenario)).then((r) => r.simulator_calibration).catch(() => null);
  if (!m) {
    ctx.render(`<div class="card method"><h2>Model not trained</h2><p>Build the data and model on the <a href="#/data">Data &amp; Upload</a> page.</p></div>`);
    return;
  }
  const models = Object.entries(m.models);
  const sel = m.selected_model;
  const metricRow = (name, t, extra = "") => `<tr class="${name === sel ? "selected" : ""}"><td>${esc(name.replace(/_/g, " "))}${name === sel ? ' <span class="badge good">selected</span>' : ""}${extra}</td>
    <td class="num">${fmt(t.accuracy * 100, 1)}%</td><td class="num">${fmt(t.precision * 100, 1)}%</td><td class="num">${fmt(t.recall * 100, 1)}%</td>
    <td class="num">${fmt(t.f1, 3)}</td><td class="num">${fmt(t.roc_auc, 3)}</td><td class="num">${fmt(t.brier, 3)}</td><td class="num">${fmt(t.log_loss, 3)}</td></tr>`;
  const calib = m.calibration_test.map((c) => ({ bin: `${Math.round(c.bin_low * 100)}–${Math.round(c.bin_high * 100)}%`, observed: c.observed_rate * 100, predicted: c.mean_predicted * 100, count: c.count }));
  const q = status?.quality;
  ctx.render(`
    <div class="grid method">
      <div class="card span-12"><h2>What the model does</h2>
        <p>ChaseIQ estimates the chasing side's probability of winning a T20 run chase from the state of the innings: runs still needed, balls left, wickets left, current and required run rate, and the target. It is trained on ${fmt(m.data.matches_total, 0)} decided 20-over chases from the uploaded Cricsheet data. The model uses score-state features only; it does not know team strength, venue, pitch, weather or which batters are in form.</p>
        <p class="note">Trained ${esc(m.trained_at)} · selected model: <strong>${esc(sel.replace(/_/g, " "))}</strong> (${esc(m.selection_rule)}).</p>
      </div>
      <div class="card flat span-12"><h2>Test results (${fmt(m.data.test_states, 0)} ball-states from ${fmt(m.data.test_matches, 0)} matches, ${esc(m.data.test_period.join(" → "))})</h2>
        <p class="note" style="padding:0 16px">Test matches were held out and scored once. They are the most recent matches by date, so the result reflects performance on later seasons. Each row is one legal ball-state; metrics are computed across all test ball-states pooled together, not averaged per match.</p>
        <div class="table-wrap"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Brier ↓</th><th>Log loss ↓</th></tr></thead><tbody>
        ${models.filter(([k]) => !k.startsWith("baseline")).map(([k, v]) => metricRow(k, v.test)).join("")}
        ${models.filter(([k]) => k.startsWith("baseline")).map(([k, v]) => metricRow(k, v.test, ' <span class="badge">baseline</span>')).join("")}
        </tbody></table></div>
        <p class="note" style="padding:12px 16px 0">Baselines: "majority" always predicts the historical chase win rate; "RRR only" uses just the required run rate. Accuracy is the share of states classified correctly at a 50% threshold; the Brier score and log loss measure probability quality (lower is better).</p>
      </div>
      <div class="card span-6"><h2>Calibration on the test set</h2>
        <p class="note">Predicted probability vs observed chase win rate, in ten bins. Points near the diagonal mean the probabilities can be read as frequencies.</p>
        ${table([{ key: "bin", label: "Predicted bin" }, { key: "predicted", label: "Mean predicted", fmt: (v) => pct(v, 1) },
          { key: "observed", label: "Observed", fmt: (v) => pct(v, 1) }, { key: "count", label: "States", fmt: (v) => fmt(v, 0) }], calib)}
      </div>
      <div class="card span-6"><h2>Feature importance (drop in ROC-AUC when shuffled)</h2>
        ${barRows(m.permutation_importance_test_auc, { label: "feature", value: "importance_mean", format: (v) => fmt(v, 4), neg: true })}
        <p class="note">Measured on a sample of the test set. Higher = the model relies on that feature more.</p>
      </div>
      <div class="card span-12"><h2>Simulator check (strategy scenarios)</h2>
        ${cal ? `<p>Across ${cal.states_compared} historical states, the simulator's balanced-scenario win chance differs from the realised win rate by ${(cal.simulator_mean_abs_error * 100).toFixed(1)} points on average (bias ${(cal.simulator_bias * 100).toFixed(1)}). The model differs by ${cal.model_mean_abs_error === null ? "—" : (cal.model_mean_abs_error * 100).toFixed(1) + " points"} on the same states, but those states are in its training data, so that number is not out-of-sample.</p>
          <p class="warning-text">Because of this gap, ChaseIQ reports the simulator's scenarios as differences from one another, not as absolute win probabilities.</p>`
          : `<p class="note">Simulator check not run yet. It runs automatically during a build.</p>`}
      </div>
      <div class="card span-12"><h2>How the strategy scenarios work</h2>
        <ul>
          <li>Each remaining ball is drawn from historical second-innings outcomes for the same phase (powerplay 1–6, middle 7–15, death 16–20) and wickets-fallen group, scaled to the chase's current scoring rate (pace factor, capped between 0.5× and 2×).</li>
          <li>Balanced keeps the current scoring pace with historical wicket rates. Conservative (×0.85 runs, ×0.80 wicket chance) and aggressive (×1.20 runs, ×1.30 wicket chance) modify that. The multipliers are assumptions, set in <code>config.py</code>, not estimates from data.</li>
          <li>Historical analogues (on the Chase Strategy page) show observed win rates for comparable states, with the sample size.</li>
        </ul>
      </div>
      <div class="card span-12"><h2>Definitions and formulas</h2>
        <ul>
          <li><strong>Current run rate (CRR)</strong> = runs scored × 6 ÷ legal balls bowled.</li>
          <li><strong>Required run rate (RRR)</strong> = runs needed × 6 ÷ legal balls left.</li>
          <li><strong>Strike rate</strong> = runs × 100 ÷ balls faced (wides excluded from balls faced).</li>
          <li><strong>Batting average</strong> = runs ÷ dismissals (retired hurt/not out not counted as dismissals).</li>
          <li><strong>Economy</strong> = runs conceded × 6 ÷ legal balls (byes and leg-byes excluded from runs conceded).</li>
          <li><strong>Bowling strike rate</strong> = legal balls ÷ wickets credited to the bowler (bowled, caught, caught &amp; bowled, lbw, stumped, hit wicket).</li>
          <li><strong>Pressure Performance Index</strong> (custom, not official): 100 × mean over pressure contexts of (strike rate in context ÷ career strike rate). Contexts: death overs; chases with required rate ≥ 10; 4+ wickets down; innings in chases the side won. A context counts only with at least 30 balls.</li>
          <li><strong>Sample-size warnings</strong>: matchups under 12 balls are "very small"; under 30 are "small".</li>
        </ul>
      </div>
      <div class="card span-12"><h2>Data, cleaning and limitations</h2>
        ${q ? `<dl class="kv"><dt>Source</dt><dd>Cricsheet T20 JSON ZIP (${esc((q.source_files || []).join(", "))})</dd>
          <dt>Matches</dt><dd>${fmt(q.matches_total, 0)} (${fmt(q.matches_without_winner, 0)} without a winner: ties, no-results)</dd>
          <dt>Deliveries</dt><dd>${fmt(q.deliveries_total, 0)} balls (${fmt(q.legal_deliveries, 0)} legal)</dd>
          <dt>Super-over innings excluded</dt><dd>${fmt(q.counters?.super_over_innings_excluded || 0, 0)}</dd>
          <dt>D/L or awarded matches</dt><dd>${fmt(q.counters?.matches_with_method_adjustment_dl || 0, 0)} kept for descriptive analytics; ${fmt(q.chase_training?.excluded_adjusted_result_matches || 0, 0)} decided 20-over chases excluded from win-probability training (adjusted targets)</dd></dl>
          <ul style="margin-top:12px">${(q.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : `<p class="note">Build the data to see the cleaning report.</p>`}
        <ul>${(m.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
        <p class="note">Attribution: ball-by-ball match data from <a href="https://cricsheet.org/" target="_blank" rel="noopener">Cricsheet</a>. Cricsheet says most of its data is under the <a href="https://opendatacommons.org/licenses/by/1.0/" target="_blank" rel="noopener">Open Data Commons Attribution License (ODC-By) 1.0</a> and its XML match data alone under CC BY 4.0; its page does not name the JSON archive individually. ChaseIQ processed and transformed this data into the tables, model and reports shown here. ChaseIQ is an independent analysis and is not affiliated with Cricsheet, the ICC or any league. See DATA_ATTRIBUTION.md.</p>
      </div>
    </div>`);
}

async function pageData(ctx) {
  const s = await api("status");
  state.status = s;
  const q = s.quality;
  ctx.render(`
    <div class="grid">
      <div class="card span-4"><h2>Current data</h2><dl class="kv">
        <dt>Matches</dt><dd>${fmt(s.matches, 0)}</dd>
        <dt>Balls</dt><dd>${fmt(s.deliveries, 0)}</dd>
        <dt>Chase states</dt><dd>${fmt(s.chase_states, 0)}</dd>
        <dt>Model</dt><dd>${s.model_available ? esc((s.model_name || "trained").replace(/_/g, " ")) : "not trained"}</dd>
        <dt>ZIPs in data/raw</dt><dd>${esc(s.raw_zips.join(", ") || "none")}</dd></dl></div>
      <div class="card span-8"><h2>Upload a Cricsheet T20 ZIP</h2>
        <p class="note">Download <code>t20s_json.zip</code> from <a href="https://cricsheet.org/downloads/" target="_blank" rel="noopener">cricsheet.org/downloads</a> and choose it here. The app saves it to <code>data/raw/</code>, then ingests, builds chase states and retrains the model. This takes a couple of minutes.</p>
        <div class="controls">
          <input type="file" id="zip-file" accept=".zip,application/zip">
          <button type="button" id="zip-upload">Upload and build</button>
          <button type="button" class="ghost" id="rebuild">Rebuild from data/raw</button>
        </div>
        <p id="build-msg" class="note"></p>
      </div>
      <div class="card span-12"><h2>Build log</h2><pre class="log" id="build-log">${esc((s.build_log || []).join("\n") || "No build run in this session yet.")}</pre></div>
      ${q ? `<div class="card span-12"><h2>Last ingestion report</h2><pre class="log">${esc(JSON.stringify(q.counters, null, 2))}</pre></div>` : ""}
    </div>`);
  if (ctx.token !== routeToken) return;
  const msg = (t, cls = "") => { const el = $("#build-msg"); el.className = cls || "note"; el.textContent = t; };
  const runBuild = async (promise) => {
    $("#zip-upload").disabled = true; $("#rebuild").disabled = true;
    msg("Building… ingesting ZIP, building chase states and training models. Please keep this tab open.");
    try {
      const r = await promise;
      const mt = r.model?.test_metrics;
      msg(`Done. Model "${r.model?.selected_model?.replace(/_/g, " ")}" — test accuracy ${mt ? (mt.accuracy * 100).toFixed(1) + "%" : "n/a"}, ROC-AUC ${mt ? mt.roc_auc.toFixed(3) : "n/a"} on ${fmt(r.model?.test_matches, 0)} held-out matches.`, "note good");
      await loadMeta();
      route();
    } catch (e) {
      msg(`Build failed: ${e.message}`, "warning-text");
    } finally {
      $("#zip-upload").disabled = false; $("#rebuild").disabled = false;
    }
  };
  $("#zip-upload").onclick = () => {
    const file = $("#zip-file").files[0];
    if (!file) { msg("Choose a .zip file first.", "warning-text"); return; }
    runBuild(fetch(`/api/upload?name=${encodeURIComponent(file.name)}&build=1`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    }).then(async (res) => { const data = await res.json(); if (!res.ok) throw new Error(data.error); return data; }));
  };
  $("#rebuild").onclick = () => runBuild(fetch("/api/build", { method: "POST" }).then(async (res) => { const data = await res.json(); if (!res.ok) throw new Error(data.error); return data; }));
}

// ---------------------------------------------------------------- shell
const PAGES = {
  overview: { title: "Overview", sub: "Historical T20 summary for the selected filters", fn: pageOverview, filters: true },
  match: { title: "Match Intelligence", sub: "Scorecard summary, manhattan and worm charts, and win probability through the chase", fn: pageMatch, filters: true },
  winprob: { title: "Win Probability", sub: "Model estimate of the chasing side's win chance for any state", fn: pageWinProb },
  strategy: { title: "Chase Strategy", sub: "Pressure, historical analogues and assumption-based scenarios", fn: pageStrategy },
  players: { title: "Player Analytics", sub: "Batting, bowling and pressure metrics for one player", fn: pagePlayers, filters: true },
  compare: { title: "Player Comparison", sub: "Side-by-side metrics and profile radar", fn: pageCompare, filters: true },
  matchup: { title: "Batter vs Bowler", sub: "Head-to-head record with sample-size warnings", fn: pageMatchup, filters: true },
  teams: { title: "Team Analytics", sub: "Win rates, chase vs defend, head-to-head and season trends", fn: pageTeams, filters: true },
  explorer: { title: "Data Explorer", sub: "Browse matches and ball-by-ball deliveries; download CSV", fn: pageExplorer, filters: true },
  methodology: { title: "Methodology & Model", sub: "How every number is produced, validated and limited", fn: pageMethodology },
  data: { title: "Data & Upload", sub: "Load Cricsheet data and rebuild the model", fn: pageData },
};

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  const page = PAGES[path] ? path : "overview";
  return { page, params: Object.fromEntries(new URLSearchParams(qs || "")) };
}

async function route() {
  const token = ++routeToken;
  const { page, params } = parseHash();
  const def = PAGES[page];
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  $("#page-title").textContent = def.title;
  $("#page-sub").textContent = def.sub;
  $("#filters").hidden = !def.filters;
  const view = $("#view");
  view.innerHTML = `<p class="spinner">Loading…</p>`;
  const ctx = {
    token,
    render(html) { if (token === routeToken) view.innerHTML = html; },
  };
  try {
    await def.fn(ctx, params);
  } catch (e) {
    if (token === routeToken) {
      view.innerHTML = `<div class="card"><h2>Something went wrong</h2><p class="warning-text">${esc(e.message)}</p>
        <p class="note">If this is a missing-data error, open <a href="#/data">Data &amp; Upload</a>.</p></div>`;
    }
  }
}

function setFilterOptions() {
  const m = state.meta;
  if (!m) return;
  const fill = (id, values, allLabel) => {
    const el = $("#" + id);
    el.innerHTML = `<option value="">${allLabel}</option>` + values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
  };
  fill("f-match_type", m.match_types, "All formats");
  fill("f-season", [...m.seasons].reverse(), "All seasons");
  fill("f-event", m.events, "All tournaments");
  fill("f-team", m.teams, "All teams");
  ["match_type", "season", "event", "team"].forEach((k) => { $("#f-" + k).value = state.filters[k]; });
}

async function loadMeta() {
  state.status = await api("status");
  const pill = $("#status-pill");
  const banner = $("#banner");
  if (state.status.data_loaded) {
    pill.className = "pill ok";
    pill.textContent = `${fmt(state.status.matches, 0)} matches · model ${state.status.model_available ? "ready" : "not trained"}`;
    state.meta = await api("filters");
    setFilterOptions();
    banner.hidden = true;
  } else {
    pill.className = "pill warn";
    pill.textContent = "No data loaded";
    banner.hidden = false;
    banner.className = "banner warn";
    banner.innerHTML = `No Cricsheet data is loaded yet. Go to <a href="#/data">Data &amp; Upload</a> to add the T20 ZIP.`;
  }
}

function bindFilters() {
  ["match_type", "season", "event", "team"].forEach((k) => {
    $("#f-" + k).addEventListener("change", (e) => { state.filters[k] = e.target.value; route(); });
  });
  $("#f-reset").addEventListener("click", () => {
    state.filters = { match_type: "", season: "", event: "", team: "" };
    setFilterOptions();
    route();
  });
}

window.addEventListener("hashchange", route);
document.addEventListener("DOMContentLoaded", async () => {
  bindFilters();
  try {
    await loadMeta();
  } catch (e) {
    $("#status-pill").textContent = "Server unreachable";
  }
  if (!location.hash) location.hash = "#/overview";
  route();
});
