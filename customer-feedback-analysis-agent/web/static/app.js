/* =========================================================================
   Customer Feedback Analysis Agent — front-end controller
   All heavy lifting happens server-side in the Python agent/tools pipeline.
   This file only renders what /api/analyze returns.
   ========================================================================= */
(function () {
  "use strict";

  // ── State ───────────────────────────────────────────────────────────────
  var STORE_KEY = "cfa_results_v1";
  var state = { results: null, single: null, pendingFile: null, groq: false, model: "" };

  var $ = function (id) { return document.getElementById(id); };
  var esc = function (v) {
    return String(v === null || v === undefined ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  };
  var num = function (n) { return (n === null || n === undefined || isNaN(n)) ? "0" : Number(n).toLocaleString(); };

  var PALETTE = {
    sentiment: { Positive: "#2D7A58", Neutral: "#7D656A", Negative: "#9C1D3A" },
    emotion: { Happy: "#2D7A58", Satisfied: "#459B73", Neutral: "#7D656A", Confused: "#5A7C99", Disappointed: "#BA6A24", Frustrated: "#B33951", Angry: "#8B1430" },
    priority: { HIGH: "#8B1430", MEDIUM: "#BA6A24", LOW: "#2D7A58" }
  };
  var WINE = "#722F37", WINE_LIGHT = "#A3485E", TEXT = "#33181E";

  var PLOT_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Plus Jakarta Sans, sans-serif", color: TEXT, size: 12 },
    margin: { l: 56, r: 22, t: 34, b: 52 },
    xaxis: { gridcolor: "rgba(114,47,55,0.08)", zeroline: false },
    yaxis: { gridcolor: "rgba(114,47,55,0.08)", zeroline: false }
  };
  var PLOT_CONFIG = { displayModeBar: false, responsive: true };

  function plot(el, data, layout) {
    if (!el || typeof Plotly === "undefined") return;
    var l = JSON.parse(JSON.stringify(PLOT_LAYOUT));
    Object.keys(layout || {}).forEach(function (k) { l[k] = layout[k]; });
    Plotly.newPlot(el, data, l, PLOT_CONFIG);
  }

  // ── Persistence (survives page reloads; serverless has no session) ──────
  function save() {
    try { sessionStorage.setItem(STORE_KEY, JSON.stringify(state.results)); } catch (e) { /* quota — fine */ }
  }
  function restore() {
    try {
      var raw = sessionStorage.getItem(STORE_KEY);
      if (raw) state.results = JSON.parse(raw);
    } catch (e) { state.results = null; }
  }

  // ── Navigation ──────────────────────────────────────────────────────────
  var PAGES = ["dashboard", "analyze", "upload", "issues", "insights", "recommendations", "activity", "reports", "explorer"];

  function go(page) {
    PAGES.forEach(function (p) {
      var sec = $("page-" + p);
      if (sec) sec.classList.toggle("hidden", p !== page);
    });
    Array.prototype.forEach.call(document.querySelectorAll("#nav button"), function (b) {
      b.classList.toggle("active", b.getAttribute("data-page") === page);
      b.setAttribute("aria-selected", b.getAttribute("data-page") === page ? "true" : "false");
    });
    closeMenu();
    window.scrollTo(0, 0);
    render(page);
  }

  function openMenu() { $("sidebar").classList.add("open"); $("backdrop").classList.remove("hidden"); $("menuBtn").setAttribute("aria-expanded", "true"); }
  function closeMenu() { $("sidebar").classList.remove("open"); $("backdrop").classList.add("hidden"); $("menuBtn").setAttribute("aria-expanded", "false"); }

  // ── Generic renderers ───────────────────────────────────────────────────
  function emptyState(msg, action) {
    return '<div class="card"><div class="empty"><div class="icon">📭</div>' +
      '<p>' + esc(msg) + '</p>' +
      (action ? '<button class="btn" data-goto="upload">📂 Upload a dataset</button>' : '') +
      '</div></div>';
  }

  function kpi(title, value, sub, color, grad) {
    return '<div class="kpi-card"><div class="bar" style="background:' + (grad || "linear-gradient(90deg,#722F37,#A3485E)") + '"></div>' +
      '<div class="kpi-title">' + esc(title) + '</div>' +
      '<div class="kpi-value"' + (color ? ' style="color:' + color + '"' : '') + '>' + esc(value) + '</div>' +
      '<div class="kpi-sub">' + esc(sub || "") + '</div></div>';
  }

  function badge(v) { return '<span class="badge ' + esc(v) + '">' + esc(v) + '</span>'; }

  function statusPill() {
    var done = !!state.results;
    var pill = $("statusPill"), dot = $("statusDot");
    pill.className = "pill " + (done ? "complete" : "ready");
    dot.className = "dot " + (done ? "complete" : "ready");
    $("statusTitle").textContent = done ? "PIPELINE ACTIVE" : "SYSTEM READY";
    $("statusTitle").style.color = done ? "#2D7A58" : WINE;
    $("statusSub").textContent = done
      ? num(state.results.metrics.total) + " records analysed"
      : "Upload a dataset or load the sample";
  }

  // ── Page: Dashboard ─────────────────────────────────────────────────────
  function renderDashboard() {
    var host = $("dashboardBody");
    if (!state.results) {
      host.innerHTML =
        '<div class="card"><h3>👋 Welcome to the Customer Feedback Analysis Agent</h3>' +
        '<p class="muted">This autonomous agent runs a 12-step pipeline over your feedback: cleaning, sentiment, emotion, ' +
        'topic detection, issue detection, recurring-issue clustering, priority scoring, trend analysis, insights and recommendations.</p>' +
        '<div class="btn-row" style="margin-top:14px">' +
        '<button class="btn" data-action="sample">📊 Load Sample Dataset</button>' +
        '<button class="btn ghost" data-goto="upload">📂 Upload Your Own</button>' +
        '<button class="btn ghost" data-goto="analyze">🔍 Analyze Single Feedback</button>' +
        '</div></div>';
      return;
    }

    var r = state.results, m = r.metrics, pd = r.priority_distribution || {};
    var html = '<div class="grid kpi">' +
      kpi("Total Feedback", num(m.total), "records analysed") +
      kpi("Positive", m.positive_pct + "%", num(m.positive) + " records", "#2D7A58", "linear-gradient(90deg,#2D7A58,#459B73)") +
      kpi("Negative", m.negative_pct + "%", num(m.negative) + " records", "#9C1D3A", "linear-gradient(90deg,#9C1D3A,#B33951)") +
      kpi("Complaints", num(m.complaints), m.complaint_pct + "% of all feedback", "#BA6A24", "linear-gradient(90deg,#BA6A24,#C5A059)") +
      kpi("High Priority", num(pd.HIGH || 0), "need urgent action", "#8B1430", "linear-gradient(90deg,#8B1430,#722F37)") +
      '</div>';

    html += '<div class="grid two" style="margin-top:18px">' +
      '<div class="card"><h3>Sentiment Distribution</h3><div class="chart" id="chSent"></div></div>' +
      '<div class="card"><h3>Priority Breakdown</h3><div class="chart" id="chPrio"></div></div>' +
      '<div class="card"><h3>Customer Emotions</h3><div class="chart" id="chEmo"></div></div>' +
      '<div class="card"><h3>Feedback Topics</h3><div class="chart" id="chTopic"></div></div>' +
      '</div>';

    if (r.trend_data && r.trend_data.date_available) {
      html += '<div class="grid two" style="margin-top:4px">' +
        '<div class="card"><h3>Sentiment Over Time</h3><div class="chart" id="chTrend"></div></div>' +
        '<div class="card"><h3>Feedback Volume Over Time</h3><div class="chart" id="chVol"></div></div>' +
        '</div>';
    }

    var q = r.quality_report || {};
    html += '<div class="card"><h3>Data Quality Report</h3><div class="grid kpi">' +
      kpi("Received", num(q.records_received), "raw records") +
      kpi("Valid", num(q.valid_records), "after cleaning", "#2D7A58") +
      kpi("Duplicates Removed", num(q.duplicates_removed), "") +
      kpi("Missing Handled", num(q.missing_handled), "") +
      '</div></div>';

    host.innerHTML = html;
    drawDashboardCharts();
  }

  function drawDashboardCharts() {
    var r = state.results;
    var sc = r.sentiment_counts || {}, labels = Object.keys(sc);
    plot($("chSent"), [{
      type: "pie", hole: 0.55, labels: labels, values: labels.map(function (k) { return sc[k]; }),
      marker: { colors: labels.map(function (k) { return PALETTE.sentiment[k] || WINE; }) },
      textinfo: "label+percent", sort: false
    }], { showlegend: true, legend: { orientation: "h", y: -0.12 }, margin: { l: 10, r: 10, t: 10, b: 40 } });

    var pd = r.priority_distribution || {}, pk = ["HIGH", "MEDIUM", "LOW"];
    plot($("chPrio"), [{
      type: "pie", hole: 0.62, labels: pk, values: pk.map(function (k) { return pd[k] || 0; }),
      marker: { colors: pk.map(function (k) { return PALETTE.priority[k]; }) },
      textinfo: "label+value", sort: false
    }], { showlegend: true, legend: { orientation: "h", y: -0.12 }, margin: { l: 10, r: 10, t: 10, b: 40 } });

    var ec = r.emotion_counts || {}, ek = Object.keys(ec).sort(function (a, b) { return ec[b] - ec[a]; });
    plot($("chEmo"), [{
      type: "bar", x: ek, y: ek.map(function (k) { return ec[k]; }),
      marker: { color: ek.map(function (k) { return PALETTE.emotion[k] || WINE_LIGHT; }) },
      text: ek.map(function (k) { return ec[k]; }), textposition: "outside"
    }], { showlegend: false });

    var tc = r.topic_counts || {}, tk = Object.keys(tc).sort(function (a, b) { return tc[a] - tc[b]; });
    plot($("chTopic"), [{
      type: "bar", orientation: "h", y: tk, x: tk.map(function (k) { return tc[k]; }),
      marker: { color: WINE }, text: tk.map(function (k) { return tc[k]; }), textposition: "outside"
    }], { showlegend: false, margin: { l: 130, r: 30, t: 20, b: 40 } });

    var td = r.trend_data || {};
    if (td.date_available) {
      var st = td.sentiment_trend || [];
      if (st.length) {
        var dkey = Object.keys(st[0])[0];
        var series = Object.keys(st[0]).filter(function (k) { return k !== dkey; });
        plot($("chTrend"), series.map(function (s) {
          return {
            type: "scatter", mode: "lines+markers", name: s,
            x: st.map(function (row) { return row[dkey]; }),
            y: st.map(function (row) { return row[s]; }),
            line: { color: PALETTE.sentiment[s] || WINE, width: 2.5 }
          };
        }), { legend: { orientation: "h", y: -0.2 } });
      }
      var vt = td.volume_trend || [];
      if (vt.length) {
        var vk = Object.keys(vt[0]);
        plot($("chVol"), [{
          type: "scatter", mode: "lines", fill: "tozeroy", name: "Volume",
          x: vt.map(function (row) { return row[vk[0]]; }),
          y: vt.map(function (row) { return row[vk[1]]; }),
          line: { color: WINE, width: 2.5 }, fillcolor: "rgba(114,47,55,0.14)"
        }], { showlegend: false });
      }
    }
  }

  // ── Page: Issues ────────────────────────────────────────────────────────
  function renderIssues() {
    var host = $("issuesBody");
    if (!state.results) { host.innerHTML = emptyState("No analysis yet. Upload a dataset to see detected issues.", true); return; }

    var r = state.results, html = "";
    var sum = r.issue_summary || [];

    html += '<div class="card"><h3>📋 Issue Summary</h3>';
    if (!sum.length) {
      html += '<p class="muted">No distinct issues were detected in this dataset.</p>';
    } else {
      html += '<div class="table-wrap"><table><thead><tr>' +
        '<th>Issue</th><th>Occurrences</th><th>% of Feedback</th><th>Avg Confidence</th><th>Avg Sentiment</th>' +
        '</tr></thead><tbody>';
      sum.forEach(function (row) {
        html += '<tr><td><strong>' + esc(row.detected_issue) + '</strong></td>' +
          '<td>' + num(row.occurrences) + '</td>' +
          '<td>' + esc(row.percentage) + '%</td>' +
          '<td>' + esc(row.avg_confidence) + '%</td>' +
          '<td>' + badge(row.avg_sentiment) + '</td></tr>';
      });
      html += '</tbody></table></div>';
    }
    html += '</div>';

    var rec = r.recurring_issues || [];
    html += '<div class="card"><h3>🔁 Recurring Issue Clusters</h3>';
    if (!rec.length) {
      html += '<p class="muted">No recurring clusters were found (a cluster needs at least 3 similar complaints).</p>';
    } else {
      html += '<div class="chart" id="chRecur"></div>';
      rec.forEach(function (c) {
        var cls = c.priority === "HIGH" ? "danger" : (c.priority === "MEDIUM" ? "warning" : "success");
        html += '<div class="item ' + cls + '"><h4>' + esc(c.issue_label) + ' ' + badge(c.priority) + '</h4>' +
          '<div class="meta"><span><strong>' + num(c.occurrences) + '</strong> occurrences</span>' +
          '<span>' + esc(c.percentage) + '% of feedback</span>' +
          '<span>Avg sentiment: ' + esc(c.avg_sentiment) + '</span></div>';
        if (c.sample_feedbacks && c.sample_feedbacks.length) {
          html += '<div style="margin-top:9px;font-size:12.5px;color:var(--text-secondary)"><em>Examples:</em><ul style="margin:5px 0 0;padding-left:18px">';
          c.sample_feedbacks.slice(0, 3).forEach(function (s) { html += '<li>' + esc(s) + '</li>'; });
          html += '</ul></div>';
        }
        html += '</div>';
      });
    }
    html += '</div>';
    host.innerHTML = html;

    if (rec.length) {
      var top = rec.slice(0, 12).slice().reverse();
      plot($("chRecur"), [{
        type: "bar", orientation: "h",
        y: top.map(function (c) { return c.issue_label; }),
        x: top.map(function (c) { return c.occurrences; }),
        marker: { color: top.map(function (c) { return PALETTE.priority[c.priority] || WINE; }) },
        text: top.map(function (c) { return c.occurrences; }), textposition: "outside"
      }], { showlegend: false, margin: { l: 170, r: 30, t: 20, b: 40 } });
    }
  }

  // ── Page: Insights ──────────────────────────────────────────────────────
  function renderInsights() {
    var host = $("insightsBody");
    if (!state.results) { host.innerHTML = emptyState("No analysis yet. Upload a dataset to generate insights.", true); return; }
    var ins = state.results.insights || [];
    if (!ins.length) { host.innerHTML = '<div class="card"><p class="muted">No insights were generated for this dataset.</p></div>'; return; }

    var html = '<div class="card"><h3>💡 ' + ins.length + ' Business Insights</h3>' +
      '<p class="muted">Each insight is derived directly from the analysed records — no guesswork.</p></div>';
    ins.forEach(function (i) {
      var sev = (i.severity || "info").toLowerCase();
      var cls = sev === "critical" || sev === "high" || sev === "danger" ? "danger"
        : sev === "warning" || sev === "medium" ? "warning"
          : sev === "positive" || sev === "success" ? "success" : "info";
      html += '<div class="item ' + cls + '"><h4>' + esc(i.icon || "") + ' ' + esc(i.title) + '</h4>' +
        '<p>' + esc(i.description) + '</p></div>';
    });
    host.innerHTML = html;
  }

  // ── Page: Recommendations ───────────────────────────────────────────────
  function renderRecs() {
    var host = $("recsBody");
    if (!state.results) { host.innerHTML = emptyState("No analysis yet. Upload a dataset to get recommendations.", true); return; }
    var recs = state.results.recommendations || [];
    if (!recs.length) { host.innerHTML = '<div class="card"><p class="muted">No recommendations were generated — no significant issues detected.</p></div>'; return; }

    var high = recs.filter(function (r) { return r.priority === "HIGH"; }).length;
    var html = '<div class="card"><h3>🎯 ' + recs.length + ' Recommended Actions</h3>' +
      '<p class="muted">' + high + ' high-priority action' + (high === 1 ? "" : "s") + ' require immediate attention.</p></div>';

    recs.forEach(function (r) {
      var cls = r.priority === "HIGH" ? "danger" : (r.priority === "MEDIUM" ? "warning" : "success");
      html += '<div class="item ' + cls + '"><h4>' + esc(r.action) + ' ' + badge(r.priority) + '</h4>' +
        '<p>' + esc(r.details) + '</p><div class="meta">' +
        '<span><strong>Issue:</strong> ' + esc(r.issue) + '</span>' +
        '<span><strong>Occurrences:</strong> ' + num(r.occurrences) + ' (' + esc(r.percentage) + '%)</span>' +
        (r.department ? '<span><strong>Owner:</strong> ' + esc(r.department) + '</span>' : '') +
        (r.kpi ? '<span><strong>KPI:</strong> ' + esc(r.kpi) + '</span>' : '') +
        '</div></div>';
    });
    host.innerHTML = html;
  }

  // ── Page: Activity log ──────────────────────────────────────────────────
  function renderActivity() {
    var host = $("activityBody");
    if (!state.results) { host.innerHTML = emptyState("No analysis yet. The agent's activity log appears here after a run.", true); return; }
    var log = state.results.activity_log || [];
    var html = '<div class="card"><h3>🤖 Agent Activity Log</h3>' +
      '<p class="muted">' + log.length + ' steps executed autonomously.</p><div style="margin-top:10px">';
    log.forEach(function (a) {
      html += '<div class="log-row ' + (a.status === "error" ? "error" : "") + '">' +
        '<div class="log-step">' + esc(a.step) + '</div><div style="min-width:0;flex:1">' +
        '<div class="log-tool">' + esc(a.tool) + '</div>' +
        '<div class="log-desc">' + esc(a.description) + '</div>' +
        '<div class="log-time">' + esc(a.timestamp) + (a.duration_ms ? ' · ' + esc(a.duration_ms) + 'ms' : '') + '</div>' +
        '</div></div>';
    });
    html += '</div></div>';
    host.innerHTML = html;
  }

  // ── Page: Reports ───────────────────────────────────────────────────────
  function renderReports() {
    var host = $("reportsBody");
    if (!state.results) { host.innerHTML = emptyState("No analysis yet. Run the agent first, then export a report.", true); return; }

    var m = state.results.metrics;
    var html = '<div class="card"><h3>📊 Export Analysis</h3>' +
      '<p class="muted">Reports are generated on the server from the current analysis (' + num(m.total) + ' records).</p>' +
      '<div class="btn-row" style="margin-top:14px">' +
      '<button class="btn" id="btnPdf">📄 Download PDF Report</button>' +
      '<button class="btn ghost" id="btnCsv">📑 Download Analysed CSV</button>' +
      '</div><div id="reportStatus"></div></div>';

    html += '<div class="card"><h3>⚡ AI Executive Summary</h3>' +
      '<p class="muted">Optional Groq LLM debrief. Requires <code>GROQ_API_KEY</code> to be set as a Vercel environment variable.</p>' +
      '<div class="btn-row" style="margin-top:12px"><button class="btn ghost" id="btnExec"' + (state.groq ? '' : ' disabled') + '>' +
      (state.groq ? '⚡ Generate Executive Summary' : '⚡ Groq LLM not configured') + '</button></div>' +
      '<div id="execBody"></div></div>';

    host.innerHTML = html;

    $("btnPdf").addEventListener("click", function () { downloadReport("pdf", this); });
    $("btnCsv").addEventListener("click", function () { downloadReport("csv", this); });
    var be = $("btnExec"); if (be && state.groq) be.addEventListener("click", execSummary);
  }

  function downloadReport(kind, btn) {
    var r = state.results, status = $("reportStatus");
    btn.disabled = true;
    var old = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Generating…';
    status.innerHTML = "";

    fetch("/api/report/" + kind, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows: r.rows,
        quality_report: r.quality_report,
        recurring_issues: r.recurring_issues,
        insights: r.insights,
        recommendations: r.recommendations,
        trend_data: { date_available: r.trend_data.date_available }
      })
    }).then(function (res) {
      if (!res.ok) return res.json().then(function (j) { throw new Error(j.error || "Export failed"); });
      return res.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = kind === "pdf" ? "feedback_analysis_report.pdf" : "analyzed_feedback.csv";
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
      status.innerHTML = '<div class="notice ok" style="margin-top:14px">✅ Download started.</div>';
    }).catch(function (err) {
      status.innerHTML = '<div class="notice error" style="margin-top:14px">' + esc(err.message) + '</div>';
    }).then(function () {
      btn.disabled = false; btn.innerHTML = old;
    });
  }

  function execSummary() {
    var btn = $("btnExec"), body = $("execBody"), r = state.results;
    btn.disabled = true;
    body.innerHTML = '<div class="progress"><i></i></div>';
    fetch("/api/executive-summary", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metrics: r.metrics, top_issues: (r.recurring_issues || []).slice(0, 5) })
    }).then(function (res) { return res.json(); }).then(function (j) {
      if (!j.ok) { body.innerHTML = '<div class="notice error" style="margin-top:12px">' + esc(j.error) + '</div>'; return; }
      var s = j.summary;
      var html = '<div class="item info" style="margin-top:14px"><h4>' + esc(s.executive_headline || "Executive Debrief") + '</h4>' +
        '<p>' + esc(s.strategic_diagnosis || "") + '</p>';
      if (s.key_priorities && s.key_priorities.length) {
        html += '<ul style="margin:10px 0 0;padding-left:20px;font-size:13.5px;color:var(--text-secondary)">';
        s.key_priorities.forEach(function (p) { html += '<li>' + esc(p) + '</li>'; });
        html += '</ul>';
      }
      html += '<div class="meta">Generated by ' + esc(j.model) + '</div></div>';
      body.innerHTML = html;
    }).catch(function (e) {
      body.innerHTML = '<div class="notice error" style="margin-top:12px">' + esc(e.message) + '</div>';
    }).then(function () { btn.disabled = false; });
  }

  // ── Page: Explorer ──────────────────────────────────────────────────────
  function renderExplorer() {
    var host = $("explorerBody");
    if (!state.results) { host.innerHTML = emptyState("No analysis yet. Upload a dataset to explore records.", true); return; }
    var r = state.results;

    var html = '<div class="card"><div class="field-row">' +
      '<div><label class="field" for="fltSearch">Search text</label><input type="search" id="fltSearch" placeholder="Search feedback…"></div>' +
      '<div><label class="field" for="fltSent">Sentiment</label><select id="fltSent"><option value="">All</option><option>Positive</option><option>Neutral</option><option>Negative</option></select></div>' +
      '<div><label class="field" for="fltPrio">Priority</label><select id="fltPrio"><option value="">All</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></select></div>' +
      '<div><label class="field" for="fltTopic">Topic</label><select id="fltTopic"><option value="">All</option>' +
      Object.keys(r.topic_counts || {}).map(function (t) { return '<option>' + esc(t) + '</option>'; }).join("") +
      '</select></div>' +
      '</div><div id="explorerCount" class="muted"></div></div>' +
      '<div id="explorerTable"></div>';
    host.innerHTML = html;

    ["fltSearch", "fltSent", "fltPrio", "fltTopic"].forEach(function (id) {
      $(id).addEventListener("input", drawExplorerTable);
      $(id).addEventListener("change", drawExplorerTable);
    });
    drawExplorerTable();
  }

  function drawExplorerTable() {
    var r = state.results, fcol = (r.columns && r.columns.feedback) || "feedback";
    var q = ($("fltSearch").value || "").toLowerCase();
    var s = $("fltSent").value, p = $("fltPrio").value, t = $("fltTopic").value;

    var rows = (r.rows || []).filter(function (row) {
      if (s && row.sentiment !== s) return false;
      if (p && row.priority !== p) return false;
      if (t && row.topic !== t) return false;
      if (q && String(row[fcol] || "").toLowerCase().indexOf(q) === -1) return false;
      return true;
    });

    $("explorerCount").textContent = "Showing " + num(Math.min(rows.length, 300)) + " of " + num(rows.length) + " matching records (of " + num(r.rows.length) + " total).";

    var html = '<div class="table-wrap"><table><thead><tr>' +
      '<th>Feedback</th><th>Sentiment</th><th>Emotion</th><th>Topic</th><th>Issue</th><th>Priority</th><th>Score</th>' +
      '</tr></thead><tbody>';
    rows.slice(0, 300).forEach(function (row) {
      html += '<tr><td class="wrap">' + esc(row[fcol]) + '</td>' +
        '<td>' + badge(row.sentiment) + '</td>' +
        '<td>' + esc(row.emotion) + '</td>' +
        '<td>' + esc(row.topic) + '</td>' +
        '<td>' + esc(row.detected_issue) + '</td>' +
        '<td>' + badge(row.priority) + '</td>' +
        '<td>' + esc(row.priority_score) + '</td></tr>';
    });
    if (!rows.length) html += '<tr><td colspan="7" style="text-align:center;padding:26px;color:var(--text-secondary)">No records match these filters.</td></tr>';
    html += '</tbody></table></div>';
    $("explorerTable").innerHTML = html;
  }

  // ── Page: Upload ────────────────────────────────────────────────────────
  function renderUploadPreview() { /* preview is rendered on file select */ }

  function setColumnOptions(cols, detected) {
    var mk = function (sel, optional) {
      sel.innerHTML = (optional ? '<option value="">— none —</option>' : '') +
        cols.map(function (c) { return '<option value="' + esc(c) + '">' + esc(c) + '</option>'; }).join("");
    };
    mk($("colFeedback"), false);
    mk($("colDate"), true);
    mk($("colRating"), true);
    if (detected) {
      if (detected.feedback) $("colFeedback").value = detected.feedback;
      if (detected.date) $("colDate").value = detected.date;
      if (detected.rating) $("colRating").value = detected.rating;
    }
  }

  function handleFile(file) {
    if (!file) return;
    state.pendingFile = file;
    $("btnAnalyzeFile").disabled = false;
    $("uploadStatus").innerHTML = '<div class="notice info" style="margin-top:14px">📄 <strong>' + esc(file.name) +
      '</strong> ready — ' + (file.size / 1024).toFixed(1) + ' KB. Column names are auto-detected; adjust below if needed.</div>';

    // Read the header row locally so the user can pick columns before uploading.
    if (/\.csv$/i.test(file.name)) {
      var reader = new FileReader();
      reader.onload = function (e) {
        var firstLine = String(e.target.result).split(/\r?\n/)[0] || "";
        var cols = firstLine.split(",").map(function (c) { return c.replace(/^"|"$/g, "").trim(); }).filter(Boolean);
        if (cols.length) {
          setColumnOptions(cols, guessColumns(cols));
          $("colPicker").classList.remove("hidden");
        }
      };
      reader.readAsText(file.slice(0, 8192));
    } else {
      $("colPicker").classList.add("hidden");
    }
  }

  function guessColumns(cols) {
    var lower = cols.map(function (c) { return c.toLowerCase(); });
    var pick = function (cands) {
      for (var i = 0; i < cands.length; i++) {
        var idx = lower.indexOf(cands[i]);
        if (idx !== -1) return cols[idx];
      }
      return "";
    };
    return {
      feedback: pick(["feedback", "review", "comment", "customer_feedback", "review_text", "complaint", "text", "message", "description"]) || cols[0],
      date: pick(["date", "created_at", "timestamp", "review_date", "feedback_date", "time"]),
      rating: pick(["rating", "score", "stars", "star_rating", "nps"])
    };
  }

  function runAnalysis(opts) {
    var status = $("uploadStatus");
    var btns = [$("btnAnalyzeFile"), $("btnSample")];
    btns.forEach(function (b) { if (b) b.disabled = true; });
    status.innerHTML = '<div class="notice info" style="margin-top:14px">🤖 The agent is running its 12-step pipeline…<div class="progress"><i></i></div></div>';

    var req;
    if (opts.sample) {
      req = fetch("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: true })
      });
    } else {
      var fd = new FormData();
      fd.append("file", state.pendingFile);
      if (!$("colPicker").classList.contains("hidden")) {
        fd.append("feedback_col", $("colFeedback").value || "");
        fd.append("date_col", $("colDate").value || "");
        fd.append("rating_col", $("colRating").value || "");
      }
      req = fetch("/api/analyze", { method: "POST", body: fd });
    }

    req.then(function (res) { return res.json().catch(function () { throw new Error("Server returned an unexpected response."); }); })
      .then(function (j) {
        if (!j.ok) throw new Error(j.error || "Analysis failed.");
        state.results = j;
        save();
        statusPill();
        var warn = j.truncated ? ' Only the first ' + num(j.max_rows) + ' rows were analysed to stay within serverless limits.' : '';
        status.innerHTML = '<div class="notice ok" style="margin-top:14px">✅ Analysis complete — ' +
          num(j.metrics.total) + ' records processed.' + esc(warn) + '</div>';
        go("dashboard");
      })
      .catch(function (err) {
        status.innerHTML = '<div class="notice error" style="margin-top:14px">❌ ' + esc(err.message) + '</div>';
      })
      .then(function () {
        btns.forEach(function (b) { if (b) b.disabled = false; });
        $("btnAnalyzeFile").disabled = !state.pendingFile;
      });
  }

  // ── Page: Single feedback ───────────────────────────────────────────────
  var EXAMPLES = [
    "My order arrived three days late and the box was completely damaged. I emailed support twice and nobody replied.",
    "The app is fast and the checkout was effortless. Delivery came a day early — really impressed with the service.",
    "I was charged twice for the same order and the refund still has not appeared after two weeks. Very frustrating."
  ];

  function analyzeSingle() {
    var text = $("singleText").value.trim();
    var host = $("singleResult");
    if (!text) { host.innerHTML = '<div class="notice error">Please enter some feedback text first.</div>'; return; }

    var btn = $("btnSingle");
    btn.disabled = true;
    host.innerHTML = '<div class="card"><div class="progress"><i></i></div></div>';

    fetch("/api/analyze-single", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (!j.ok) throw new Error(j.error);
      state.single = j.result;
      renderSingle(j.result, text);
    }).catch(function (e) {
      host.innerHTML = '<div class="notice error">' + esc(e.message) + '</div>';
    }).then(function () { btn.disabled = false; });
  }

  function renderSingle(res, text) {
    var html = '<div class="grid kpi">' +
      kpi("Sentiment", res.sentiment, (res.sentiment_confidence || 0) + "% confidence", PALETTE.sentiment[res.sentiment]) +
      kpi("Emotion", res.emotion, (res.emotion_confidence || 0) + "% confidence", PALETTE.emotion[res.emotion]) +
      kpi("Topic", res.topic, (res.topic_confidence || 0) + "% confidence") +
      kpi("Priority", res.priority, "score " + (res.priority_score || 0), PALETTE.priority[res.priority]) +
      '</div>';

    html += '<div class="card" style="margin-top:18px"><h3>Agent Findings</h3>' +
      '<div class="meta item info"><h4>' + esc(res.detected_issue || "No specific issue") + ' ' + badge(res.priority) + '</h4>' +
      '<p><strong>Complaint:</strong> ' + (res.is_complaint ? "Yes" : "No") +
      ' &nbsp;·&nbsp; <strong>Issue confidence:</strong> ' + esc(res.issue_confidence || 0) + '%</p></div>';

    if (res.priority_reasons && res.priority_reasons.length) {
      html += '<p class="muted" style="margin-top:10px"><strong>Why this priority:</strong></p><ul style="margin:4px 0 0;padding-left:20px;font-size:13.5px;color:var(--text-secondary)">';
      res.priority_reasons.forEach(function (r) { html += '<li>' + esc(r) + '</li>'; });
      html += '</ul>';
    }
    html += '</div>';

    html += '<div class="card"><h3>🎯 Recommended Action</h3>' +
      '<div class="item success"><h4>' + esc(res.recommended_action || "") + '</h4>' +
      '<p>' + esc(res.recommendation_details || "") + '</p></div></div>';

    html += '<div class="card"><h3>⚡ AI Deep Dive</h3>' +
      '<p class="muted">Optional Groq LLM root-cause breakdown. Requires <code>GROQ_API_KEY</code>.</p>' +
      '<div class="btn-row" style="margin-top:10px"><button class="btn ghost" id="btnExplain"' + (state.groq ? '' : ' disabled') + '>' +
      (state.groq ? '⚡ Explain with Groq LLM' : '⚡ Groq LLM not configured') + '</button></div>' +
      '<div id="explainBody"></div></div>';

    $("singleResult").innerHTML = html;

    var be = $("btnExplain");
    if (be && state.groq) {
      be.addEventListener("click", function () {
        be.disabled = true;
        $("explainBody").innerHTML = '<div class="progress"><i></i></div>';
        fetch("/api/explain", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: text, sentiment: res.sentiment, topic: res.topic,
            issue: res.detected_issue, priority: res.priority
          })
        }).then(function (r) { return r.json(); }).then(function (j) {
          $("explainBody").innerHTML = j.ok
            ? '<div class="pre" style="margin-top:12px">' + esc(j.explanation) + '</div>'
            : '<div class="notice error" style="margin-top:12px">' + esc(j.error) + '</div>';
        }).catch(function (e) {
          $("explainBody").innerHTML = '<div class="notice error" style="margin-top:12px">' + esc(e.message) + '</div>';
        }).then(function () { be.disabled = false; });
      });
    }
  }

  // ── Router ──────────────────────────────────────────────────────────────
  function render(page) {
    statusPill();
    if (page === "dashboard") renderDashboard();
    else if (page === "issues") renderIssues();
    else if (page === "insights") renderInsights();
    else if (page === "recommendations") renderRecs();
    else if (page === "activity") renderActivity();
    else if (page === "reports") renderReports();
    else if (page === "explorer") renderExplorer();
    else if (page === "upload") renderUploadPreview();
  }

  // ── Init ────────────────────────────────────────────────────────────────
  function init() {
    restore();

    document.getElementById("nav").addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-page]");
      if (btn) go(btn.getAttribute("data-page"));
    });

    document.body.addEventListener("click", function (e) {
      var g = e.target.closest("[data-goto]");
      if (g) { go(g.getAttribute("data-goto")); return; }
      var a = e.target.closest('[data-action="sample"]');
      if (a) { go("upload"); runAnalysis({ sample: true }); }
    });

    $("menuBtn").addEventListener("click", function () {
      $("sidebar").classList.contains("open") ? closeMenu() : openMenu();
    });
    $("backdrop").addEventListener("click", closeMenu);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeMenu(); });

    // Upload interactions
    $("fileInput").addEventListener("change", function (e) { handleFile(e.target.files[0]); });
    var dz = $("dropzone");
    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("drag"); });
    });
    dz.addEventListener("drop", function (e) {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        $("fileInput").files = e.dataTransfer.files;
        handleFile(e.dataTransfer.files[0]);
      }
    });
    $("btnAnalyzeFile").addEventListener("click", function () { runAnalysis({ sample: false }); });
    $("btnSample").addEventListener("click", function () { runAnalysis({ sample: true }); });

    // Single feedback
    $("btnSingle").addEventListener("click", analyzeSingle);
    $("btnSingleSample").addEventListener("click", function () {
      $("singleText").value = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
    });

    // Server status (Groq availability)
    fetch("/api/status").then(function (r) { return r.json(); }).then(function (j) {
      state.groq = !!j.groq_available;
      state.model = j.groq_model || "";
      var pill = $("llmPill"), dot = $("llmDot");
      pill.className = "pill " + (state.groq ? "complete" : "ready");
      dot.className = "dot " + (state.groq ? "complete" : "ready");
      $("llmTitle").textContent = state.groq ? "GROQ LLM ACTIVE" : "GROQ LLM OFFLINE";
      $("llmTitle").style.color = state.groq ? "#2D7A58" : WINE;
      $("llmSub").textContent = state.groq ? state.model : "Optional — rule-based engine active";
    }).catch(function () { /* non-fatal */ });

    go("dashboard");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
