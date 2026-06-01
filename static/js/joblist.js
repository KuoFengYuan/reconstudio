// Joblist UI — filter / search / paginate / SSE refresh.
// The hx-trigger="load" on #joblist does the initial fetch; this file drives
// every refresh after that, both user-initiated (chip / search / load-more)
// and server-initiated (SSE 'refresh' on any job state change).
(function () {
  "use strict";
  const LS_KEY = "rs_joblist_v1";
  const DEFAULTS = { q: "", kind: "all", status: "all", limit: 50 };

  function loadState() {
    try { return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(LS_KEY) || "{}")); }
    catch (_) { return Object.assign({}, DEFAULTS); }
  }
  function saveState() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(state)); } catch (_) {}
  }
  const state = loadState();
  window._jobUi = state;

  function buildUrl() {
    const u = new URLSearchParams();
    if (state.q) u.set("q", state.q);
    if (state.kind && state.kind !== "all") u.set("kind", state.kind);
    if (state.status && state.status !== "all") u.set("status", state.status);
    if (state.limit && state.limit !== DEFAULTS.limit) u.set("limit", String(state.limit));
    const qs = u.toString();
    return "/ui/joblist" + (qs ? "?" + qs : "");
  }

  // Refetch the joblist fragment with current state. Throttled so a burst of
  // SSE/chip clicks doesn't spam the server.
  let reloadPending = false;
  function reloadJoblist() {
    if (!window.htmx) return;
    if (reloadPending) return;
    reloadPending = true;
    setTimeout(() => {
      reloadPending = false;
      window.htmx.ajax("GET", buildUrl(), { target: "#joblist", swap: "innerHTML" });
    }, 50);
  }
  window.reloadJoblist = reloadJoblist;

  // Instant (no round-trip) reflection of the current kind/status filter on the
  // rows already in the DOM, plus the chips' active state. The authoritative
  // refetch (reloadJoblist) still runs in the background to fix pagination and
  // the "顯示 X / 共 Y" footer — this just removes the perceived click latency,
  // since chip counts are full-set totals and never change with the filter.
  function applyChipActive() {
    document.querySelectorAll("#joblist .jobfilter button[data-axis]").forEach((b) => {
      const axis = b.getAttribute("data-axis");
      b.classList.toggle("active", b.getAttribute("data-val") === (state[axis] || "all"));
    });
  }
  function applyClientFilter() {
    document.querySelectorAll("#joblist tbody tr[data-kind]").forEach((tr) => {
      const okKind = state.kind === "all" || tr.getAttribute("data-kind") === state.kind;
      const okStatus = state.status === "all" || tr.getAttribute("data-status") === state.status;
      tr.style.display = okKind && okStatus ? "" : "none";
    });
  }

  // Chip handlers (kind / status). 'all' is the unset value.
  window.setJobFilter = function (axis, val) {
    if (axis !== "kind" && axis !== "status") return;
    state[axis] = val || "all";
    // 'Load more' progress is per-filter-view: reset on filter change so the
    // user doesn't keep an inflated limit from a previous view.
    state.limit = DEFAULTS.limit;
    saveState();
    applyChipActive();    // instant feedback — don't wait for the server
    applyClientFilter();
    reloadJoblist();      // authoritative: correct pagination + footer counts
  };
  // Back-compat aliases (older templates still call these)
  window.filterJobs = (k) => window.setJobFilter("kind", k);
  window.filterJobsStatus = (s) => window.setJobFilter("status", s);

  window.loadMoreJobs = function () {
    state.limit = (state.limit || DEFAULTS.limit) + DEFAULTS.limit;
    saveState();
    reloadJoblist();
  };

  let searchT = null;
  window.onJobSearchInput = function (v) {
    clearTimeout(searchT);
    searchT = setTimeout(() => {
      const next = (v || "").trim();
      if (next === state.q) return;
      state.q = next;
      state.limit = DEFAULTS.limit;
      saveState();
      reloadJoblist();
    }, 250);
  };

  // Select-all checkbox in the table header
  window.toggleAllJobs = function (cb) {
    document.querySelectorAll(".jobsel").forEach((x) => { x.checked = cb.checked; });
  };

  // Multi-select delete
  window.deleteSelected = function () {
    const ids = Array.from(document.querySelectorAll(".jobsel:checked")).map((x) => x.value);
    if (!ids.length) { alert("請先勾選要刪除的紀錄"); return; }
    if (!confirm("刪除 " + ids.length + " 筆紀錄?(進行中的會先取消;已完成的會連同 log 一併移除)")) return;
    fetch("/api/jobs/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: ids }),
    }).then((r) => r.json()).then(() => reloadJoblist());
  };

  // Preserve table state across swaps: checked rows, scroll position, and
  // the search box's focus + caret (SSE refresh shouldn't kill mid-typing).
  let savedSel = new Set();
  let savedScroll = 0;
  let savedSearchFocus = null;   // {value, selStart, selEnd} or null
  document.body.addEventListener("htmx:beforeSwap", function (e) {
    const t = e.detail && e.detail.target;
    if (!t || t.id !== "joblist") return;
    savedSel = new Set(Array.from(document.querySelectorAll(".jobsel:checked")).map((x) => x.value));
    const sc = t.closest(".col"); savedScroll = sc ? sc.scrollTop : 0;
    const inp = document.getElementById("jobsearch");
    if (inp && document.activeElement === inp) {
      savedSearchFocus = { value: inp.value, selStart: inp.selectionStart, selEnd: inp.selectionEnd };
    } else {
      savedSearchFocus = null;
    }
  });
  document.body.addEventListener("htmx:afterSwap", function (e) {
    const t = e.detail && e.detail.target;
    if (!t || t.id !== "joblist") return;
    document.querySelectorAll(".jobsel").forEach((x) => { if (savedSel.has(x.value)) x.checked = true; });
    const sc = t.closest(".col"); if (sc) sc.scrollTop = savedScroll;
    if (savedSearchFocus) {
      const inp = document.getElementById("jobsearch");
      if (inp) {
        inp.value = savedSearchFocus.value;
        inp.focus();
        try { inp.setSelectionRange(savedSearchFocus.selStart, savedSearchFocus.selEnd); } catch (_) {}
      }
    }
  });

  // ONE multiplexed WebSocket per tab, shared by the whole UI (window.rsBus).
  // It carries BOTH the job-list refresh signal AND the watched job's log lines,
  // replacing the old EventSource pair (/api/jobs/stream + /api/jobs/{id}/logs).
  // A tab now holds a single connection regardless of how many jobs/logs it
  // shows, so it can't exhaust the browser's ~6-per-host HTTP connection cap —
  // the root cause of the "Loading…" hangs while a long job ran across tabs.
  const bus = {
    ws: null,
    watching: null,            // job_id whose log we're tailing, or null
    onLog: null,               // log viewer sets this: (job_id, linesArray) => void
    onLogEnd: null,            // log viewer sets this: (job_id, status) => void
    _send(obj) {
      try { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj)); }
      catch (_) {}
    },
    watchLog(jobId) { this.watching = jobId || null; this._send({ action: "watch_log", job_id: jobId }); },
    unwatchLog() { this.watching = null; this._send({ action: "unwatch_log" }); },
    connect() {
      if (this.ws && (this.ws.readyState === 0 || this.ws.readyState === 1)) return;
      let sock;
      try {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        sock = new WebSocket(proto + "//" + location.host + "/ws");
      } catch (_) { return; }
      this.ws = sock;
      sock.onopen = () => { if (this.watching) this._send({ action: "watch_log", job_id: this.watching }); };
      sock.onmessage = (ev) => {
        let m; try { m = JSON.parse(ev.data); } catch (_) { return; }
        if (m.type === "jobs") reloadJoblist();
        else if (m.type === "log") { if (this.onLog) this.onLog(m.job, m.lines || []); }
        else if (m.type === "log_end") { if (this.onLogEnd) this.onLogEnd(m.job, m.status); }
      };
      // Reconnect after a short delay on drop (server restart, sleep, …); onopen
      // re-subscribes the active log so tailing resumes from the tail.
      sock.onclose = () => { this.ws = null; setTimeout(() => this.connect(), 1500); };
      sock.onerror = () => { try { sock.close(); } catch (_) {} };
    },
  };
  window.rsBus = bus;

  // Initial population guard:
  //   - If persisted state differs from defaults, refetch with those filters.
  //   - If after a short grace period #joblist is still empty (e.g., the
  //     hx-trigger="load" failed to parse, or htmx was blocked), force a fetch.
  //     This keeps the table populated even if the htmx attribute breaks.
  function applyInitialState() {
    if (state.q || state.kind !== "all" || state.status !== "all" || state.limit !== DEFAULTS.limit) {
      reloadJoblist();
      return;
    }
    setTimeout(() => {
      const div = document.getElementById("joblist");
      if (div && !div.innerHTML.trim()) reloadJoblist();
    }, 800);
  }

  function boot() { bus.connect(); applyInitialState(); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
