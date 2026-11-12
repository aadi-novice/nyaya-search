// Nyaya Search page. Renders the SearchResponse and AskResponse JSON contracts
// (contracts/schema.json). On submit it calls POST /search and shows the ranked
// passages at once, then POST /ask and renders the checked answer beside them.
// Plain DOM only: response text is always set with textContent, never parsed as HTML.
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, props, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "className") node.className = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children || []) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// --- notices and health ------------------------------------------------------------------

function renderNotices(notices) {
  const list = $("notice-list");
  clear(list);
  for (const n of notices || []) list.append(el("li", { text: n }));
}

async function loadNotices() {
  try {
    const r = await fetch("/health");
    if (!r.ok) return;
    const h = await r.json();
    renderNotices(h.notices);
    if (h.active_version) {
      $("index-version").textContent = "index " + h.active_version;
      $("index-version").hidden = false;
    }
  } catch (_) {
    // The page still works; each response carries its notices.
  }
}

// --- passage and citation details ----------------------------------------------------------

const ZONE_LABELS = {
  issue: "Issue for consideration (editorial summary)",
  editorial: "Headnote (editorial summary)",
  judgment: null,
};

const OPINION_LABELS = { dissenting: "Dissenting opinion", concurring: "Concurring opinion" };

function judgmentLink(ref) {
  return "/judgment/" + encodeURIComponent(ref.judgment_id) +
    "?highlight=" + encodeURIComponent(ref.chunk_id);
}

function plural(list, one, many) {
  return (list.length === 1 ? one : many) + " " + list.join(", ");
}

// "[2025] 1 S.C.R. 230, 2025 INSC 31, para 13, p. 236"
function citationText(ref) {
  const parts = [ref.scr_citation];
  if (ref.neutral_citation) parts.push(ref.neutral_citation);
  if (ref.paras && ref.paras.length) parts.push(plural(ref.paras, "para", "paras"));
  if (ref.pages && ref.pages.length) parts.push(plural(ref.pages, "p.", "pp."));
  return parts.filter(Boolean).join(", ");
}

function refTags(ref) {
  return [ZONE_LABELS[ref.zone], OPINION_LABELS[ref.opinion_type]]
    .filter(Boolean)
    .map((l) => el("span", { className: "tag", text: l }));
}

// "Case v. State, [2025] 1 S.C.R. 230, para 13, p. 236" plus zone and opinion labels.
function refDetails(ref) {
  return el("span", { className: "ref" }, [
    el("a", { href: judgmentLink(ref), className: "case-name", text: ref.case_name }),
    ", " + citationText(ref),
    ...refTags(ref),
  ]);
}

// A verified citation: a chip that jumps to the source passage, the reference, the quote.
function renderCitation(c) {
  const chip = el("button", {
    type: "button",
    className: "chip",
    text: c.passage_label,
    title: "Show this passage in the sources",
    "aria-label": "Show passage " + c.passage_label + " in the sources",
    onclick: () => {
      if (!focusSource(c.chunk_id, c.quote)) location.href = judgmentLink(c);
    },
  });
  return el("li", { className: "citation" }, [
    chip,
    refDetails(c),
    el("blockquote", { text: "“" + c.quote + "”" }),
  ]);
}

// A plain list of PassageRefs (no text): used when /search gave no passages.
function renderPassages(passages, refusedNote) {
  return el("ol", { className: "passages" }, passages.map((p) =>
    el("li", { className: "passage" + (refusedNote ? " not-answer" : "") }, [
      refDetails(p),
      refusedNote ? el("span", { className: "tag warn", text: "Does not answer the question" }) : null,
    ])));
}

// --- sources (POST /search) ----------------------------------------------------------------

const sources = new Map(); // chunk_id -> { card, textEl, text }

function formatDate(iso) {
  if (!iso) return null;
  const d = new Date(iso + "T00:00:00");
  if (isNaN(d)) return iso;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

function scoreView(score) {
  if (score === null || score === undefined) return null;
  const pct = Math.max(0, Math.min(1, score)) * 100;
  const bar = el("span", { className: "score-bar", "aria-hidden": "true" }, [el("span")]);
  bar.firstChild.style.width = pct.toFixed(0) + "%";
  return el("span", { className: "score", title: "Reranker relevance score" }, [
    bar, "relevance " + score.toFixed(2),
  ]);
}

function sourceCard(p) {
  const textEl = el("div", { className: "source-text", text: p.text });
  const card = el("li", { className: "source", id: "src-" + p.rank, tabindex: "-1" });
  const toggle = el("button", {
    type: "button",
    className: "link-button",
    text: "Show full passage",
    "aria-expanded": "false",
    onclick: () => setOpen(card, !card.classList.contains("open")),
  });
  const meta = [citationText(p), formatDate(p.decision_date)].filter(Boolean).join(" · ");
  card.append(
    el("div", { className: "source-head" }, [
      el("span", { className: "rank", text: String(p.rank) }),
      el("span", { className: "source-title" }, [
        el("a", { href: judgmentLink(p), text: p.case_name }),
      ]),
      el("span", { className: "cited-as", hidden: true }),
    ]),
    el("p", { className: "source-meta" }, [meta, ...refTags(p), scoreView(p.rerank_score)]),
    textEl,
    el("div", { className: "source-actions" }, [
      toggle,
      el("a", { href: judgmentLink(p), text: "Open in judgment" }),
    ]),
  );
  sources.set(p.chunk_id, { card, textEl, text: p.text, toggle });
  return card;
}

function setOpen(card, open) {
  card.classList.toggle("open", open);
  for (const s of sources.values()) {
    if (s.card === card) {
      s.toggle.textContent = open ? "Show less" : "Show full passage";
      s.toggle.setAttribute("aria-expanded", String(open));
    }
  }
}

function searchBanner(data) {
  if (data.status === "refused" && data.passages.length) {
    return "No passage scored above the refusal threshold. These are the nearest passages; " +
      "they may not answer the question.";
  }
  if (data.status === "refused") return "No passage matched the question.";
  if (data.status === "no_match_with_filters") {
    return "No judgment matches the filters: " + describeFilters(data.filters_applied || {}) + ".";
  }
  if (data.status === "error") return data.passages_note || "The passage search failed.";
  return data.passages_note ? capitalise(data.passages_note) + "." : null;
}

function capitalise(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function renderSources(data) {
  const box = $("sources");
  clear(box);
  sources.clear();
  const n = data.passages.length;
  const secs = data.timings_ms && data.timings_ms.total ? " · " + seconds(data.timings_ms.total) : "";
  $("sources-meta").textContent = n ? n + " passage" + (n === 1 ? "" : "s") + secs : "";
  const banner = searchBanner(data);
  if (banner) box.append(el("p", { className: "status-banner", text: banner }));
  if (n) box.append(el("ol", { className: "sources" }, data.passages.map(sourceCard)));
}

// Mark `quote` inside a passage's text. The match ignores case, punctuation and
// spacing, as the code check normalises them; no match leaves the text plain.
function quotePattern(quote) {
  const words = (quote || "").match(/[\p{L}\p{N}]+/gu);
  if (!words || words.length < 3) return null;
  return new RegExp(words.join("[^\\p{L}\\p{N}]+"), "iu");
}

function unmarkAll() {
  for (const s of sources.values()) {
    if (s.textEl.querySelector("mark")) s.textEl.textContent = s.text;
    s.card.classList.remove("flash");
  }
}

function markQuote(s, quote) {
  const re = quotePattern(quote);
  const m = re ? re.exec(s.text) : null;
  if (!m) return null;
  clear(s.textEl);
  const mark = el("mark", { text: m[0] });
  s.textEl.append(s.text.slice(0, m.index), mark, s.text.slice(m.index + m[0].length));
  return mark;
}

function focusSource(chunkId, quote) {
  const s = sources.get(chunkId);
  if (!s) return false;
  unmarkAll();
  setOpen(s.card, true);
  const mark = markQuote(s, quote);
  s.card.classList.add("flash");
  s.card.focus({ preventScroll: true });
  const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  (mark || s.card).scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "center" });
  setTimeout(() => s.card.classList.remove("flash"), 1800);
  return true;
}

// After /ask: mark the cited cards, or every card when the answer is a refusal.
function annotateSources(data) {
  const cited = new Map();
  for (const s of data.answer || []) {
    for (const c of s.citations) {
      if (!cited.has(c.chunk_id)) cited.set(c.chunk_id, new Set());
      cited.get(c.chunk_id).add(c.passage_label);
    }
  }
  for (const [chunkId, s] of sources) {
    const badge = s.card.querySelector(".cited-as");
    const labels = cited.get(chunkId);
    s.card.classList.toggle("cited", !!labels);
    badge.hidden = !labels;
    badge.textContent = labels ? "Cited as " + [...labels].join(", ") : "";
    const refused = data.status === "refused";
    s.card.classList.toggle("not-answer", refused);
    const meta = s.card.querySelector(".source-meta");
    if (refused && !meta.querySelector(".tag.warn")) {
      meta.append(el("span", { className: "tag warn", text: "Does not answer the question" }));
    }
  }
}

// --- answer parts --------------------------------------------------------------------------

function renderAnswer(sentences) {
  return el("ol", { className: "answer" }, sentences.map((s) =>
    el("li", { className: "sentence" }, [
      el("p", {}, [s.sentence, el("span", { className: "verified", text: "✓ quote checked" })]),
      el("ul", { className: "citations" }, s.citations.map(renderCitation)),
    ])));
}

function renderGaps(gaps) {
  if (!gaps || !gaps.length) return null;
  return el("div", { className: "gaps" }, [
    el("h3", { text: "Not covered by the passages" }),
    ...gaps.map((g) => el("p", { text: g })),
  ]);
}

const REASONS = {
  unknown_label: "cites a passage label that was not retrieved",
  quote_not_found: "the quote does not appear in the cited passage",
  quote_length: "the quote is not 8 to 40 words long",
  uncited_case_name: "names a case that is not among the passages",
  no_citation: "has no citation",
};

// Unverified sentences show only the sentence, the labels the model gave and the
// reason. They never show case names, citations or quotes.
function renderUnverified(items) {
  if (!items || !items.length) return null;
  const n = items.length;
  return el("details", { className: "unverified" }, [
    el("summary", { text: "Show " + n + " unverified sentence" + (n === 1 ? "" : "s") }),
    el("p", { className: "hint", text: "These sentences failed the citation check. Do not rely on them." }),
    el("ul", {}, items.map((u) =>
      el("li", {}, [
        el("p", { text: u.sentence }),
        el("p", { className: "why" }, [
          "Labels: " + (u.labels.length ? u.labels.join(", ") : "none") + ". ",
          "Not verified: " + (REASONS[u.reason] || u.reason) + ".",
        ]),
      ]))),
  ]);
}

function seconds(ms) {
  return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + " s";
}

function renderTimings(data) {
  const t = data.timings_ms || {};
  const parts = Object.entries(t).map(([k, v]) => k + " " + Math.round(v) + " ms");
  const v = data.versions || {};
  return el("p", { className: "timings" }, [
    parts.join(" · "),
    v.index_version ? " · index " + v.index_version : "",
    v.reranker_fallback_used ? " · fallback reranker used" : "",
  ]);
}

function describeFilters(f) {
  const out = [];
  if (f.year_from || f.year_to) out.push("years " + (f.year_from || "any") + "–" + (f.year_to || "any"));
  if (f.judge) out.push("judge " + f.judge);
  if (f.act) out.push("Act " + f.act);
  if (f.outcome) out.push("outcome " + f.outcome);
  for (const [k, v] of Object.entries(f)) {
    if (!["year_from", "year_to", "judge", "act", "outcome"].includes(k)) out.push(k + " " + v);
  }
  return out.join("; ");
}

// --- one /ask response ---------------------------------------------------------------------

function title(text) {
  return el("h3", { className: "state-title", text });
}

// `haveSources`: the passages are already on the page (from /search), so the
// answer panel does not list them again.
function renderResponse(data, haveSources) {
  const out = [];
  switch (data.status) {
    case "answered":
      out.push(renderAnswer(data.answer));
      out.push(renderGaps(data.gaps));
      out.push(renderUnverified(data.unverified));
      break;
    case "refused":
      out.push(title("No relevant judgment found"));
      out.push(el("p", { className: "state-body", text:
        "The retrieved passages do not support an answer, so none is given." +
        (haveSources && data.nearest_passages.length ? " The nearest passages are listed under Sources." : "") }));
      out.push(renderGaps(data.gaps));
      out.push(renderUnverified(data.unverified));
      if (!haveSources && data.nearest_passages.length) {
        out.push(el("h3", { className: "subhead", text: "Nearest passages" }));
        out.push(renderPassages(data.nearest_passages, true));
      }
      break;
    case "passages":
      out.push(title(data.route === "citation_lookup" ? "Judgment found by citation" : "Passages only"));
      if (data.passages_note && data.route !== "citation_lookup") {
        out.push(el("p", { className: "note", text: capitalise(data.passages_note) + "." }));
      }
      out.push(el("p", { className: "state-body", text: data.route === "citation_lookup"
        ? "Its issue and headnote passages are listed under Sources."
        : "No written answer for this question; the passages are listed under Sources." }));
      if (!haveSources) out.push(renderPassages(data.passages, false));
      break;
    case "no_match_with_filters":
      out.push(title("No judgment matches the filters"));
      out.push(el("p", { text: "Filters: " + describeFilters(data.filters_applied || {}) + "." }));
      break;
    default:
      out.push(title("The answer could not be written"));
      out.push(el("p", { className: "error", text: data.passages_note || (data.notices || [])[0] || "The request failed." }));
  }
  out.push(renderTimings(data));
  return out;
}

function renderPending() {
  return el("div", { className: "pending" }, [
    el("p", { text: "The answer appears here once each of its quotes has been checked against the cited passage." }),
    ...[1, 2, 3, 4, 5].map(() => el("div", { className: "skeleton" })),
  ]);
}

// --- form ------------------------------------------------------------------------------

function readFilters() {
  const f = {};
  for (const key of ["year_from", "year_to"]) {
    const v = $(key).value.trim();
    if (v) f[key] = parseInt(v, 10);
  }
  for (const key of ["judge", "act", "outcome"]) {
    const v = $(key).value.trim();
    if (v) f[key] = v;
  }
  return Object.keys(f).length ? f : null;
}

function updateFilterCount() {
  const n = Object.keys(readFilters() || {}).length;
  $("filter-count").textContent = String(n);
  $("filter-count").hidden = n === 0;
}

function showError(message) {
  const p = $("form-error");
  p.textContent = message;
  p.hidden = !message;
}

function errorMessage(status, body) {
  if (body && Array.isArray(body.detail)) {
    return body.detail.map((d) => d.msg).join("; ");
  }
  if (body && typeof body.detail === "string") return body.detail;
  return "The server returned " + status + ".";
}

let progressTimer = null;

function setBusy(busy, label) {
  $("submit").disabled = busy;
  $("progress").hidden = !busy;
  clearInterval(progressTimer);
  if (busy) {
    const start = Date.now();
    const tick = () => {
      const s = Math.floor((Date.now() - start) / 1000);
      $("progress-text").textContent = label + (s ? " … " + s + " s" : "");
    };
    tick();
    progressTimer = setInterval(tick, 500);
  }
}

// POST a JSON body; returns { data } or { error }.
async function post(path, body, signal) {
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    let data = null;
    try { data = await r.json(); } catch (_) { data = null; }
    if (!r.ok || !data) return { error: errorMessage(r.status, data) };
    return { data };
  } catch (e) {
    if (e && e.name === "AbortError") return { aborted: true };
    return { error: "Could not reach the server." };
  }
}

let inflight = null;

async function onSubmit(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  const body = { question };
  const filters = readFilters();
  if (filters) body.filters = filters;

  if (inflight) inflight.abort();
  const controller = new AbortController();
  inflight = controller;
  const url = new URL(location.href);
  url.searchParams.set("q", question);
  history.replaceState(null, "", url);

  showError("");
  $("about").hidden = true;
  $("results").hidden = false;
  clear($("answer"));
  clear($("sources"));
  sources.clear();
  $("sources-meta").textContent = "";
  $("sources").append(el("div", { className: "pending" }, [1, 2, 3].map(() => el("div", { className: "skeleton" }))));
  $("answer").append(renderPending());

  try {
    // 1. the ranked passages, no LLM
    setBusy(true, "Searching passages");
    const s = await post("/search", body, controller.signal);
    if (s.aborted) return;
    if (s.data) {
      renderNotices(s.data.notices);
      renderSources(s.data);
    } else {
      clear($("sources"));
      $("sources").append(el("p", { className: "status-banner", text: s.error }));
    }
    // 2. the written answer, checked quote by quote
    setBusy(true, "Writing a checked answer");
    const a = await post("/ask", body, controller.signal);
    if (a.aborted) return;
    clear($("answer"));
    if (!a.data) {
      $("answer").append(title("The answer could not be written"), el("p", { className: "error", text: a.error }));
      return;
    }
    renderNotices(a.data.notices);
    const haveSources = sources.size > 0;
    $("answer").append(...renderResponse(a.data, haveSources).filter(Boolean));
    annotateSources(a.data);
  } finally {
    if (inflight === controller) {
      inflight = null;
      setBusy(false);
    }
  }
}

function clearFilters() {
  for (const key of ["year_from", "year_to", "judge", "act", "outcome"]) $(key).value = "";
  updateFilterCount();
}

// --- judgment view (/judgment/{path}?highlight=chunk_id) --------------------------------

async function showJudgment() {
  $("ask-view").hidden = true;
  $("judgment-view").hidden = false;
  const box = $("judgment");
  const r = await fetch(location.pathname + location.search, {
    headers: { accept: "application/json" },
  });
  let j = null;
  try { j = await r.json(); } catch (_) { j = null; }
  if (!r.ok || !j) {
    box.append(el("p", { className: "error", text: errorMessage(r.status, j) }));
    return;
  }
  document.title = j.title + " — Nyaya Search";
  box.append(
    el("h2", { text: j.title }),
    el("p", { className: "meta", text: [j.scr_citation, j.neutral_citation, formatDate(j.decision_date)]
      .filter(Boolean).join(" · ") }),
    el("p", { className: "meta", text: "Bench: " + j.judges.join(", ") +
      (j.author_judge ? " · Author: " + j.author_judge : "") +
      " · Disposal: " + j.disposal }),
  );
  if (!j.clean_text) {
    box.append(el("p", { className: "note", text: "No PDF text is available for this judgment." }));
    return;
  }
  const text = j.clean_text;
  const h = j.highlight;
  const body = el("div", { className: "judgment-text" });
  if (h && h.char_end > h.char_start && h.char_end <= text.length) {
    const mark = el("mark", { id: "highlight", text: text.slice(h.char_start, h.char_end) });
    body.append(text.slice(0, h.char_start), mark, text.slice(h.char_end));
    box.append(body);
    mark.scrollIntoView({ block: "center" });
  } else {
    body.textContent = text;
    box.append(body);
  }
}

// --- start -----------------------------------------------------------------------------

loadNotices();
if (location.pathname.startsWith("/judgment/")) {
  showJudgment();
} else {
  const form = $("ask-form");
  form.addEventListener("submit", onSubmit);
  $("question").addEventListener("keydown", (e) => {
    // Enter submits; Shift+Enter adds a line.
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  for (const id of ["year_from", "year_to", "judge", "act", "outcome"]) {
    $(id).addEventListener("input", updateFilterCount);
  }
  $("clear-filters").addEventListener("click", clearFilters);
  for (const b of document.querySelectorAll(".example")) {
    b.addEventListener("click", () => {
      $("question").value = b.dataset.q;
      form.requestSubmit();
    });
  }
  const q = new URLSearchParams(location.search).get("q");
  if (q) {
    $("question").value = q;
    form.requestSubmit();
  }
}
