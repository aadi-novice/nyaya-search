// Nyaya Search page. Renders the AskResponse JSON contract (contracts/schema.json).
// Plain DOM only: response text is always set with textContent, never parsed as HTML.
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, props, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "className") node.className = value;
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

// --- notices -------------------------------------------------------------------------

function renderNotices(notices) {
  const list = $("notice-list");
  clear(list);
  for (const n of notices || []) list.append(el("li", { text: n }));
}

async function loadNotices() {
  try {
    const r = await fetch("/health");
    if (r.ok) renderNotices((await r.json()).notices);
  } catch (_) {
    // The page still works; each answer carries its notices.
  }
}

// --- passage and citation details ------------------------------------------------------

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

// "Case v. State, [2025] 1 S.C.R. 230, para 13, p. 236" plus zone and opinion labels.
function refDetails(ref) {
  const parts = [ref.scr_citation];
  if (ref.neutral_citation) parts.push(ref.neutral_citation);
  if (ref.paras && ref.paras.length) parts.push(plural(ref.paras, "para", "paras"));
  if (ref.pages && ref.pages.length) parts.push(plural(ref.pages, "p.", "pp."));
  const labels = [ZONE_LABELS[ref.zone], OPINION_LABELS[ref.opinion_type]].filter(Boolean);
  return el("span", { className: "ref" }, [
    el("a", { href: judgmentLink(ref), className: "case-name", text: ref.case_name }),
    ", " + parts.join(", "),
    ...labels.map((l) => el("span", { className: "tag", text: l })),
  ]);
}

function renderCitation(c) {
  return el("li", { className: "citation" }, [
    el("span", { className: "label", text: c.passage_label }),
    refDetails(c),
    el("blockquote", { text: "“" + c.quote + "”" }),
  ]);
}

function renderPassages(passages, refusedNote) {
  return el("ol", { className: "passages" }, passages.map((p) =>
    el("li", { className: "passage" + (refusedNote ? " not-answer" : "") }, [
      refDetails(p),
      refusedNote ? el("span", { className: "tag warn", text: "Does not answer the question" }) : null,
    ])));
}

// --- answer parts ----------------------------------------------------------------------

function renderAnswer(sentences) {
  return el("ol", { className: "answer" }, sentences.map((s) =>
    el("li", { className: "sentence" }, [
      el("p", { text: s.sentence }),
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

// --- one response ----------------------------------------------------------------------

function renderResponse(data) {
  const out = [];
  switch (data.status) {
    case "answered":
      out.push(renderAnswer(data.answer));
      out.push(renderGaps(data.gaps));
      out.push(renderUnverified(data.unverified));
      break;
    case "refused":
      out.push(el("h2", { text: "No relevant judgment found" }));
      out.push(renderGaps(data.gaps));
      out.push(renderUnverified(data.unverified));
      if (data.nearest_passages.length) {
        out.push(el("h3", { text: "Nearest passages" }));
        out.push(renderPassages(data.nearest_passages, true));
      }
      break;
    case "passages":
      if (data.passages_note) out.push(el("p", { className: "note", text: data.passages_note }));
      out.push(renderPassages(data.passages, false));
      break;
    case "no_match_with_filters":
      out.push(el("h2", { text: "No judgment matches the filters" }));
      out.push(el("p", { text: "Filters: " + describeFilters(data.filters_applied || {}) + "." }));
      break;
    default:
      out.push(el("h2", { className: "error", text: "The request failed" }));
  }
  out.push(renderTimings(data));
  return out;
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

function setBusy(busy) {
  $("submit").disabled = busy;
  $("progress").hidden = !busy;
  clearInterval(progressTimer);
  if (busy) {
    const start = Date.now();
    $("progress-text").textContent = "Searching";
    progressTimer = setInterval(() => {
      const s = Math.floor((Date.now() - start) / 1000);
      $("progress-text").textContent = "Searching … " + s + " s";
    }, 500);
  }
}

async function onSubmit(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  const body = { question };
  const filters = readFilters();
  if (filters) body.filters = filters;

  showError("");
  clear($("result"));
  setBusy(true);
  try {
    const r = await fetch("/ask", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await r.json(); } catch (_) { data = null; }
    if (!r.ok || !data) {
      showError(errorMessage(r.status, data));
      return;
    }
    renderNotices(data.notices);
    $("result").append(...renderResponse(data).filter(Boolean));
  } catch (_) {
    showError("Could not reach the server.");
  } finally {
    setBusy(false);
  }
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
    el("p", { className: "meta", text: [j.scr_citation, j.neutral_citation, j.decision_date]
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
  $("ask-form").addEventListener("submit", onSubmit);
}
