/* Intellex presentation UI — every question is independent. */
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const rebuildBtn = document.getElementById("rebuild");
const statusEl = document.getElementById("status");

let busy = false;

function typesetMath(container = document) {
  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    window.MathJax.typesetPromise([container]).catch(() => {});
  }
}


function escapeHtml(s = "") {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatText(text = "") {
  let s = escapeHtml(text);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  return s.replace(/\n/g, "<br>");
}

function extractDomain(url = "") {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch { return "web source"; }
}

function extractDate(snippet = "") {
  const patterns = [
    /\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b/i,
    /\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b/i,
    /\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b/i
  ];
  for (const p of patterns) { const m = String(snippet).match(p); if (m) return m[0]; }
  return "";
}

function cleanSnippet(snippet = "") {
  return String(snippet)
    .replace(/^\s*(?:\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4})\s*[-–—:]\s*/i, "")
    .replace(/\s+/g, " ").trim();
}

function sourceCards(items, kind, query) {
  if (!Array.isArray(items) || !items.length) return "";
  const icon = kind === "web" ? "🌐" : "📚";
  const title = kind === "web" ? "Web sources" : "Database sources";
  let html = `<details class="source-block"><summary class="source-toggle"><span>${icon} ${title} <b>${items.length}</b></span><span class="chevron">⌄</span></summary><div class="source-list">`;
  items.slice(0, 5).forEach((item, i) => {
    const raw = item.snippet || item.evidence || item.text || "";
    const snippet = cleanSnippet(raw).slice(0, 420);
    const date = extractDate(raw);
    const domain = item.url ? extractDomain(item.url) : "Database document";
    const titleText = item.title || item.file || "Source";
    const href = item.url ? ` href="${escapeHtml(item.url)}" target="_blank" rel="noopener"` : "";
    html += `<article class="source-card">
      <div class="source-top">
        <span class="num">${String(i+1).padStart(2,"0")}</span>
        <div class="source-main">
          ${item.url ? `<a${href} class="source-title">${escapeHtml(titleText)}</a>` : `<strong class="source-title">${escapeHtml(titleText)}</strong>`}
          <div class="source-meta"><span>${escapeHtml(domain)}</span>${date ? `<span class="date">${escapeHtml(date)}</span>` : ""}${item.page ? `<span>Page ${escapeHtml(String(item.page))}</span>` : ""}${item.score ? `<span class="match">${Math.round(item.score*100)}% match</span>` : ""}</div>
        </div>
      </div>
      ${snippet ? `<p>${escapeHtml(snippet)}</p>` : ""}
      ${item.url ? `<a class="open" href="${escapeHtml(item.url)}" target="_blank" rel="noopener">Open source ↗</a>` : ""}
    </article>`;
  });
  html += `</div></details>`;
  return html;
}

function answerHtml(data) {
  const source = data.source || "";
  let badge = "";
  if (data.case === 3 || source === "aerocalc") badge = `<span class="badge aero">🧮 AeroCalc</span>`;
  else if (data.case === 2 || source === "database") badge = `<span class="badge db">📚 Answered from your database</span>`;
  else if (source === "web_unavailable") badge = `<span class="badge warn">⚠️ Web search unavailable</span>`;
  else if (data.case === 1 || source === "web") badge = `<span class="badge web">🌐 Answered from the web</span>`;

  let html = `<div class="answer-card">${badge}<div class="answer-text">${formatText(data.answer || "")}</div>`;
  if (data.aerocalc) {
    html += `<div class="aero-panel"><div class="aero-title">🧮 AeroCalc</div>`;
    if (data.aerocalc.error) html += `<div class="error">${escapeHtml(data.aerocalc.error)}</div>`;
    (data.aerocalc.groups || []).forEach(g => {
      html += `<div class="aero-group"><div class="aero-group-title">${escapeHtml(g.title || "")}</div>`;
      (g.rows || []).forEach(r => html += `<div class="aero-row"><span>${escapeHtml(r.label || "")}</span><strong>${escapeHtml(String(r.display || ""))}${r.unit ? " " + escapeHtml(r.unit) : ""}</strong></div>`);
      html += `</div>`;
    });
    html += `</div>`;
  }
  if (source === "database" || source === "database+web") html += sourceCards(data.db_results, "db", data.message);
  if (source === "web" || source === "database+web") html += sourceCards(data.web_results, "web", data.message);
  html += `</div>`;
  return html;
}

function addUser(text) {
  const d = document.createElement("div");
  d.className = "msg user";
  d.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
  chatEl.appendChild(d);
}

function addBot(html) {
  const d = document.createElement("div");
  d.className = "msg bot";
  d.innerHTML = `<div class="bot-avatar">🤖</div><div class="bot-wrap">${html}</div>`;
  chatEl.appendChild(d);
  typesetMath(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

function addTyping() {
  return addBot(`<div class="answer-card typing-card"><span></span><span></span><span></span></div>`);
}

async function send(message) {
  if (busy) return;
  busy = true; sendBtn.disabled = true;
  addUser(message);
  const typing = addTyping();
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({message})
    });
    const data = await r.json();
    typing.remove();
    addBot(answerHtml(data));
  } catch (e) {
    typing.remove();
    addBot(`<div class="answer-card"><div class="error">Could not reach Intellex. Check that the server is running.</div></div>`);
  } finally {
    busy = false; sendBtn.disabled = false; inputEl.focus();
  }
}

formEl.addEventListener("submit", e => {
  e.preventDefault();
  const msg = inputEl.value.trim();
  if (!msg) return;
  inputEl.value = ""; send(msg);
});

document.querySelectorAll(".quick button").forEach(b => b.addEventListener("click", () => send(b.dataset.q)));

document.addEventListener("toggle", (event) => {
  const block = event.target.closest?.(".source-block");
  if (!block) return;
  const chevron = block.querySelector(".chevron");
  if (chevron) chevron.textContent = block.open ? "⌃" : "⌄";
  if (block.open) typesetMath(block);
}, true);


rebuildBtn.addEventListener("click", async () => {
  rebuildBtn.disabled = true;
  rebuildBtn.innerHTML = "Rebuilding…";
  try {
    const r = await fetch("/api/rebuild", {method:"POST"});
    const data = await r.json();
    addBot(`<div class="answer-card"><span class="badge db">✓ Knowledge base updated</span><div class="answer-text">Index rebuilt successfully.</div></div>`);
    fetchHealth();
  } catch {
    addBot(`<div class="answer-card"><div class="error">Could not rebuild the knowledge base.</div></div>`);
  } finally {
    rebuildBtn.disabled = false; rebuildBtn.innerHTML = "↻ <span>Rebuild index</span>";
  }
});

async function fetchHealth() {
  try {
    const r = await fetch("/api/health"); const h = await r.json();
    statusEl.innerHTML = `<i></i> Knowledge base ready · ${escapeHtml(String(h.docs_loaded || 0))} chunks`;
    statusEl.className = "status ok";
  } catch {
    statusEl.innerHTML = `<i></i> Backend offline`; statusEl.className = "status warn";
  }
}
fetchHealth(); inputEl.focus();
