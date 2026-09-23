// Animated supply map for mfa-engine (Streamlit custom component, v2 API).
//
// Everything is prepared in Python (ui/supply_map.py): projected country outlines, circle sizes,
// arc geometry and the cascade rounds. This file only draws and animates, with plain SVG and the
// Web Animations API, no libraries. Motion is decoration, never information: with
// prefers-reduced-motion the map is drawn in its final state and the particles stay off.

const NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

function htmlEl(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

const ICONS = {
  play: "M8 5.5v13l11-6.5z",
  pause: "M7 5h4v14H7zM13 5h4v14h-4z",
  reset: "M12 5V2L7 6l5 4V7a5 5 0 1 1-5 5H5a7 7 0 1 0 7-7z",
};

function button(parent, label, icon, cls) {
  const b = htmlEl("button", "sm-btn" + (cls ? " " + cls : ""), parent);
  b.type = "button";
  const s = svgEl("svg", { viewBox: "0 0 24 24", width: "16", height: "16", "aria-hidden": "true" }, b);
  const p = svgEl("path", { d: ICONS[icon], fill: "currentColor" }, s);
  htmlEl("span", null, b, label);
  b.setIcon = (name) => p.setAttribute("d", ICONS[name]);
  b.setLabel = (text) => { b.querySelector("span").textContent = text; };
  return b;
}

// Point on a quadratic Bezier arc at parameter t.
function bez(f, t) {
  const u = 1 - t;
  return [u * u * f.x1 + 2 * u * t * f.cx + t * t * f.x2, u * u * f.y1 + 2 * u * t * f.cy + t * t * f.y2];
}

export default function mount(component) {
  const { data, parentElement } = component;
  if (!data || !data.map) return;
  if (parentElement.__smCleanup) {
    try { parentElement.__smCleanup(); } catch (err) { /* previous render already gone */ }
  }

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const host = parentElement.querySelector(".sm-mount") || parentElement;
  const root = htmlEl("div", "sm-root", host);
  // Tell the page the map has drawn: the space held for it while loading (see ui/theme.py) is
  // released, and the element takes the map's own height, whatever the legend wraps to.
  const hostEl = parentElement.host || parentElement;
  const holder = hostEl.closest ? hostEl.closest('[data-testid="stElementContainer"]') : null;
  if (holder) holder.dataset.smReady = "1";
  root.style.setProperty("--acc", data.style.accent);
  root.style.setProperty("--acc-rgb", data.style.accentRgb);
  root.style.setProperty("--crit", data.style.critical);

  const timers = [];
  const later = (fn, ms) => { const id = setTimeout(fn, ms); timers.push(id); return id; };
  let raf = 0, running = false, flowsOn = !reduce, visible = true, last = 0;
  let pinned = null, played = false;

  // ------------------------------------------------------------------ toolbar
  const bar = htmlEl("div", "sm-bar", root);
  const playBtn = button(bar, "Play the disruption", "play", "primary");
  const flowBtn = button(bar, reduce ? "Show moving flows" : "Pause flows", reduce ? "play" : "pause");
  const resetBtn = button(bar, "Reset", "reset");
  resetBtn.hidden = true;
  const status = htmlEl("div", "sm-status", bar);
  status.setAttribute("aria-live", "polite");
  if (!data.cascade || !data.removed) {
    playBtn.disabled = true;
    playBtn.title = data.removed ? "This build of crm-trade-network does not report cascade rounds."
                                 : "The removed supplier has no position on the map.";
  }

  // ------------------------------------------------------------------ stage
  const W = data.map.width, H = data.map.height;
  const stage = htmlEl("div", "sm-stage", root);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": data.a11y }, stage);
  const gGrat = svgEl("g", { class: "grat" }, svg);
  const gLand = svgEl("g", { class: "land" }, svg);
  const gArcs = svgEl("g", { class: "arcs" }, svg);
  const gParticles = svgEl("g", { class: "particles" }, svg);
  const gBubbles = svgEl("g", { class: "bubbles" }, svg);
  const gFx = svgEl("g", { class: "fx" }, svg);
  const gLabels = svgEl("g", { class: "labels" }, svg);
  const tip = htmlEl("div", "sm-tip", stage);
  tip.setAttribute("role", "tooltip");

  data.map.graticule.forEach((d) => svgEl("path", { d }, gGrat));

  const landByCode = {};
  data.map.countries.forEach((c) => {
    const p = svgEl("path", { d: c.d }, gLand);
    c.codes.forEach((code) => { if (!landByCode[code]) landByCode[code] = p; });
    const code = c.codes.find((k) => data.info[k] || data.tint[k]);
    if (code) { p.dataset.a3 = code; p.classList.add("hot"); }
    const t = c.codes.map((k) => data.tint[k]).find((v) => v != null);
    if (t != null) p.style.fill = `rgba(${data.style.accentRgb},${t})`;
  });

  // ------------------------------------------------------------------ arcs + particles
  const arcs = data.flows.map((f, i) => {
    const el = svgEl("path", {
      d: `M${f.x1},${f.y1}Q${f.cx},${f.cy} ${f.x2},${f.y2}`, class: "arc", "stroke-width": f.w,
    }, gArcs);
    el.dataset.from = f.from;
    el.dataset.to = f.to;
    const len = el.getTotalLength();
    const g = svgEl("g", { class: "pgroup" }, gParticles);
    const n = Math.max(2, Math.min(7, Math.round(1.5 + f.w)));
    const size = 0.9 + f.w * 0.3;
    const dots = [];
    for (let j = 0; j < n; j++) {
      const halo = svgEl("circle", { r: size * 2.4, class: "p-halo" }, g);
      const t2 = svgEl("circle", { r: size * 0.55, class: "p-t2" }, g);
      const t1 = svgEl("circle", { r: size * 0.8, class: "p-t1" }, g);
      const head = svgEl("circle", { r: size, class: "p-head" }, g);
      dots.push({ phase: j / n, parts: [halo, t2, t1, head] });
    }
    // Speed follows arc length, so a long intercontinental flow takes longer to cross than a
    // short regional one; the bigger the flow, the more particles ride it.
    const period = 2.2 + len / 240;
    return { f, el, g, len, dots, period, i, cut: false };
  });

  // ------------------------------------------------------------------ circles + labels
  const bubbleEls = data.bubbles.map((b, i) => {
    const c = svgEl("circle", { cx: b.x, cy: b.y, r: Math.max(b.r, 2.2), class: "bubble", tabindex: "0" }, gBubbles);
    c.dataset.a3 = b.a3;
    c.setAttribute("aria-label", `${b.name}: ${b.value}${b.share ? ", " + b.share : ""}`);
    return { b, c, i };
  });
  // Labels for the largest producers. Each tries above its circle, then below, right and left,
  // and is left off rather than printed over another label; the tooltip still has the number.
  const placed = [];
  const clash = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  data.bubbles.slice(0, 6).forEach((b) => {
    const r = Math.max(b.r, 2.2);
    const t = svgEl("text", { class: "label" }, gLabels);
    t.textContent = b.name + " ";
    const v = svgEl("tspan", { class: "v" }, t);
    v.textContent = b.value;
    const spots = [[b.x, b.y - r - 5, "middle"], [b.x, b.y + r + 12, "middle"],
                   [b.x + r + 4, b.y + 4, "start"], [b.x - r - 4, b.y + 4, "end"]];
    for (const [x, y, anchor] of spots) {
      t.setAttribute("x", x);
      t.setAttribute("y", y);
      t.setAttribute("text-anchor", anchor);
      const bb = t.getBBox();
      const box = { x: bb.x - 2, y: bb.y - 1, w: bb.width + 4, h: bb.height + 2 };
      if (bb.x >= 0 && bb.x + bb.width <= W && bb.y >= 0 && !placed.some((p) => clash(p, box))) {
        placed.push(box);
        return;
      }
    }
    t.remove();
  });

  // ------------------------------------------------------------------ legend
  const legend = htmlEl("div", "sm-legend", root);
  const item = (svgMarkup, text) => {
    const it = htmlEl("div", "sm-li", legend);
    const s = svgEl("svg", { viewBox: "0 0 44 20", width: "44", height: "20", "aria-hidden": "true" }, it);
    s.innerHTML = svgMarkup;
    htmlEl("span", null, it, text);
  };
  item(`<circle cx="10" cy="10" r="4" class="bubble"/><circle cx="28" cy="10" r="8" class="bubble"/>`,
       `${data.text.measure} (${data.text.unit}); countries shaded by the same measure`);
  item(`<path d="M2 14Q22 0 42 14" class="arc" stroke-width="3"/><circle cx="30" cy="8" r="2.4" class="p-head"/>`,
       `${data.text.flows}; thicker is larger, dots move exporter to importer`);
  if (data.removed) {
    item(`<circle cx="22" cy="10" r="6" class="crit-dot"/>`, `${data.text.removedName}: the supplier removed in the disruption`);
  }

  // ------------------------------------------------------------------ focus + tooltip
  const infoText = (a3) => {
    const k = data.info[a3];
    if (!k) return null;
    const lines = [];
    if (k.mined) lines.push([data.text.measure, k.mined]);
    if (k.exports) lines.push(["Exports, 2023", k.exports]);
    if (k.imports) lines.push(["Imports, 2023", k.imports]);
    return { name: k.name, lines, buyers: k.buyers || [], hub: k.hub };
  };

  function showTip(a3, clientX, clientY) {
    const k = infoText(a3);
    if (!k) { tip.classList.remove("on"); return; }
    tip.replaceChildren();
    htmlEl("div", "t-name", tip, k.name);
    k.lines.forEach(([label, value]) => {
      const row = htmlEl("div", "t-row", tip);
      htmlEl("span", "t-k", row, label);
      htmlEl("span", "t-v", row, value);
    });
    if (k.buyers.length) {
      htmlEl("div", "t-k", tip, "Largest buyers");
      k.buyers.forEach((b) => htmlEl("div", "t-buyer", tip, b));
    }
    if (k.hub) htmlEl("div", "t-note", tip, "Re-export hub: part of this trade is transit, not origin.");
    const box = stage.getBoundingClientRect();
    tip.classList.add("on");
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = clientX - box.left + 14, y = clientY - box.top + 14;
    if (x + tw > box.width - 6) x = clientX - box.left - tw - 14;
    if (y + th > box.height - 6) y = Math.max(6, clientY - box.top - th - 14);
    tip.style.left = `${Math.max(6, x)}px`;
    tip.style.top = `${y}px`;
  }

  function focus(a3) {
    root.classList.toggle("focused", !!a3);
    arcs.forEach((a) => {
      const on = !a3 || a.f.from === a3 || a.f.to === a3;
      a.el.classList.toggle("dim", !on);
      a.g.classList.toggle("dim", !on);
    });
    bubbleEls.forEach(({ c, b }) => c.classList.toggle("dim", !!a3 && b.a3 !== a3));
  }

  const pointTarget = (ev) => ev.target.closest && ev.target.closest("[data-a3]");
  svg.addEventListener("pointermove", (ev) => {
    const t = pointTarget(ev);
    if (t) { showTip(t.dataset.a3, ev.clientX, ev.clientY); if (!pinned) focus(t.dataset.a3); }
    else { tip.classList.remove("on"); if (!pinned) focus(null); }
  });
  svg.addEventListener("pointerleave", () => { tip.classList.remove("on"); if (!pinned) focus(null); });
  svg.addEventListener("click", (ev) => {
    const t = pointTarget(ev);
    pinned = t && pinned !== t.dataset.a3 ? t.dataset.a3 : null;
    focus(pinned);
    status.textContent = pinned ? `Showing flows for ${(data.info[pinned] || {}).name || pinned}. Click it again to clear.` : "";
  });
  bubbleEls.forEach(({ c, b }) => {
    c.addEventListener("focus", () => { const r = c.getBoundingClientRect(); focus(b.a3); showTip(b.a3, r.right, r.top); });
    c.addEventListener("blur", () => { if (!pinned) focus(null); tip.classList.remove("on"); });
    c.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pinned = pinned === b.a3 ? null : b.a3; focus(pinned); }
      if (ev.key === "Escape") { pinned = null; focus(null); tip.classList.remove("on"); }
    });
  });

  // ------------------------------------------------------------------ particle loop
  function frame(now) {
    raf = 0;
    if (!running) return;
    last = now;
    const clock = now / 1000;
    for (const a of arcs) {
      if (a.cut) continue;
      for (const d of a.dots) {
        const t = (clock / a.period + d.phase) % 1;
        const pts = [t, Math.max(0, t - 0.05), Math.max(0, t - 0.025), t];
        const fade = t < 0.08 ? t / 0.08 : t > 0.92 ? (1 - t) / 0.08 : 1;
        d.parts.forEach((el, k) => {
          const [x, y] = bez(a.f, pts[k]);
          el.setAttribute("cx", x.toFixed(1));
          el.setAttribute("cy", y.toFixed(1));
          el.style.opacity = fade;
        });
      }
    }
    raf = requestAnimationFrame(frame);
  }

  function setRunning() {
    const want = flowsOn && visible && !document.hidden;
    gParticles.style.display = flowsOn ? "" : "none";
    if (want && !running) { running = true; last = 0; raf = requestAnimationFrame(frame); }
    if (!want && running) { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
  }

  flowBtn.addEventListener("click", () => {
    flowsOn = !flowsOn;
    flowBtn.setIcon(flowsOn ? "pause" : "play");
    flowBtn.setLabel(flowsOn ? "Pause flows" : "Show moving flows");
    setRunning();
  });
  const io = new IntersectionObserver((entries) => { visible = entries[0].isIntersecting; setRunning(); });
  io.observe(stage);
  const onVis = () => setRunning();
  document.addEventListener("visibilitychange", onVis);

  // ------------------------------------------------------------------ entrance
  const ease = "cubic-bezier(.22,.8,.26,1)";
  if (!reduce) {
    gLand.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 600, easing: "ease-out" });
    gGrat.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 900, easing: "ease-out" });
    bubbleEls.forEach(({ c, i }) => c.animate(
      [{ transform: "scale(0)", opacity: 0 }, { transform: "scale(1)", opacity: 1 }],
      { duration: 750, delay: 250 + i * 70, easing: "cubic-bezier(.34,1.56,.64,1)", fill: "backwards" }));
    arcs.forEach((a) => {
      a.el.style.strokeDasharray = `${a.len}`;
      const anim = a.el.animate([{ strokeDashoffset: a.len }, { strokeDashoffset: 0 }],
        { duration: 1100, delay: 500 + a.i * 40, easing: ease, fill: "backwards" });
      anim.onfinish = () => { a.el.style.strokeDasharray = ""; };
    });
    gParticles.style.opacity = 0;
    later(() => { gParticles.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 600, fill: "forwards" }); }, 900 + arcs.length * 40);
    later(() => { gParticles.style.opacity = 1; }, 1600 + arcs.length * 40);
  }
  setRunning();

  // ------------------------------------------------------------------ disruption replay
  function ripple(x, y, r0, delay) {
    if (reduce) return;
    const c = svgEl("circle", { cx: x, cy: y, r: r0, class: "ripple" }, gFx);
    const grow = (r0 * 3.2 + 14) / r0;
    const a = c.animate([{ transform: "scale(1)", opacity: 0.85 }, { transform: `scale(${grow})`, opacity: 0 }],
      { duration: 1300, delay: delay || 0, easing: "cubic-bezier(.2,.6,.3,1)", fill: "both" });
    a.onfinish = () => c.remove();
  }

  function cutArcsTouching(a3) {
    arcs.forEach((a) => {
      if (!a.cut && (a.f.from === a3 || a.f.to === a3)) {
        a.cut = true;
        a.el.classList.add("cut");
        a.g.classList.add("cut");
      }
    });
  }

  function play() {
    if (played || !data.cascade || !data.removed) return;
    played = true;
    pinned = null;
    focus(null);
    playBtn.disabled = true;
    resetBtn.hidden = false;
    const R = data.removed;
    const step = reduce ? 450 : 1150;
    let down = 0;
    status.textContent = `${R.name} stops exporting.`;
    const src = bubbleEls.find(({ b }) => b.a3 === R.a3);
    if (src) src.c.classList.add("down");
    if (landByCode[R.a3]) landByCode[R.a3].classList.add("down");
    for (let k = 0; k < 3; k++) ripple(R.x, R.y, 6, k * 260);
    arcs.forEach((a) => {
      if (a.f.from !== R.a3) return;
      a.el.classList.add("severed");
      a.g.classList.add("cut");
      a.cut = true;
      if (!reduce) {
        a.el.style.strokeDasharray = "5 7";
        a.el.animate([{ strokeDashoffset: 0, opacity: 0.95 }, { strokeDashoffset: -60, opacity: 0.22 }],
          { duration: 1400, easing: "ease-in", fill: "forwards" });
      }
    });
    data.cascade.rounds.forEach((round, r) => {
      later(() => {
        round.countries.forEach((c, j) => {
          if (landByCode[c.a3]) landByCode[c.a3].classList.add("down");
          const bub = bubbleEls.find(({ b }) => b.a3 === c.a3);
          if (bub) bub.c.classList.add("down");
          ripple(c.x, c.y, 3, Math.min(j * 12, 500));
          cutArcsTouching(c.a3);
        });
        down += round.count;
        status.textContent = `Round ${r + 1}: ${round.count} more ${round.count === 1 ? "country" : "countries"} lose over `
          + `20% of their trade and fail (${down} of ${data.cascade.universe - 1} so far).`;
      }, step * (r + 1));
    });
    later(() => {
      const off = data.cascade.offmap ? ` ${data.cascade.offmap} of them have no position on this map.` : "";
      status.textContent = `Cascade over: losing ${R.name} takes down ${down} of the ${data.cascade.universe - 1} `
        + `other countries in this trade network within ${data.cascade.rounds.length + 1} rounds.${off}`;
    }, step * (data.cascade.rounds.length + 1) + 200);
  }

  function reset() {
    cleanup();
    parentElement.__smCleanup = null;
    mount(component);
  }

  playBtn.addEventListener("click", play);
  resetBtn.addEventListener("click", reset);

  // Labels only where they can be read: below ~560 px wide they would be a few pixels tall.
  const ro = new ResizeObserver(() => root.classList.toggle("compact", stage.clientWidth < 560));
  ro.observe(stage);

  function cleanup() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    timers.forEach(clearTimeout);
    io.disconnect();
    ro.disconnect();
    document.removeEventListener("visibilitychange", onVis);
    root.remove();
  }
  parentElement.__smCleanup = cleanup;
  return cleanup;
}
