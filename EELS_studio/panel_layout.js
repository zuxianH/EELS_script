// Resize panels in place; send one settings update after each completed gesture.
export default function ({ data, setTriggerValue, parentElement }) {
    // V2 calls the renderer again for new data, but only invokes its returned
    // cleanup on unmount. Retire the previous observers before installing new ones.
    const cleanupKey = Symbol.for("eels.panel-layout.cleanup");
    parentElement[cleanupKey]?.();
    const settings = { ...data };
    const specs = {
        detector: { panel: ".st-key-detector_panel", plot: ".st-key-detector_click_target", height: "panel_detector_height" },
        spectrum: { panel: ".st-key-spectrum_panel", plot: ".st-key-spectrum_axis_target", height: "panel_spectrum_height" },
    };
    const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
    const created = new Set();
    const titles = new Set();
    const modifiedColumns = new Set();
    const observed = new WeakSet();
    let frame = null, drag = null, disposed = false;
    const style = document.createElement("style");
    style.textContent = `
        .eels-panel-divider {position:absolute; top:0; bottom:0; width:16px; z-index:3;
            transform:translateX(-50%); cursor:col-resize; touch-action:none; border:0; background:transparent; padding:0;}
        .eels-panel-divider::after {content:""; position:absolute; top:20px; bottom:20px; left:6px;
            width:4px; border-radius:4px; background:#b7cbd2;}
        .eels-height-handle {display:block; height:18px; width:100%; flex-shrink:0;
            cursor:row-resize; touch-action:none; border:0; padding:0; background:transparent;}
        .eels-height-handle::after {content:""; display:block; width:60px; height:4px;
            border-radius:4px; margin:auto; background:#b7cbd2;}
        .eels-panel-divider:hover::after, .eels-panel-divider:focus-visible::after,
        .eels-height-handle:hover::after, .eels-height-handle:focus-visible::after {background:#137c78;}
        .eels-movable-title {cursor:grab; user-select:none;}
        .eels-movable-title::before {content:"⠿"; padding-right:8px; color:#137c78;}
        .eels-drop-target {outline:2px dashed #137c78; outline-offset:2px;}
    `;
    document.head.appendChild(style);
    const notify = () => setTriggerValue("layout", {
        panel_detector_width: settings.panel_detector_width,
        panel_spectrum_height: settings.panel_spectrum_height,
        panel_detector_height: settings.panel_detector_height,
        panel_detector_first: settings.panel_detector_first,
    });
    const panels = () => Object.fromEntries(Object.entries(specs).map(([name, spec]) => [name, document.querySelector(spec.panel)]));
    const split = () => {
        const p = panels();
        const detector = p.detector?.closest('[data-testid="stColumn"]');
        const spectrum = p.spectrum?.closest('[data-testid="stColumn"]');
        if (!detector || !spectrum || detector.parentElement !== spectrum.parentElement) return null;
        return { detector, spectrum, row: detector.parentElement };
    };
    const schedule = () => { if (!disposed && frame === null) frame = requestAnimationFrame(apply); };
    const resizeObserver = new ResizeObserver(schedule);
    const watch = (node) => {
        if (node && !observed.has(node)) { resizeObserver.observe(node); observed.add(node); }
    };
    const button = (className, label, orientation) => {
        const node = document.createElement("button");
        node.type = "button";
        node.className = className;
        node.title = label;
        node.setAttribute("role", "separator");
        node.setAttribute("aria-label", label);
        node.setAttribute("aria-orientation", orientation);
        created.add(node);
        return node;
    };
    const start = (event, kind, name) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        const layout = split();
        drag = { kind, name, y: event.clientY, initial: settings[specs[name]?.height],
                 bounds: layout?.row.getBoundingClientRect() };
        event.currentTarget.setPointerCapture(event.pointerId);
    };
    const resizeChart = (anchor, height) => {
        const chart = anchor.querySelector(".js-plotly-plot");
        const host = chart?.closest('[data-testid="stPlotlyChart"]');
        if (!chart?._fullLayout || !host || !window.Plotly) return;
        const width = Math.floor(anchor.getBoundingClientRect().width);
        if (width < 50 || anchor.getClientRects().length === 0) return;
        host.style.height = `${height}px`;
        const wrapper = host.closest('[data-testid="stElementContainer"]');
        if (wrapper) wrapper.style.height = `${height}px`;
        if (Math.abs(chart._fullLayout.width - width) > 1 || Math.abs(chart._fullLayout.height - height) > 1) {
            // Explicit sizes keep Plotly's canvases and pointer coordinates aligned.
            Promise.resolve(window.Plotly.relayout(chart, { width, height })).catch(() => {});
        }
    };
    const apply = () => {
        frame = null;
        const p = panels();
        const layout = split();
        if (layout) {
            const { detector, spectrum, row } = layout;
            watch(row);
            row.style.position = "relative";
            const stacked = row.getBoundingClientRect().width < 720;
            const gap = parseFloat(getComputedStyle(row).columnGap) || 16;
            const available = row.getBoundingClientRect().width - gap;
            const detectorWidth = clamp(available * settings.panel_detector_width, 260, available - 340);
            for (const [col, first, width] of [[detector, settings.panel_detector_first, detectorWidth],
                                              [spectrum, !settings.panel_detector_first, available - detectorWidth]]) {
                modifiedColumns.add(col);
                col.style.order = first ? "0" : "1";
                col.style.minWidth = "0";
                col.style.flex = stacked ? "0 0 100%" : `0 0 ${width}px`;
                col.style.width = stacked ? "100%" : `${width}px`;
            }
            row.style.flexWrap = stacked ? "wrap" : "nowrap";
            let handle = row.querySelector(":scope > .eels-panel-divider");
            if (!handle) {
                handle = button("eels-panel-divider", "Resize Detector and EELS Spectrum panels", "vertical");
                handle.addEventListener("pointerdown", (e) => start(e, "width"));
                handle.addEventListener("keydown", (e) => {
                    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
                    e.preventDefault();
                    const direction = (e.key === "ArrowRight" ? 1 : -1) * (settings.panel_detector_first ? 1 : -1);
                    settings.panel_detector_width = clamp(settings.panel_detector_width + direction * .03, .2, .7);
                    apply(); notify();
                });
                row.appendChild(handle);
            }
            handle.hidden = stacked;
            handle.style.left = `${(settings.panel_detector_first ? detectorWidth : available - detectorWidth) + gap / 2}px`;
            handle.setAttribute("aria-valuemin", "20");
            handle.setAttribute("aria-valuemax", "70");
            handle.setAttribute("aria-valuenow", String(Math.round(settings.panel_detector_width * 100)));
        }
        for (const [name, panel] of Object.entries(p)) {
            if (!panel) continue;
            panel.style.minHeight = "0px";
            watch(panel);
            const spec = specs[name];
            const anchor = panel.querySelector(spec.plot);
            if (anchor) {
                let handle = anchor.querySelector(":scope > .eels-height-handle");
                if (!handle) {
                    handle = button("eels-height-handle", `Resize ${name === "detector" ? "Detector" : "EELS Spectrum"} height`, "horizontal");
                    handle.addEventListener("pointerdown", (e) => start(e, "height", name));
                    handle.addEventListener("keydown", (e) => {
                        if (!["ArrowUp", "ArrowDown"].includes(e.key)) return;
                        e.preventDefault();
                        settings[spec.height] = clamp(settings[spec.height] + (e.key === "ArrowDown" ? 24 : -24), 260, 1000);
                        apply(); notify();
                    });
                    anchor.appendChild(handle);
                }
                handle.dataset.renderRevision = String(data.render_revision);
                handle.setAttribute("aria-valuemin", "260");
                handle.setAttribute("aria-valuemax", "1000");
                handle.setAttribute("aria-valuenow", String(settings[spec.height]));
                resizeChart(anchor, settings[spec.height]);
            }
            const title = panel.querySelector(".eels-panel-title");
            if (title && !titles.has(title) && p.detector && p.spectrum) {
                titles.add(title);
                title.draggable = true;
                title.tabIndex = 0;
                title.classList.add("eels-movable-title");
                title.title = "Drag onto the other panel to swap positions, or press Enter";
                title.ondragstart = (e) => { e.dataTransfer.setData("text/eels-panel", name); e.dataTransfer.effectAllowed = "move"; };
                title.onkeydown = (e) => {
                    if (!["Enter", " "].includes(e.key)) return;
                    e.preventDefault(); settings.panel_detector_first = !settings.panel_detector_first; apply(); notify();
                };
                panel.ondragover = (e) => {
                    if (!e.dataTransfer.types.includes("text/eels-panel")) return;
                    e.preventDefault(); panel.classList.add("eels-drop-target");
                };
                panel.ondragleave = () => panel.classList.remove("eels-drop-target");
                panel.ondrop = (e) => {
                    const source = e.dataTransfer.getData("text/eels-panel");
                    if (!source) return;
                    e.preventDefault(); panel.classList.remove("eels-drop-target");
                    if (source !== name) { settings.panel_detector_first = !settings.panel_detector_first; apply(); notify(); }
                };
            }
        }
    };
    const move = (e) => {
        if (!drag) return;
        e.preventDefault();
        if (drag.kind === "height") {
            settings[specs[drag.name].height] = Math.round(clamp(drag.initial + e.clientY - drag.y, 260, 1000));
        } else if (drag.bounds) {
            const gap = parseFloat(getComputedStyle(split().row).columnGap) || 16;
            const fraction = (e.clientX - drag.bounds.left - gap / 2) / (drag.bounds.width - gap);
            settings.panel_detector_width = clamp(settings.panel_detector_first ? fraction : 1 - fraction, .2, .7);
        }
        schedule();
    };
    const finish = () => {
        if (!drag) return;
        drag = null;
        apply();
        notify();
    };
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener("pointermove", move, { passive: false });
    document.addEventListener("pointerup", finish);
    document.addEventListener("pointercancel", finish);
    window.addEventListener("resize", schedule);
    schedule();
    const cleanup = () => {
        if (disposed) return;
        disposed = true;
        observer.disconnect(); resizeObserver.disconnect();
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", finish);
        document.removeEventListener("pointercancel", finish);
        window.removeEventListener("resize", schedule);
        if (frame !== null) cancelAnimationFrame(frame);
        created.forEach((node) => node.remove());
        titles.forEach((title) => {
            title.draggable = false; title.removeAttribute("tabindex"); title.removeAttribute("title");
            title.classList.remove("eels-movable-title"); title.ondragstart = title.onkeydown = null;
        });
        Object.values(panels()).filter(Boolean).forEach((p) => {
            p.ondragover = p.ondragleave = p.ondrop = null;
            p.classList.remove("eels-drop-target");
        });
        modifiedColumns.forEach((col) => {
            for (const prop of ["order", "min-width", "flex", "width"]) col.style.removeProperty(prop);
        });
        const layout = split();
        if (layout) { layout.row.style.removeProperty("position"); layout.row.style.removeProperty("flex-wrap"); }
        style.remove();
    };
    parentElement[cleanupKey] = cleanup;
    return () => {
        cleanup();
        if (parentElement[cleanupKey] === cleanup) delete parentElement[cleanupKey];
    };
}
