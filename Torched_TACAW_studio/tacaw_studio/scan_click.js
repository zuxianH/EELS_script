export default function ({ data, setTriggerValue }) {
    let start = null;
    const target = () => document.querySelector(".st-key-trajectory_plot .js-plotly-plot");
    const down = event => {
        const chart = target();
        if (!data.enabled || event.button !== 0 || !chart?.contains(event.target)) return;
        if (!event.target.closest(".nsewdrag, .draglayer, .scatterlayer")) return;
        start = { x: event.clientX, y: event.clientY, chart };
    };
    const up = event => {
        const saved = start;
        start = null;
        if (!saved || !data.enabled || event.button !== 0) return;
        if (Math.hypot(event.clientX - saved.x, event.clientY - saved.y) > 4) return;
        const chart = saved.chart, layout = chart._fullLayout;
        if (!layout?.xaxis || !layout?.yaxis) return;
        const box = chart.getBoundingClientRect();
        const px = event.clientX - box.left - layout.xaxis._offset;
        const py = event.clientY - box.top - layout.yaxis._offset;
        if (px < 0 || py < 0 || px > layout.xaxis._length || py > layout.yaxis._length) return;
        const x = layout.xaxis.p2l(px), y = layout.yaxis.p2l(py);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        setTriggerValue("clicked", { x, y, preview_id: data.preview_id,
                                   role: data.role, event_id: crypto.randomUUID() });
    };
    document.addEventListener("pointerdown", down, true);
    document.addEventListener("pointerup", up, true);
    return () => {
        document.removeEventListener("pointerdown", down, true);
        document.removeEventListener("pointerup", up, true);
    };
}
