export default function ({ data, setTriggerValue }) {
    const selector = '.st-key-background_preview_plot .js-plotly-plot';
    let press = null;
    const pointInPlot = (event, chart) => {
        const layout = chart?._fullLayout;
        if (!layout?.xaxis || !layout?.yaxis) return null;
        // Use the plotting area's coordinates, including after pan/zoom.
        const rect = chart.getBoundingClientRect();
        const x = event.clientX - rect.left - layout.xaxis._offset;
        const y = event.clientY - rect.top - layout.yaxis._offset;
        if (x < 0 || x > layout.xaxis._length || y < 0 || y > layout.yaxis._length) return null;
        return { energy: layout.xaxis.p2d(x), layout };
    };
    const down = (event) => {
        press = null;
        if (!data?.target || event.button !== 0 || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
        const chart = event.target.closest?.(selector);
        // Ignore legend, modebar and axis handles, even when they overlap the plot.
        if (!chart || !event.target.closest?.('.nsewdrag') || !pointInPlot(event, chart)) return;
        press = { chart, x: event.clientX, y: event.clientY, id: event.pointerId };
    };
    const up = (event) => {
        const start = press;
        press = null;
        if (!start || start.id !== event.pointerId || !start.chart.isConnected) return;
        if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > 5) return;
        const point = pointInPlot(event, start.chart);
        if (!point || !Number.isFinite(point.energy)) return;
        // Keep the current viewport when changing the draft fit bounds reruns Python.
        start.chart.emit?.('plotly_relayout', {
            'xaxis.range': [...point.layout.xaxis.range],
            'yaxis.range': [...point.layout.yaxis.range],
        });
        setTriggerValue('clicked', {
            energy: point.energy, target: data.target, context: data.context,
            event_id: crypto.randomUUID(),
        });
    };
    const cancel = () => { press = null; };
    document.addEventListener('pointerdown', down, true);
    document.addEventListener('pointerup', up, true);
    document.addEventListener('pointercancel', cancel, true);
    window.addEventListener('blur', cancel);
    return () => {
        document.removeEventListener('pointerdown', down, true);
        document.removeEventListener('pointerup', up, true);
        document.removeEventListener('pointercancel', cancel, true);
        window.removeEventListener('blur', cancel);
    };
}
