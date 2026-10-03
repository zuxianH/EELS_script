// Scoped pointer dragging: no global listeners, observers, or drag-time reruns.
export default function ({ data, parentElement, setTriggerValue }) {
    const items = data.items;
    const before = items.map(item => item.path);
    let dragging = null;
    let pending = false;
    const style = document.createElement('style');
    style.textContent = `
        .file-order {list-style:none; margin:0; padding:2px; max-height:280px; overflow:auto;
            font:14px system-ui,sans-serif; color:#172b3a;}
        .file-order li {display:flex; align-items:center; gap:8px; box-sizing:border-box;
            min-height:42px; margin-bottom:6px; padding:8px; background:white;
            border:1px solid #dce5ec; border-radius:7px; cursor:grab; user-select:none; touch-action:none;}
        .file-order li:focus-visible {outline:2px solid #137c78; outline-offset:-2px;}
        .file-order li[data-drop=before] {box-shadow:inset 0 3px #137c78;}
        .file-order li[data-drop=after] {box-shadow:inset 0 -3px #137c78;}
        .file-order .grip {color:#137c78; flex-shrink:0;}
        .file-order .filename {overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
    `;
    const list = document.createElement('ol');
    list.className = 'file-order';
    list.dataset.testid = 'file-order-list';
    list.setAttribute('aria-label', 'Comparison file order');
    parentElement.replaceChildren(style, list);
    const clearTargets = () => list.querySelectorAll('[data-drop]').forEach(row => delete row.dataset.drop);
    const commit = paths => {
        if (paths.every((path, i) => path === before[i])) return;
        pending = true;
        render(paths);
        setTriggerValue('order', {before, paths});
    };
    const render = paths => {
        list.replaceChildren();
        paths.forEach((path, index) => {
            const item = items.find(item => item.path === path);
            const row = document.createElement('li');
            row.dataset.path = path;
            row.draggable = false;
            row.tabIndex = 0;
            row.title = item.label;
            row.setAttribute('aria-label', `${index + 1}. ${item.label}`);
            const grip = document.createElement('span');
            grip.className = 'grip';
            grip.textContent = '⠿';
            grip.setAttribute('aria-hidden', 'true');
            const name = document.createElement('span');
            name.className = 'filename';
            name.textContent = `${index + 1}. ${item.label}`;
            row.append(grip, name);
            row.onpointerdown = event => {
                if (pending || event.button !== 0) return;
                event.preventDefault();
                row.focus({preventScroll: true});
                dragging = {path, y: event.clientY, moved: false, ordered: before};
                row.setPointerCapture(event.pointerId);
            };
            row.onpointermove = event => {
                if (!dragging) return;
                if (!dragging.moved && Math.abs(event.clientY - dragging.y) < 4) return;
                dragging.moved = true;
                const bounds = list.getBoundingClientRect();
                if (event.clientY < bounds.top + 20) list.scrollTop -= 12;
                if (event.clientY > bounds.bottom - 20) list.scrollTop += 12;
                const others = [...list.children].filter(row => row.dataset.path !== dragging.path);
                let insert = others.findIndex(row => {
                    const box = row.getBoundingClientRect();
                    return event.clientY < box.top + box.height / 2;
                });
                if (insert < 0) insert = others.length;
                const ordered = others.map(row => row.dataset.path);
                ordered.splice(insert, 0, dragging.path);
                dragging.ordered = ordered;
                clearTargets();
                if (insert < others.length) others[insert].dataset.drop = 'before';
                else if (others.length) others[others.length - 1].dataset.drop = 'after';
            };
            row.onpointerup = event => {
                if (!dragging) return;
                const finished = dragging;
                dragging = null;
                row.releasePointerCapture(event.pointerId);
                clearTargets();
                const bounds = list.getBoundingClientRect();
                if (finished.moved && event.clientX >= bounds.left && event.clientX <= bounds.right
                        && event.clientY >= bounds.top && event.clientY <= bounds.bottom) {
                    commit(finished.ordered);
                }
            };
            row.onpointercancel = row.onlostpointercapture = () => { dragging = null; clearTargets(); };
            row.onkeydown = event => {
                if (pending || !event.altKey || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
                event.preventDefault();
                const next = index + (event.key === 'ArrowUp' ? -1 : 1);
                if (next < 0 || next >= before.length) return;
                const ordered = [...before];
                [ordered[index], ordered[next]] = [ordered[next], ordered[index]];
                commit(ordered);
                [...list.children].find(row => row.dataset.path === path)?.focus();
            };
            list.appendChild(row);
        });
    };
    render(before);
}
