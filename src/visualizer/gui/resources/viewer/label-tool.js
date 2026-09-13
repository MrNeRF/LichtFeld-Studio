/*
 * Label tool for the self-contained HTML viewer export.
 *
 * Pick points on the model, name them, and save/load the resulting
 * annotation set as a simple .lbl file (JSON). Placement reuses the
 * viewer's depth-based `Picker` (same as the measure tool); once a label
 * is selected a `TranslateGizmo` is attached so a mis-picked point can be
 * nudged precisely before it is saved.
 *
 * IMPORTANT: like gizmo.js and measure-tool.js, this file is NOT
 * self-contained at runtime. It is concatenated (see
 * src/io/formats/html.cpp) after index.js, gizmo.js and measure-tool.js
 * and wrapped in an IIFE at HTML export time, so it shares index.js's
 * bundled PlayCanvas engine classes (Vec3, Entity, Picker, ...) and
 * gizmo.js's `Gizmo`/`TranslateGizmo` by closure rather than a real ES
 * import. Do not add real `import` statements here; the trailing
 * `export` below is stripped at export time.
 *
 * Interactions (label mode must be toggled on via the #labels button):
 *   - click empty model space -> pick a point, prompt for text, add label
 *   - click an existing label -> select it (gizmo attached, drag to move)
 *   - double-click a label    -> rename (empty text deletes)
 *   - right-click a label     -> delete
 * Save/load work regardless of mode via #labelSave / #labelLoad.
 */

// Screen-space offset (px) of the text anchor from the label point; the
// text is drawn with text-anchor: end so it extends to the left of it.
const LABEL_TEXT_DX = -14;
const LABEL_TEXT_DY = -12;
// Pointer must move less than this (px) between down and up for a click to
// register as a pick; matches the viewer's own double-tap tolerance.
const LABEL_CLICK_DEADZONE = 8;
// Screen-space radius (px) around a label's point that counts as a click
// on that label.
const LABEL_PICK_TOLERANCE = 12;
// Rough clickable width (px) of the label text to the left of its anchor.
const LABEL_TEXT_WIDTH = 120;

function initLabelTool(global) {
    const { app, camera, events } = global;
    const canvas = app.graphicsDevice.canvas;

    // ---- state -------------------------------------------------------
    const labels = [];   // { text: string, position: Vec3 }
    const groups = [];   // SVG <g> per label, parallel to `labels`
    let active = false;
    let selection = -1;
    let lastPlacedTime = 0;
    let gizmo = null;
    let gizmoLayer = null;
    let pivot = null;
    let picker = null;

    const _screen = new Vec3();

    // ---- SVG overlay (leader line + point + text per label) -----------
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('id', 'labelToolSvg');
    svg.classList.add('hidden');
    document.getElementById('ui').appendChild(svg);

    const createLabelGroup = (text) => {
        const g = document.createElementNS(svgNS, 'g');

        const lineBottom = document.createElementNS(svgNS, 'line');
        lineBottom.setAttribute('class', 'leaderBottom');
        const lineTop = document.createElementNS(svgNS, 'line');
        lineTop.setAttribute('class', 'leaderTop');
        const point = document.createElementNS(svgNS, 'circle');
        point.setAttribute('class', 'labelPoint');
        point.setAttribute('r', '4');
        const textEl = document.createElementNS(svgNS, 'text');
        textEl.setAttribute('class', 'labelText');
        textEl.textContent = text;

        g.appendChild(lineBottom);
        g.appendChild(lineTop);
        g.appendChild(point);
        g.appendChild(textEl);
        svg.appendChild(g);
        return g;
    };

    // ---- gizmo (lazily created on first selection) ---------------------
    const ensureGizmo = () => {
        if (gizmo) return;
        gizmoLayer = Gizmo.createLayer(app, 'LfsLabelGizmo');
        gizmo = new TranslateGizmo(camera.camera, gizmoLayer);
        gizmo.size = 0.8;
        // Left button only: the viewer binds right-drag to pan, so only the
        // left button should be able to grab a handle.
        gizmo.mouseButtons[1] = false;
        gizmo.mouseButtons[2] = false;
        pivot = new Entity('labelPivot');
        app.root.addChild(pivot);
        gizmo.on('render:update', () => {
            app.renderNextFrame = true;
        });
        gizmo.on('transform:move', () => {
            if (selection >= 0 && selection < labels.length) {
                labels[selection].position.copy(pivot.getPosition());
                refreshVisuals();
            }
        });
    };

    const syncGizmo = () => {
        ensureGizmo();
        gizmo.detach();
        if (active && selection >= 0 && selection < labels.length) {
            pivot.setPosition(labels[selection].position);
            gizmo.attach(pivot);
        }
        app.renderNextFrame = true;
        refreshVisuals();
    };

    // ---- visuals --------------------------------------------------------
    const refreshVisuals = () => {
        svg.classList.toggle('hidden', labels.length === 0);
        for (let i = 0; i < labels.length; i++) {
            camera.camera.worldToScreen(labels[i].position, _screen);
            const g = groups[i];
            const x = String(_screen.x);
            const y = String(_screen.y);
            const ax = String(_screen.x + LABEL_TEXT_DX);
            const ay = String(_screen.y + LABEL_TEXT_DY);
            const [lineBottom, lineTop, point, textEl] = g.childNodes;
            lineBottom.setAttribute('x1', x);
            lineBottom.setAttribute('y1', y);
            lineBottom.setAttribute('x2', ax);
            lineBottom.setAttribute('y2', ay);
            lineTop.setAttribute('x1', x);
            lineTop.setAttribute('y1', y);
            lineTop.setAttribute('x2', ax);
            lineTop.setAttribute('y2', ay);
            point.setAttribute('cx', x);
            point.setAttribute('cy', y);
            textEl.setAttribute('x', ax);
            textEl.setAttribute('y', ay);
        }
    };

    // ---- label CRUD ------------------------------------------------------
    const addLabel = (text, position) => {
        labels.push({ text, position: position.clone() });
        groups.push(createLabelGroup(text));
        selection = labels.length - 1;
        lastPlacedTime = Date.now();
        syncGizmo();
    };

    const removeLabel = (index) => {
        labels.splice(index, 1);
        const g = groups.splice(index, 1)[0];
        if (g && g.parentNode) {
            g.parentNode.removeChild(g);
        }
        if (selection === index) {
            selection = -1;
            if (gizmo) {
                gizmo.detach();
            }
        }
        else if (selection > index) {
            selection--;
        }
        refreshVisuals();
    };

    const replaceLabels = (next) => {
        for (const g of groups) {
            if (g.parentNode) {
                g.parentNode.removeChild(g);
            }
        }
        groups.length = 0;
        labels.length = 0;
        for (const label of next) {
            labels.push(label);
            groups.push(createLabelGroup(label.text));
        }
        selection = -1;
        if (gizmo) {
            gizmo.detach();
        }
        refreshVisuals();
    };

    // ---- hit testing ------------------------------------------------------
    const findLabelAt = (mx, my) => {
        for (let i = 0; i < labels.length; i++) {
            camera.camera.worldToScreen(labels[i].position, _screen);
            if (Math.abs(_screen.x - mx) <= LABEL_PICK_TOLERANCE &&
                Math.abs(_screen.y - my) <= LABEL_PICK_TOLERANCE) {
                return i;
            }
            // Text sits to the left of its anchor (text-anchor: end).
            if (mx <= _screen.x + LABEL_TEXT_DX &&
                mx >= _screen.x + LABEL_TEXT_DX - LABEL_TEXT_WIDTH &&
                Math.abs(_screen.y + LABEL_TEXT_DY - my) <= 12) {
                return i;
            }
        }
        return -1;
    };

    const selectLabel = (index) => {
        selection = index;
        syncGizmo();
    };

    // ---- .lbl file save/load ----------------------------------------------
    const serializeLabels = () => JSON.stringify({
        format: 'lfs-labels',
        version: 1,
        labels: labels.map((label) => ({
            text: label.text,
            position: [label.position.x, label.position.y, label.position.z]
        }))
    }, null, 2);

    const suggestedName = () => {
        try {
            const name = decodeURIComponent(new URL(location.href).pathname.split('/').pop() || '');
            const base = name.replace(/\.[^.]*$/, '');
            if (base) {
                return base + '.lbl';
            }
        }
        catch (e) {
            // fall through to the default name
        }
        return 'labels.lbl';
    };

    const saveLabels = async () => {
        const text = serializeLabels();
        const name = suggestedName();
        if (window.showSaveFilePicker) {
            try {
                const handle = await window.showSaveFilePicker({
                    suggestedName: name,
                    types: [{ description: 'LichtFeld labels (.lbl)', accept: { 'application/json': ['.lbl', '.json'] } }]
                });
                const writable = await handle.createWritable();
                await writable.write(text);
                await writable.close();
                return;
            }
            catch (err) {
                if (err && err.name === 'AbortError') {
                    return;
                }
            }
        }
        const blob = new Blob([text], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = name;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };

    const parseLabels = (raw) => {
        const parsed = JSON.parse(raw);
        const arr = Array.isArray(parsed) ? parsed : (parsed && parsed.labels);
        if (!Array.isArray(arr)) {
            throw new Error('no labels array found');
        }
        return arr.map((entry, i) => {
            const text = entry && entry.text != null ? String(entry.text) : 'Label ' + (i + 1);
            const pos = Array.isArray(entry.position) ? entry.position : entry.position && [entry.position.x, entry.position.y, entry.position.z];
            if (!Array.isArray(pos) || pos.length < 3 || pos.slice(0, 3).some((v) => !Number.isFinite(Number(v)))) {
                throw new Error('label ' + (i + 1) + ' has an invalid position');
            }
            return {
                text,
                position: new Vec3(Number(pos[0]), Number(pos[1]), Number(pos[2]))
            };
        });
    };

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.lbl,.json,application/json';
    fileInput.classList.add('hidden');
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files && fileInput.files[0];
        fileInput.value = '';
        if (!file) {
            return;
        }
        let next;
        try {
            next = parseLabels(await file.text());
        }
        catch (err) {
            window.alert('Failed to load labels: ' + err.message);
            return;
        }
        if (labels.length > 0 && !window.confirm('Replace ' + labels.length + ' existing label(s) with ' + next.length + ' from the file?')) {
            return;
        }
        replaceLabels(next);
    });
    document.getElementById('ui').appendChild(fileInput);

    // ---- point picking on click (drag = camera navigation, not a pick) ---
    const isPrimary = (e) => (e.pointerType === 'mouse' ? e.button === 0 : e.isPrimary);
    let tracking = false;
    let picking = false;
    let rightTracking = false;
    let downX = 0;
    let downY = 0;

    const onPointerDown = (e) => {
        if (!active) {
            return;
        }
        if (isPrimary(e)) {
            tracking = true;
            downX = e.clientX;
            downY = e.clientY;
        }
        else if (e.pointerType === 'mouse' && e.button === 2) {
            rightTracking = true;
            downX = e.clientX;
            downY = e.clientY;
        }
    };
    const onPointerMove = (e) => {
        const moved = Math.abs(e.clientX - downX) > LABEL_CLICK_DEADZONE || Math.abs(e.clientY - downY) > LABEL_CLICK_DEADZONE;
        if (tracking && moved) {
            tracking = false;
        }
        if (rightTracking && moved) {
            rightTracking = false;
        }
    };
    const onPointerUp = async (e) => {
        if (!active || !tracking || !isPrimary(e) || picking) {
            return;
        }
        tracking = false;

        // Clicking an existing label selects it (gizmo attached) instead of
        // placing a new one.
        const hit = findLabelAt(e.offsetX, e.offsetY);
        if (hit >= 0) {
            selectLabel(hit);
            return;
        }

        if (!picker) {
            picker = new Picker(app, camera);
        }
        picking = true;
        let result = null;
        try {
            result = await picker.pick(e.offsetX, e.offsetY);
        }
        finally {
            picking = false;
        }
        if (!result) {
            return;
        }

        const text = window.prompt('Label text:', 'Label ' + (labels.length + 1));
        if (text === null) {
            return;
        }
        const trimmed = text.trim();
        if (!trimmed) {
            return;
        }
        addLabel(trimmed, result);
    };

    const onDoubleClick = (e) => {
        if (!active || !isPrimary(e)) {
            return;
        }
        // Ignore the double-click that immediately follows a placement: the
        // second click of that pair lands on the freshly added label and
        // would open a surprising rename prompt.
        if (selection >= 0 && selection < labels.length && Date.now() - lastPlacedTime < 1000) {
            return;
        }
        const hit = findLabelAt(e.offsetX, e.offsetY);
        if (hit < 0) {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        const text = window.prompt('Rename label (leave empty to delete):', labels[hit].text);
        if (text === null) {
            return;
        }
        const trimmed = text.trim();
        if (!trimmed) {
            removeLabel(hit);
            return;
        }
        labels[hit].text = trimmed;
        groups[hit].childNodes[3].textContent = trimmed;
    };

    const onContextMenu = (e) => {
        if (!active || e.button !== 2 || !rightTracking) {
            return;
        }
        rightTracking = false;
        const hit = findLabelAt(e.offsetX, e.offsetY);
        if (hit < 0) {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        removeLabel(hit);
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp, true);
    canvas.addEventListener('dblclick', onDoubleClick);
    canvas.addEventListener('contextmenu', onContextMenu);

    app.on('postrender', refreshVisuals);

    // ---- toolbar buttons ---------------------------------------------------
    const modeButton = document.getElementById('labels');
    const saveButton = document.getElementById('labelSave');
    const loadButton = document.getElementById('labelLoad');

    const setActive = (state) => {
        active = state;
        if (modeButton) {
            modeButton.classList.toggle('active', active);
        }
        if (canvas) {
            canvas.style.cursor = state ? 'crosshair' : '';
        }
        if (!active) {
            selection = -1;
            if (gizmo) {
                gizmo.detach();
            }
        }
        app.renderNextFrame = true;
        refreshVisuals();
    };

    modeButton?.addEventListener('click', () => {
        const next = !active;
        if (next) {
            // Only one pick tool should be live at a time: both attach a
            // left-button gizmo, so turn the measure tool off if it is on.
            const measureButton = document.getElementById('measure');
            if (measureButton && measureButton.classList.contains('active')) {
                measureButton.click();
            }
        }
        setActive(next);
    });

    saveButton?.addEventListener('click', () => saveLabels());
    loadButton?.addEventListener('click', () => fileInput.click());

    events?.on('inputEvent', (name) => {
        if (name === 'cancel' && active) {
            setActive(false);
        }
    });
}

export { initLabelTool };
