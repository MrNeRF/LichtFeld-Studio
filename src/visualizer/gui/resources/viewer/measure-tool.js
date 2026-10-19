/*
 * Measurement tool for the self-contained HTML viewer export.
 *
 * Reimplemented (not copied) from SuperSplat's editor-only measure tool
 * (https://github.com/playcanvas/supersplat, src/tools/measure-tool.ts, MIT
 * licensed) against the viewer's much simpler runtime: there is no PCUI, no
 * editor `Scene`/`Splat`/undo-history abstractions here, and measurement
 * points are plain world-space Vec3s rather than positions relative to a
 * splat's own transform.
 *
 * Unlike the original (a single A->B segment), this tool keeps ALL completed
 * measurements visible at once: each click on empty model space alternates
 * between placing point A of a new measurement and point B that completes it,
 * so a series of clicks produces a list of independent segments rendered in
 * parallel. Completed measurements can be selected (gizmo drag), resized via
 * the per-measurement length input, or deleted individually.
 *
 * IMPORTANT: like gizmo.js, this file is NOT self-contained at runtime. It
 * is concatenated (see src/io/formats/html.cpp) after index.js and gizmo.js
 * and wrapped in an IIFE at HTML export time, so it shares index.js's
 * bundled PlayCanvas engine classes and gizmo.js's `Gizmo`/`TranslateGizmo`
 * by closure rather than a real ES import. Do not add real `import`
 * statements here; the trailing `export` below is stripped at export time.
 *
 * Interactions (measure mode must be toggled on via the #measure button):
 *   - click empty model space -> place next point (A of a new measurement,
 *     then B completing it; repeats for further measurements)
 *   - click an existing endpoint -> select that measurement (gizmo attached,
 *     drag to move just that endpoint)
 *   - right-click an endpoint  -> delete its whole measurement
 * The #measurePanel lists every completed measurement with an editable length
 * input and a per-row delete button.
 */

const SCREEN_PICK_TOLERANCE = 8;
// Pointer must move less than this (px) between down and up for a click to
// register as a point pick; matches the viewer's own double-tap tolerance.
// Without a deadzone, ordinary mouse/trackpad jitter between press and
// release cancels almost every pick attempt.
const CLICK_DEADZONE = 8;

function initMeasureTool(global) {
    const { app, camera, events } = global;
    const canvas = app.graphicsDevice.canvas;

    // ---- state -------------------------------------------------------
    // Each measurement is { a: Vec3, b: Vec3|null }; while being built it has
    // only `a` (b === null) and is tracked via openIdx.
    const measurements = [];
    let openIdx = -1;      // index of the in-progress measurement (-1 none)
    let selM = -1;         // selected measurement whose endpoint is gizmored
    let selEnd = 0;        // which endpoint of selM (0=a, 1=b)
    let active = false;
    let gizmo = null;
    let gizmoLayer = null;
    let pivot = null;
    let picker = null;

    const _screen = new Vec3();
    const _view = new Vec3();

    // A point behind the camera (positive view-space z, since the camera
    // looks down -Z) would otherwise still project onto screen via
    // worldToScreen() and become visible/clickable in the middle of the view
    // when flying past it.
    const isBehindCamera = (position) => {
        camera.camera.viewMatrix.transformPoint(position, _view);
        return _view.z >= 0;
    };

    // ---- SVG overlay (per-measurement group: line + endpoint markers) --
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('id', 'measureToolSvg');
    svg.classList.add('hidden');
    document.getElementById('ui').appendChild(svg);

    // One <g> per measurement; each carries its own line pair and endpoint
    // circles so segments can be positioned independently. (A single shared
    // line definition cannot serve multiple measurements at once.)
    const groups = [];   // SVG <g> per measurement, parallel to `measurements`

    const createMeasureGroup = () => {
        const g = document.createElementNS(svgNS, 'g');
        g.setAttribute('class', 'measureGroup');

        const lineBottom = document.createElementNS(svgNS, 'line');
        lineBottom.setAttribute('class', 'measureLineBottom');
        const lineTop = document.createElementNS(svgNS, 'line');
        lineTop.setAttribute('class', 'measureLineTop');
        const startCircle = document.createElementNS(svgNS, 'circle');
        startCircle.setAttribute('class', 'measurePoint measureStart');
        const endCircle = document.createElementNS(svgNS, 'circle');
        endCircle.setAttribute('class', 'measurePoint measureEnd');

        g.appendChild(lineBottom);
        g.appendChild(lineTop);
        g.appendChild(startCircle);
        g.appendChild(endCircle);
        svg.appendChild(g);
        return g;
    };

    // ---- floating measurement list panel -------------------------------
    const panel = document.createElement('div');
    panel.id = 'measurePanel';
    panel.classList.add('hidden');
    panel.addEventListener('pointerdown', (e) => e.stopPropagation());
    document.getElementById('ui').appendChild(panel);

    const rows = [];       // { row, input, state } per measurement
    const rowInputs = [];  // length <input> per measurement, parallel to `measurements`

    const createMeasureRow = (index) => {
        const row = document.createElement('div');
        row.className = 'measureRow';

        const idxLabel = document.createElement('span');
        idxLabel.className = 'measureIndex';
        idxLabel.textContent = String(index + 1);

        const input = document.createElement('input');
        input.type = 'number';
        input.step = '0.01';
        input.min = '0.0001';
        input.className = 'measureLengthInput';

        const delButton = document.createElement('button');
        delButton.type = 'button';
        delButton.className = 'measureDelete';
        delButton.title = 'Delete measurement';
        delButton.textContent = '\u00d7';

        row.appendChild(idxLabel);
        row.appendChild(input);
        row.appendChild(delButton);
        panel.appendChild(row);

        // Per-row drag state: editing a length input moves that measurement's
        // B endpoint along its own A->B direction. Kept per row (not shared)
        // so interleaved focus/change events between rows can't cross-wire
        // directions.
        const state = { dir: new Vec3(), startLength: 0 };

        input.addEventListener('focus', () => {
            const m = measurements[index];
            if (m && m.b) {
                state.dir.sub2(m.b, m.a);
                state.startLength = state.dir.length();
                if (state.startLength > 1e-6) {
                    state.dir.normalize();
                }
            }
        });
        input.addEventListener('change', () => {
            const m = measurements[index];
            if (!m || !m.b) return;
            const newLength = parseFloat(input.value);
            if (!Number.isFinite(newLength) || newLength <= 0) {
                // Reject 0/empty/negative input instead of silently keeping the
                // old length or snapping the points together.
                input.value = state.startLength.toFixed(3);
                return;
            }
            m.b.copy(state.dir).mulScalar(newLength).add(m.a);
            if (selM === index && selEnd === 1 && pivot) {
                pivot.setPosition(m.b);
            }
            app.renderNextFrame = true;
            refreshVisuals();
        });
        delButton.addEventListener('click', () => removeMeasurement(index));

        return { row, input, state };
    };

    const rebuildRows = () => {
        for (const r of rows) {
            if (r.row.parentNode) {
                r.row.parentNode.removeChild(r.row);
            }
        }
        rows.length = 0;
        rowInputs.length = 0;
        for (let i = 0; i < measurements.length; i++) {
            const built = createMeasureRow(i);
            rows.push(built);
            rowInputs[i] = built.input;
        }
    };

    // ---- gizmo (lazily created on first selection) ---------------------
    const ensureGizmo = () => {
        if (gizmo) return;
        gizmoLayer = Gizmo.createLayer(app, 'LfsMeasureGizmo');
        gizmo = new TranslateGizmo(camera.camera, gizmoLayer);
        gizmo.size = 0.8;
        // Left button only: the viewer binds right-drag to pan and
        // middle/right drags are otherwise meaningful camera gestures, so
        // only the left button should be able to grab a handle.
        gizmo.mouseButtons[1] = false;
        gizmo.mouseButtons[2] = false;
        pivot = new Entity('measurePivot');
        app.root.addChild(pivot);
        gizmo.on('render:update', () => {
            app.renderNextFrame = true;
        });
        gizmo.on('transform:move', () => {
            if (selM >= 0 && selM < measurements.length) {
                const m = measurements[selM];
                const p = selEnd === 0 ? m.a : m.b;
                if (p) {
                    p.copy(pivot.getPosition());
                    refreshVisuals();
                }
            }
        });
    };

    const syncGizmo = () => {
        ensureGizmo();
        gizmo.detach();
        if (active && selM >= 0 && selM < measurements.length) {
            const m = measurements[selM];
            const p = selEnd === 0 ? m.a : m.b;
            if (p) {
                pivot.setPosition(p);
                gizmo.attach(pivot);
            }
        }
        app.renderNextFrame = true;
        refreshVisuals();
    };

    // ---- visuals ---------------------------------------------------------
    const refreshVisuals = () => {
        if (!active) {
            svg.classList.add('hidden');
            panel.classList.add('hidden');
            return;
        }
        svg.classList.remove('hidden');

        for (let i = 0; i < measurements.length; i++) {
            const m = measurements[i];
            const g = groups[i];
            if (!m.b || isBehindCamera(m.a) || isBehindCamera(m.b)) {
                g.classList.add('hidden');
            } else {
                g.classList.remove('hidden');
                camera.camera.worldToScreen(m.a, _screen);
                const ax = String(_screen.x), ay = String(_screen.y);
                camera.camera.worldToScreen(m.b, _screen);
                const bx = String(_screen.x), by = String(_screen.y);
                const [lineBottom, lineTop, startCircle, endCircle] = g.childNodes;
                for (const ln of [lineBottom, lineTop]) {
                    ln.setAttribute('x1', ax);
                    ln.setAttribute('y1', ay);
                    ln.setAttribute('x2', bx);
                    ln.setAttribute('y2', by);
                }
                startCircle.setAttribute('cx', ax);
                startCircle.setAttribute('cy', ay);
                endCircle.setAttribute('cx', bx);
                endCircle.setAttribute('cy', by);
            }

            const row = rows[i];
            if (row) {
                // In-progress measurements (no B yet) keep their row hidden.
                row.row.classList.toggle('hidden', !m.b);
                if (m.b && document.activeElement !== row.input) {
                    row.input.value = m.a.distance(m.b).toFixed(3);
                }
            }
        }

        panel.classList.toggle('hidden', measurements.every((mm) => !mm.b));
    };

    // ---- measurement CRUD --------------------------------------------------
    const placePoint = (position) => {
        let idx;
        if (openIdx >= 0 && measurements[openIdx] && !measurements[openIdx].b) {
            // Second click of the current pair: complete it.
            measurements[openIdx].b = position.clone();
            idx = openIdx;
            openIdx = -1;
        } else {
            // First click: start a new measurement.
            measurements.push({ a: position.clone(), b: null });
            groups.push(createMeasureGroup());
            idx = measurements.length - 1;
            openIdx = idx;
        }
        selM = idx;
        selEnd = measurements[idx].b ? 1 : 0;
        rebuildRows();
        syncGizmo();
    };

    const selectEndpoint = (m, end) => {
        selM = m;
        selEnd = end;
        syncGizmo();
    };

    const removeMeasurement = (index) => {
        measurements.splice(index, 1);
        const g = groups.splice(index, 1)[0];
        if (g && g.parentNode) {
            g.parentNode.removeChild(g);
        }
        if (openIdx === index) {
            openIdx = -1;
        } else if (openIdx > index) {
            openIdx--;
        }
        if (selM === index) {
            selM = -1;
        } else if (selM > index) {
            selM--;
        }
        rebuildRows();
        syncGizmo();
    };

    const replaceMeasurements = (list) => {
        for (const g of groups) {
            if (g.parentNode) {
                g.parentNode.removeChild(g);
            }
        }
        groups.length = 0;
        measurements.length = 0;
        for (const entry of list) {
            const a = new Vec3(entry.a[0], entry.a[1], entry.a[2]);
            const b = new Vec3(entry.b[0], entry.b[1], entry.b[2]);
            measurements.push({ a, b });
            groups.push(createMeasureGroup());
        }
        openIdx = -1;
        selM = -1;
        if (gizmo) {
            gizmo.detach();
        }
        rebuildRows();
        refreshVisuals();
    };

    // ---- hit testing ------------------------------------------------------
    const findEndpointAt = (mx, my) => {
        for (let i = 0; i < measurements.length; i++) {
            const m = measurements[i];
            for (let end = 0; end < 2; end++) {
                const p = end === 0 ? m.a : m.b;
                // A point hidden behind the camera must not be clickable
                // either (it would otherwise still project onto the visible
                // screen).
                if (!p || isBehindCamera(p)) {
                    continue;
                }
                camera.camera.worldToScreen(p, _screen);
                if (Math.abs(_screen.x - mx) <= SCREEN_PICK_TOLERANCE && Math.abs(_screen.y - my) <= SCREEN_PICK_TOLERANCE) {
                    return { m: i, end };
                }
            }
        }
        return null;
    };

    // ---- point picking on click (drag = camera navigation, not a pick) --
    const isPrimary = (e) => (e.pointerType === 'mouse' ? e.button === 0 : e.isPrimary);
    let tracking = false;
    let picking = false;
    let rightTracking = false;
    let downX = 0;
    let downY = 0;

    const onPointerDown = (e) => {
        if (!active) return;
        if (isPrimary(e)) {
            tracking = true;
            downX = e.clientX;
            downY = e.clientY;
        } else if (e.pointerType === 'mouse' && e.button === 2) {
            rightTracking = true;
            downX = e.clientX;
            downY = e.clientY;
        }
    };
    const onPointerMove = (e) => {
        const moved = Math.abs(e.clientX - downX) > CLICK_DEADZONE || Math.abs(e.clientY - downY) > CLICK_DEADZONE;
        if (tracking && moved) {
            tracking = false;
        }
        if (rightTracking && moved) {
            rightTracking = false;
        }
    };
    const onPointerUp = async (e) => {
        if (!active || !tracking || !isPrimary(e) || picking) return;
        tracking = false;

        // Clicking an existing endpoint selects its measurement (gizmo
        // attached to that endpoint) instead of placing a new point.
        const hit = findEndpointAt(e.offsetX, e.offsetY);
        if (hit) {
            selectEndpoint(hit.m, hit.end);
            return;
        }

        if (!picker) {
            picker = new Picker(app, camera);
        }
        picking = true;
        let result = null;
        try {
            result = await picker.pick(e.offsetX, e.offsetY);
        } finally {
            picking = false;
        }
        if (result) {
            placePoint(result);
        }
    };

    const onContextMenu = (e) => {
        if (!active || e.button !== 2 || !rightTracking) return;
        rightTracking = false;
        const hit = findEndpointAt(e.offsetX, e.offsetY);
        if (!hit) return;
        e.preventDefault();
        e.stopPropagation();
        removeMeasurement(hit.m);
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp, true);
    canvas.addEventListener('contextmenu', onContextMenu);

    app.on('postrender', refreshVisuals);

    // ---- toolbar toggle ---------------------------------------------------
    const button = document.getElementById('measure');
    const setActive = (state) => {
        active = state;
        button?.classList.toggle('active', active);
        if (!active && gizmo) {
            gizmo.detach();
        } else if (active && selM >= 0) {
            syncGizmo();
        }
        app.renderNextFrame = true;
        refreshVisuals();
    };
    button?.addEventListener('click', () => {
        const next = !active;
        if (next) {
            // Only one pick tool should be live at a time: both attach a
            // left-button gizmo, so turn the label tool off if it is on.
            const labelButton = document.getElementById('labels');
            if (labelButton && labelButton.classList.contains('active')) {
                labelButton.click();
            }
        }
        setActive(next);
    });

    events?.on('inputEvent', (name) => {
        if (name === 'cancel' && active) {
            setActive(false);
        }
    });

    // Hooks consumed by the label tool so a single .labels.json save/load can
    // round-trip measurements alongside labels.
    window.__lfsMeasureTool = {
        get: () => measurements
            .filter((m) => m.b)
            .map((m) => ({
                a: [m.a.x, m.a.y, m.a.z],
                b: [m.b.x, m.b.y, m.b.z]
            })),
        set: (list) => replaceMeasurements(list || [])
    };
}

export { initMeasureTool };
