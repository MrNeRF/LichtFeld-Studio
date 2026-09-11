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
 * IMPORTANT: like gizmo.js, this file is NOT self-contained at runtime. It
 * is concatenated (see src/io/formats/html.cpp) after index.js and gizmo.js
 * and wrapped in an IIFE at HTML export time, so it shares index.js's
 * bundled PlayCanvas engine classes and gizmo.js's `Gizmo`/`TranslateGizmo`
 * by closure rather than a real ES import. Do not add real `import`
 * statements here; the trailing `export` below is stripped at export time.
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
    const points = [];
    let selection = -1;
    let active = false;
    let gizmo = null;
    let gizmoLayer = null;
    let pivot = null;
    let picker = null;

    const _screen = new Vec3();
    const _dragDir = new Vec3();
    let _dragStartLength = 0;

    // ---- SVG overlay (line + endpoint markers) ------------------------
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('id', 'measureToolSvg');
    svg.classList.add('hidden');

    const defs = document.createElementNS(svgNS, 'defs');
    const lineDef = document.createElementNS(svgNS, 'line');
    lineDef.setAttribute('id', 'measureLine');
    defs.appendChild(lineDef);

    const lineBottom = document.createElementNS(svgNS, 'use');
    lineBottom.setAttribute('id', 'measureLineBottom');
    lineBottom.setAttribute('href', '#measureLine');

    const lineTop = document.createElementNS(svgNS, 'use');
    lineTop.setAttribute('id', 'measureLineTop');
    lineTop.setAttribute('href', '#measureLine');

    const startCircle = document.createElementNS(svgNS, 'circle');
    startCircle.setAttribute('id', 'measureLineStart');

    const endCircle = document.createElementNS(svgNS, 'circle');
    endCircle.setAttribute('id', 'measureLineEnd');

    svg.appendChild(defs);
    svg.appendChild(lineBottom);
    svg.appendChild(lineTop);
    svg.appendChild(startCircle);
    svg.appendChild(endCircle);
    document.getElementById('ui').appendChild(svg);

    // ---- floating length panel -----------------------------------------
    const panel = document.createElement('div');
    panel.id = 'measurePanel';
    panel.classList.add('hidden');

    const lengthLabel = document.createElement('span');
    lengthLabel.id = 'measureLengthLabel';
    lengthLabel.textContent = 'Length';

    const lengthInput = document.createElement('input');
    lengthInput.id = 'measureLengthInput';
    lengthInput.type = 'number';
    lengthInput.step = '0.01';
    lengthInput.min = '0.0001';

    const clearButton = document.createElement('button');
    clearButton.id = 'measureClear';
    clearButton.type = 'button';
    clearButton.title = 'Clear measurement';
    clearButton.textContent = '\u00d7';

    panel.appendChild(lengthLabel);
    panel.appendChild(lengthInput);
    panel.appendChild(clearButton);
    panel.addEventListener('pointerdown', (e) => e.stopPropagation());
    document.getElementById('ui').appendChild(panel);

    // ---- gizmo (lazily created on first use) ---------------------------
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
            if (selection >= 0 && selection < points.length) {
                points[selection].copy(pivot.getPosition());
                refreshVisuals();
            }
        });
    };

    const syncGizmo = () => {
        ensureGizmo();
        gizmo.detach();
        if (active && selection >= 0 && selection < points.length) {
            pivot.setPosition(points[selection]);
            gizmo.attach(pivot);
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

        lineBottom.setAttribute('visibility', points.length > 1 ? 'visible' : 'hidden');
        lineTop.setAttribute('visibility', points.length > 1 ? 'visible' : 'hidden');

        for (let i = 0; i < 2; i++) {
            const circle = i === 0 ? startCircle : endCircle;
            if (i < points.length) {
                camera.camera.worldToScreen(points[i], _screen);
                const x = String(_screen.x);
                const y = String(_screen.y);
                if (i === 0) {
                    lineDef.setAttribute('x1', x);
                    lineDef.setAttribute('y1', y);
                } else {
                    lineDef.setAttribute('x2', x);
                    lineDef.setAttribute('y2', y);
                }
                circle.setAttribute('cx', x);
                circle.setAttribute('cy', y);
                circle.setAttribute('visibility', 'visible');
            } else {
                circle.setAttribute('visibility', 'hidden');
            }
        }

        if (points.length === 2) {
            panel.classList.remove('hidden');
            if (document.activeElement !== lengthInput) {
                lengthInput.value = points[0].distance(points[1]).toFixed(3);
            }
        } else {
            panel.classList.add('hidden');
        }
    };

    // ---- length input: dragging point B along the A->B direction --------
    lengthInput.addEventListener('focus', () => {
        if (points.length === 2) {
            _dragDir.sub2(points[1], points[0]);
            _dragStartLength = _dragDir.length();
            if (_dragStartLength > 1e-6) {
                _dragDir.normalize();
            }
        }
    });
    lengthInput.addEventListener('change', () => {
        if (points.length !== 2 || _dragStartLength <= 1e-6) return;
        const newLength = parseFloat(lengthInput.value);
        if (!Number.isFinite(newLength) || newLength <= 0) {
            // Reject 0/empty/negative input instead of silently keeping the
            // old length or snapping the points together.
            lengthInput.value = _dragStartLength.toFixed(3);
            return;
        }
        points[1].copy(_dragDir).mulScalar(newLength).add(points[0]);
        if (selection === 1 && pivot) {
            pivot.setPosition(points[1]);
        }
        app.renderNextFrame = true;
        refreshVisuals();
    });

    clearButton.addEventListener('click', () => {
        points.length = 0;
        selection = -1;
        syncGizmo();
    });

    // ---- point picking on click (drag = camera navigation, not a pick) --
    const isPrimary = (e) => (e.pointerType === 'mouse' ? e.button === 0 : e.isPrimary);
    let clicked = false;
    let picking = false;
    let downX = 0;
    let downY = 0;

    const onPointerDown = (e) => {
        if (active && !clicked && isPrimary(e)) {
            clicked = true;
            downX = e.clientX;
            downY = e.clientY;
        }
    };
    const onPointerMove = (e) => {
        if (!clicked) return;
        if (Math.abs(e.clientX - downX) > CLICK_DEADZONE || Math.abs(e.clientY - downY) > CLICK_DEADZONE) {
            clicked = false;
        }
    };
    const onPointerUp = async (e) => {
        if (!active || !clicked || !isPrimary(e)) return;
        clicked = false;
        if (picking) return;

        let closestIdx = -1;
        for (let i = 0; i < points.length; i++) {
            camera.camera.worldToScreen(points[i], _screen);
            if (Math.abs(_screen.x - e.offsetX) < SCREEN_PICK_TOLERANCE && Math.abs(_screen.y - e.offsetY) < SCREEN_PICK_TOLERANCE) {
                closestIdx = i;
                break;
            }
        }

        if (closestIdx >= 0) {
            selection = closestIdx;
            syncGizmo();
            return;
        }

        if (points.length < 2) {
            if (!picker) {
                picker = new Picker(app, camera);
            }
            picking = true;
            try {
                const result = await picker.pick(e.offsetX, e.offsetY);
                if (result && points.length < 2) {
                    points.push(result.clone());
                    selection = points.length - 1;
                    syncGizmo();
                }
            } finally {
                picking = false;
            }
        }
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp, true);

    app.on('postrender', refreshVisuals);

    // ---- toolbar toggle ---------------------------------------------------
    const button = document.getElementById('measure');
    const setActive = (state) => {
        active = state;
        button?.classList.toggle('active', active);
        if (!active && gizmo) {
            gizmo.detach();
        } else if (active && selection >= 0) {
            syncGizmo();
        }
        app.renderNextFrame = true;
        refreshVisuals();
    };
    button?.addEventListener('click', () => setActive(!active));

    events?.on('inputEvent', (name) => {
        if (name === 'cancel' && active) {
            setActive(false);
        }
    });
}

export { initMeasureTool };
