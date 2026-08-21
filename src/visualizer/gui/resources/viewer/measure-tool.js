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
 * is spliced (via tools/generate_html_viewer_resources.py) into the same
 * script scope as gizmo.js and the bundled PlayCanvas engine classes already
 * present in index.js (Vec3, Entity, EventHandler, Picker, etc). Do not add
 * real `import`/`export` statements here.
 */

const SCREEN_PICK_TOLERANCE = 8;

// Prevent the viewer's own orbit/fly camera controller from consuming mouse
// deltas while the measurement gizmo is being dragged. `InputController` only
// accumulates raw input into `this.frame.deltas` each frame (some other part
// of the viewer reads and applies those deltas to the camera); starving it of
// deltas for a frame is enough to keep the camera perfectly still, without
// having to fight DOM event propagation order against the gizmo's own
// pointer handlers (both are plain listeners on the same canvas element, so
// stopPropagation()/preventDefault() alone can't select one over the other).
let _lfsCameraFrozen = false;
const _lfsPatchCameraFreeze = () => {
    if (typeof InputController === 'undefined' || !InputController.prototype) return;
    const proto = InputController.prototype;
    if (proto.update.__lfsFreezePatched) return;
    const originalUpdate = proto.update;
    const patched = function (...args) {
        if (_lfsCameraFrozen) {
            // drain accumulated input so movement doesn't jump once unfrozen
            this._desktopInput?.read();
            this._orbitInput?.read();
            this._flyInput?.read();
            this._gamepadInput?.read();
            return;
        }
        return originalUpdate.apply(this, args);
    };
    patched.__lfsFreezePatched = true;
    proto.update = patched;
};

function initMeasureTool(global) {
    const { app, camera, events } = global;
    const canvas = app.graphicsDevice.canvas;

    _lfsPatchCameraFreeze();

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
        pivot = new Entity('measurePivot');
        app.root.addChild(pivot);
        gizmo.on('render:update', () => {
            app.renderNextFrame = true;
        });
        gizmo.on('transform:start', () => {
            _lfsCameraFrozen = true;
        });
        gizmo.on('transform:move', () => {
            if (selection >= 0 && selection < points.length) {
                points[selection].copy(pivot.getPosition());
                refreshVisuals();
            }
        });
        gizmo.on('transform:end', () => {
            _lfsCameraFrozen = false;
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
        const newLength = Math.max(0.0001, parseFloat(lengthInput.value) || _dragStartLength);
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

    const onPointerDown = (e) => {
        if (active && !clicked && isPrimary(e)) {
            clicked = true;
        }
    };
    const onPointerMove = () => {
        clicked = false;
    };
    const onPointerUp = async (e) => {
        if (!active || !clicked || !isPrimary(e)) return;
        clicked = false;

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
            const result = await picker.pick(e.offsetX, e.offsetY);
            if (result) {
                points.push(result.clone());
                selection = points.length - 1;
                syncGizmo();
            }
            e.preventDefault();
            e.stopPropagation();
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
        // defensive: don't leave the camera frozen if the tool is toggled
        // off mid-drag
        _lfsCameraFrozen = false;
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
