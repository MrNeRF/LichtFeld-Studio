/*
 * VR annotations for the self-contained HTML viewer export.
 *
 * When a WebXR session starts (immersive-vr or immersive-ar), this module
 * rebuilds the current label + measurement state as in-world 3D entities so
 * they remain visible inside the headset: labels become billboarded text
 * quads with orange point markers, measurements become white dotted lines
 * with endpoint dots and a distance readout at the midpoint. The rest of the GUI
 * (#ui) is hidden for the duration of the session and restored on exit.
 *
 * Locomotion: the viewer's XrNavigation script moves the rig with the LEFT
 * thumbstick (ground plane only) and uses the RIGHT stick for snap turning.
 * This module extends it without touching index.js:
 *   - right thumbstick Y axis -> vertical movement (fly up/down)
 *   - both grips squeezed     -> pinch dolly: hands apart moves the rig
 *     backward along the view direction (zoom out), together moves it forward
 *     (zoom in). Dolly is used instead of scaling the model so that labels and
 *     measurements stay aligned with the splat content.
 *
 * IMPORTANT: like gizmo.js / measure-tool.js / label-tool.js, this file is NOT
 * self-contained at runtime. It is concatenated (see src/io/formats/html.cpp)
 * after them and wrapped in an IIFE at HTML export time, so it shares
 * index.js's bundled PlayCanvas engine classes by closure rather than a real
 * ES import. Do not add real `import` statements here; the trailing `export`
 * below is stripped at export time.
 */

function initVrAnnotations(global) {
    const { app, camera } = global;

    // ---- world-space sizing (meters) --------------------------------------
    const TEXT_HEIGHT_M = 0.12;             // rendered height of a text quad
    const PX_TO_M = TEXT_HEIGHT_M / 96;     // canvas is 96px tall -> m per pixel
    const DOT_SIZE_M = 0.035;               // endpoint / label point markers
    const LABEL_GAP_M = 0.05;               // gap between marker and text quad
    const MEASURE_TEXT_OFFSET_M = 0.07;     // readout offset above the midpoint
    const LINE_DOT_SIZE_M = 0.02;           // dots that make up a measurement line
    const LINE_DOT_MIN_SPACING_M = 0.03;    // minimum gap between line dots
    const LINE_DOT_MAX_COUNT = 80;          // cap per line so long lines stay cheap

    // ---- shared static resources (created once, kept across sessions) ------
    let planeMesh = null;
    let dotTextureOrange = null;
    let dotTextureWhite = null;

    const _dir = new Vec3();
    const _xAxis = new Vec3();
    const _yAxis = new Vec3();
    const _zAxis = new Vec3();
    const _mat4 = new Mat4();
    const _quat = new Quat();

    // ---- per-session state ---------------------------------------------------
    let sessionRoot = null;       // parent of all in-world annotation entities
    const billboards = [];        // { entity, position } re-oriented every frame
    const sessionTextures = [];   // per-session textures, destroyed on end
    const sessionMaterials = [];  // per-session materials, destroyed on end

    const ensureStaticResources = () => {
        if (planeMesh) return;
        planeMesh = Mesh.fromGeometry(app.graphicsDevice, new PlaneGeometry({ widthSegments: 1, lengthSegments: 1 }));
        // MeshInstance.destroy() destroys its mesh once the mesh's refCount
        // drops below 1. This mesh is shared by every session and is cached
        // here, so hold our own reference; otherwise ending the first VR
        // session destroys it and every later session fails in draw() with
        // "Cannot read properties of undefined (reading 'impl')".
        planeMesh.incRefCount();
        dotTextureOrange = createDotTexture('#F60');
        dotTextureWhite = createDotTexture('#FFF');
    };

    // ---- textures ---------------------------------------------------------------
    const finalizeCanvasTexture = (canvas) => {
        const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        // Force the color channels of semi-transparent edge pixels to black so
        // the anti-aliased silhouette blends correctly under SRC_ALPHA blending.
        for (let i = 0; i < data.length; i += 4) {
            if (data[i + 3] < 255) {
                data[i] = 0;
                data[i + 1] = 0;
                data[i + 2] = 0;
            }
        }
        return new Texture(app.graphicsDevice, {
            width: canvas.width,
            height: canvas.height,
            format: PIXELFORMAT_RGBA8,
            magFilter: FILTER_LINEAR,
            minFilter: FILTER_LINEAR,
            mipmaps: false,
            addressU: ADDRESS_CLAMP_TO_EDGE,
            addressV: ADDRESS_CLAMP_TO_EDGE,
            levels: [new Uint8Array(data.buffer)]
        });
    };

    const createTextTexture = (text) => {
        const canvas = document.createElement('canvas');
        let ctx = canvas.getContext('2d');
        const font = 'bold 64px Arial';
        ctx.font = font;
        const padding = 16;
        const textWidth = Math.ceil(ctx.measureText(text).width);
        canvas.width = Math.max(64, textWidth + padding * 2);
        canvas.height = 96;
        // Resizing the canvas resets its context state, so re-apply the font.
        ctx = canvas.getContext('2d');
        ctx.font = font;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 12;
        ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
        ctx.fillStyle = 'white';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);
        return finalizeCanvasTexture(canvas);
    };

    const createDotTexture = (fillColor) => {
        const size = 64;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        // Clear with the border color at zero alpha so the anti-aliased border
        // edge blends correctly.
        ctx.fillStyle = 'black';
        ctx.globalAlpha = 0;
        ctx.fillRect(0, 0, size, size);
        ctx.globalAlpha = 1.0;
        const c = size / 2;
        const radius = (size / 2) - 6;
        ctx.beginPath();
        ctx.arc(c, c, radius, 0, Math.PI * 2);
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.lineWidth = 8;
        ctx.strokeStyle = 'black';
        ctx.stroke();
        return finalizeCanvasTexture(canvas);
    };

    // ---- materials -----------------------------------------------------------------
    const createQuadMaterial = (texture) => {
        const material = new StandardMaterial();
        material.diffuse = Color.BLACK;
        material.emissive.copy(Color.WHITE);   // unlit: texture at full brightness
        material.emissiveMap = texture;
        material.opacityMap = texture;
        material.blendState = new BlendState(true, BLENDEQUATION_ADD, BLENDMODE_SRC_ALPHA, BLENDMODE_ONE_MINUS_SRC_ALPHA, BLENDEQUATION_ADD, BLENDMODE_ONE, BLENDMODE_ONE);
        material.depthTest = true;
        material.depthWrite = false;
        material.cull = CULLFACE_NONE;
        material.useLighting = false;
        material.update();
        return material;
    };

    // ---- billboard math -----------------------------------------------------------------
    // Orient a quad at `position` to face `cameraPos`, keeping text upright.
    // The fork's PlaneGeometry lies in local XZ with normal +Y, and the text
    // top sits on the local -Z side of the quad (canvas row 0 -> v=0), so:
    //   Y' = normalize(cameraPos - position)      (quad normal toward camera)
    //   Z' = normalize(-U + (U.dir)*dir)         (local +Z ~ world down, so
    //        local -Z -- the text top -- points up)
    //   X' = cross(Y', Z')                        (right-handed completion)
    const orientBillboard = (entity, position, cameraPos) => {
        _dir.sub2(cameraPos, position);
        if (_dir.lengthSq() < 1e-8) return;
        _dir.normalize();

        // When the camera is directly above/below, -U vanishes under that
        // projection, so project world +X onto the perpendicular plane instead.
        const uDotDir = _dir.y;
        if (Math.abs(uDotDir) > 0.999) {
            _zAxis.set(1 - _dir.x * _dir.x, -_dir.x * _dir.y, -_dir.x * _dir.z);
        } else {
            _zAxis.set(-_dir.x * uDotDir, -1 + uDotDir * uDotDir, -_dir.z * uDotDir);
        }
        if (_zAxis.lengthSq() < 1e-8) {
            _zAxis.set(1, 0, 0);
        }
        _zAxis.normalize();

        _yAxis.copy(_dir);
        _xAxis.cross(_yAxis, _zAxis);

        const d = _mat4.data;
        d[0] = _xAxis.x; d[1] = _xAxis.y; d[2] = _xAxis.z; d[3] = 0;
        d[4] = _yAxis.x; d[5] = _yAxis.y; d[6] = _yAxis.z; d[7] = 0;
        d[8] = _zAxis.x; d[9] = _zAxis.y; d[10] = _zAxis.z; d[11] = 0;
        d[12] = position.x; d[13] = position.y; d[14] = position.z; d[15] = 1;
        _quat.setFromMat4(_mat4);
        entity.setRotation(_quat);
    };

    // A measurement line drawn as evenly spaced small dot quads. Both
    // app.drawLine() (immediate-mode layer) and a single stretched quad
    // blanked/stalled the page in the headset, whereas dot quads render
    // reliably. All dots of a line share one material, and are billboarded
    // so they face the viewer.
    const createDotLine = (a, b) => {
        const length = a.distance(b);
        const count = Math.max(2, Math.min(LINE_DOT_MAX_COUNT, Math.floor(length / LINE_DOT_MIN_SPACING_M)));
        const material = createQuadMaterial(dotTextureWhite);
        sessionMaterials.push(material);
        // Skip the endpoints (they already have full-size dots); place dots at
        // the interior fractions i / count.
        for (let i = 1; i < count; i++) {
            const t = i / count;
            const position = new Vec3(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
            );
            const meshInstance = new MeshInstance(planeMesh, material);
            const entity = new Entity('lfsVrMeasureLineDot');
            entity.addComponent('render', { meshInstances: [meshInstance] });
            entity.setPosition(position);
            entity.setLocalScale(LINE_DOT_SIZE_M, 1, LINE_DOT_SIZE_M);
            sessionRoot.addChild(entity);
            billboards.push({ entity, position });
        }
    };

    // ---- quad entities ---------------------------------------------------------------------
    const createQuadEntity = (name, texture, position, widthM, heightM) => {
        const material = createQuadMaterial(texture);
        sessionMaterials.push(material);
        const meshInstance = new MeshInstance(planeMesh, material);
        const entity = new Entity(name);
        entity.addComponent('render', { meshInstances: [meshInstance] });
        entity.setPosition(position);
        // The plane geometry is a unit quad in local XZ, so the entity scale
        // (x, z) directly sets its world width/height. (Entity has no
        // setScale(); sessionRoot sits at identity under app.root, so local
        // scale equals world scale.)
        entity.setLocalScale(widthM, 1, heightM);
        return entity;
    };

    const createDotEntity = (name, texture, position) => {
        const material = createQuadMaterial(texture);
        sessionMaterials.push(material);
        const meshInstance = new MeshInstance(planeMesh, material);
        const entity = new Entity(name);
        entity.addComponent('render', { meshInstances: [meshInstance] });
        entity.setPosition(position);
        entity.setLocalScale(DOT_SIZE_M, 1, DOT_SIZE_M);
        return entity;
    };

    // ---- session build/destroy -----------------------------------------------------------------
    const buildAnnotations = () => {
        ensureStaticResources();

        sessionRoot = new Entity('lfsVrAnnotations');
        app.root.addChild(sessionRoot);
        billboards.length = 0;
        sessionTextures.length = 0;
        sessionMaterials.length = 0;

        // Each item is built in its own try/catch: a failure on one label or
        // measurement must not abort the others (or the whole XR session).
        const labels = (window.__lfsLabelTool && window.__lfsLabelTool.get()) || [];
        for (const label of labels) {
            try {
                const position = new Vec3(label.position[0], label.position[1], label.position[2]);

                sessionRoot.addChild(createDotEntity('lfsVrLabelPoint', dotTextureOrange, position));

                if (!label.text) continue;
                const texture = createTextTexture(label.text);
                sessionTextures.push(texture);
                const textPos = new Vec3(position.x, position.y + LABEL_GAP_M, position.z);
                const quad = createQuadEntity(
                    'lfsVrLabelText',
                    texture,
                    textPos,
                    texture.width * PX_TO_M,
                    TEXT_HEIGHT_M
                );
                sessionRoot.addChild(quad);
                billboards.push({ entity: quad, position: textPos });
            } catch (err) {
                console.error('[LFS VR] failed to build label', label, err);
            }
        }

        const measurements = (window.__lfsMeasureTool && window.__lfsMeasureTool.get()) || [];
        for (const m of measurements) {
            try {
                const a = new Vec3(m.a[0], m.a[1], m.a[2]);
                const b = new Vec3(m.b[0], m.b[1], m.b[2]);

                sessionRoot.addChild(createDotEntity('lfsVrMeasurePointA', dotTextureWhite, a));
                sessionRoot.addChild(createDotEntity('lfsVrMeasurePointB', dotTextureWhite, b));

                const mid = new Vec3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
                const textPos = new Vec3(mid.x, mid.y + MEASURE_TEXT_OFFSET_M, mid.z);

                // Connecting line (skipped for degenerate / non-finite measurements).
                const length = a.distance(b);
                if (Number.isFinite(length) && length > 1e-6) {
                    createDotLine(a, b);
                }

                const texture = createTextTexture(length.toFixed(3));
                sessionTextures.push(texture);
                const quad = createQuadEntity(
                    'lfsVrMeasureText',
                    texture,
                    textPos,
                    texture.width * PX_TO_M,
                    TEXT_HEIGHT_M
                );
                sessionRoot.addChild(quad);
                billboards.push({ entity: quad, position: textPos });
            } catch (err) {
                console.error('[LFS VR] failed to build measurement', m, err);
            }
        }
    };

    const destroyAnnotations = () => {
        if (!sessionRoot) return;
        sessionRoot.destroy(true);
        sessionRoot = null;
        billboards.length = 0;
        for (const texture of sessionTextures) {
            texture.destroy();
        }
        for (const material of sessionMaterials) {
            material.destroy();
        }
        sessionTextures.length = 0;
        sessionMaterials.length = 0;
    };

    // ---- VR locomotion extension ---------------------------------------------------------
    const rig = camera.parent;

    let verticalSpeed = 1.5;        // m/s, synced from XrNavigation if present
    let verticalThreshold = 0.1;    // thumbstick deadzone

    const navScript = (rig && rig.script ? rig.script.scripts : [])
        .find((s) => s.constructor && s.constructor.scriptName === 'xrNavigation');
    if (navScript) {
        verticalSpeed = navScript.movementSpeed || 1.5;
        verticalThreshold = navScript.movementThreshold ?? 0.1;
    }

    const PINCH_GAIN = 4.0;         // world meters per meter of hand-spread change
    const PINCH_MAX_STEP_M = 0.5;   // clamp per-frame dolly to avoid jumps

    const xrInputSources = new Set();
    const squeezeState = new Map(); // inputSource -> boolean
    let pinchDist = null;           // hand distance while pinching (meters)

    if (app.xr && app.xr.input) {
        app.xr.input.on('add', (inputSource) => {
            xrInputSources.add(inputSource);
        });
        app.xr.input.on('remove', (inputSource) => {
            xrInputSources.delete(inputSource);
            squeezeState.delete(inputSource);
            pinchDist = null;
        });
        app.xr.input.on('squeezestart', (inputSource) => {
            squeezeState.set(inputSource, true);
        });
        app.xr.input.on('squeezeend', (inputSource) => {
            squeezeState.set(inputSource, false);
            pinchDist = null;
        });
    }

    const _pinchVecA = new Vec3();
    const _pinchVecB = new Vec3();
    const _dollyVec = new Vec3();

    const handleLocomotion = (dt) => {
        if (!rig || !app.xr.active) return;

        // Vertical movement: right thumbstick Y axis.
        for (const inputSource of xrInputSources) {
            if (!inputSource.gamepad || inputSource.handedness !== 'right') continue;
            const stickY = inputSource.gamepad.axes[3];
            if (Math.abs(stickY) > verticalThreshold) {
                rig.translate(0, stickY * verticalSpeed * dt, 0);
            }
        }

        // Pinch dolly: both grips squeezed -> hand-spread change moves the rig
        // along the camera's view direction.
        let left = null;
        let right = null;
        for (const inputSource of xrInputSources) {
            if (!squeezeState.get(inputSource)) continue;
            if (inputSource.handedness === 'left') left = inputSource;
            else if (inputSource.handedness === 'right') right = inputSource;
        }

        if (left && right) {
            _pinchVecA.copy(left.getOrigin());
            _pinchVecB.copy(right.getOrigin());
            const dist = _pinchVecA.distance(_pinchVecB);
            if (pinchDist === null) {
                pinchDist = dist; // first frame of the pinch: no jump
            } else {
                let step = (pinchDist - dist) * PINCH_GAIN; // >0 when hands come together
                if (step > PINCH_MAX_STEP_M) step = PINCH_MAX_STEP_M;
                if (step < -PINCH_MAX_STEP_M) step = -PINCH_MAX_STEP_M;
                if (Math.abs(step) > 0.01) { // ignore hand jitter
                    _dollyVec.copy(camera.forward);
                    rig.translate(_dollyVec.x * step, _dollyVec.y * step, _dollyVec.z * step);
                }
            }
            pinchDist = dist;
        } else {
            pinchDist = null;
        }
    };

    // ---- per-frame update (locomotion + billboards) -------------------------------------
    let updateErrorLogged = false;
    const onUpdate = (dt) => {
        if (!app.xr || !app.xr.active) return;

        handleLocomotion(dt);

        if (!sessionRoot) return;
        try {
            const camPos = camera.getPosition();
            for (let i = 0; i < billboards.length; i++) {
                orientBillboard(billboards[i].entity, billboards[i].position, camPos);
            }
        } catch (err) {
            if (!updateErrorLogged) {
                updateErrorLogged = true;
                console.error('[LFS VR] annotation update failed', err);
            }
        }
    };
    app.on('update', onUpdate);

    // ---- XR session lifecycle ---------------------------------------------------------------
    // IMPORTANT: PlayCanvas fires 'start' from inside a promise chain whose
    // .catch() calls session.end(). Any exception escaping this listener
    // therefore terminates the headset session immediately (and without a
    // matching 'end' event). Never let annotation problems propagate.
    const onXrStart = () => {
        try {
            buildAnnotations();
        } catch (err) {
            console.error('[LFS VR] failed to build annotations; continuing without them', err);
            try {
                destroyAnnotations();
            } catch (cleanupErr) {
                console.error('[LFS VR] annotation cleanup failed', cleanupErr);
            }
        }
        const ui = document.getElementById('ui');
        if (ui) {
            ui.classList.add('hidden');
        }
    };

    const onXrEnd = () => {
        try {
            destroyAnnotations();
        } catch (err) {
            console.error('[LFS VR] annotation cleanup failed', err);
        }
        const ui = document.getElementById('ui');
        if (ui) {
            ui.classList.remove('hidden');
        }
    };

    if (app.xr) {
        app.xr.on('start', onXrStart);
        app.xr.on('end', onXrEnd);
    }
}

export { initVrAnnotations };
