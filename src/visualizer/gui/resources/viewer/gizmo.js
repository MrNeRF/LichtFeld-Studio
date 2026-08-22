/*
 * Trimmed/adapted translate-gizmo module from the PlayCanvas Engine
 * (https://github.com/playcanvas/engine), MIT licensed:
 *
 *   Copyright (c) 2011-2024 PlayCanvas Ltd.
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *   THE SOFTWARE.
 *
 * Only the pieces needed for a single-point TranslateGizmo are kept: the
 * `Gizmo` base class, `TransformGizmo` base class, and `TranslateGizmo` plus
 * its shape helpers (`Shape`, `SphereShape`, `PlaneShape`, `ArrowShape`) and
 * the `Tri`/`TriData` ray-picking helpers. RotateGizmo, ScaleGizmo, snapping
 * UI, and deprecated legacy aliases were removed since they are unused here.
 *
 * IMPORTANT: this file is NOT a self-contained module at runtime. It is
 * concatenated (see src/io/formats/html.cpp) after index.js and wrapped in
 * an IIFE together with measure-tool.js when the HTML viewer is exported, so
 * it shares index.js's bundled PlayCanvas engine classes (Vec3, Quat, Mat4,
 * Ray, Plane, Color, Layer, Entity, MeshInstance, Mesh, ShaderMaterial,
 * EventHandler, math, and the various geometry/blend/cull-mode constants) by
 * closure rather than a real ES import. Do not add real `import` statements
 * here; the trailing `export` below is stripped at export time.
 *
 * Input handling note: `Gizmo` registers its pointer listeners on the canvas
 * with `{ capture: true }` and calls `stopImmediatePropagation()` on a
 * handle hit (and for the remainder of an active drag), so the viewer's own
 * orbit/fly camera controller - whose listeners are bubble-phase listeners
 * on the same canvas - never observes drag input. Per the DOM spec, capturing
 * listeners registered on the event target itself run before bubbling ones
 * registered on that same target, regardless of add order (see
 * https://dom.spec.whatwg.org/#dispatching-events); this is honored by
 * current Chromium and Firefox releases.
 */

// ---------------------------------------------------------------------------
// Tri / TriData - ray-triangle intersection helpers used for gizmo picking.
// ---------------------------------------------------------------------------

const _triE1 = new Vec3();
const _triE2 = new Vec3();
const _triH = new Vec3();
const _triS = new Vec3();
const _triQ = new Vec3();
const TRI_EPSILON = 1e-6;

class Tri {
    constructor(v0 = Vec3.ZERO, v1 = Vec3.ZERO, v2 = Vec3.ZERO) {
        this.v0 = new Vec3();
        this.v1 = new Vec3();
        this.v2 = new Vec3();
        this.set(v0, v1, v2);
    }

    set(v0, v1, v2) {
        this.v0.copy(v0);
        this.v1.copy(v1);
        this.v2.copy(v2);
        return this;
    }

    intersectsRay(ray, point) {
        _triE1.sub2(this.v1, this.v0);
        _triE2.sub2(this.v2, this.v0);
        _triH.cross(ray.direction, _triE2);
        const a = _triE1.dot(_triH);
        if (a > -TRI_EPSILON && a < TRI_EPSILON) {
            return false;
        }
        const f = 1 / a;
        _triS.sub2(ray.origin, this.v0);
        const u = f * _triS.dot(_triH);
        if (u < 0 || u > 1) {
            return false;
        }
        _triQ.cross(_triS, _triE1);
        const v = f * ray.direction.dot(_triQ);
        if (v < 0 || u + v > 1) {
            return false;
        }
        const t = f * _triE2.dot(_triQ);
        if (t > TRI_EPSILON) {
            if (point instanceof Vec3) {
                point.copy(ray.direction).mulScalar(t).add(ray.origin);
            }
            return true;
        }
        return false;
    }
}

const _triDataV0 = new Vec3();
const _triDataV1 = new Vec3();
const _triDataV2 = new Vec3();

class TriData {
    constructor(geometry, priority = 0) {
        this._priority = 0;
        this._transform = new Mat4();
        this.tris = [];
        this.fromGeometry(geometry);
        this._priority = priority;
    }

    get transform() {
        return this._transform;
    }

    get priority() {
        return this._priority;
    }

    setTransform(pos = new Vec3(), rot = new Quat(), scale = new Vec3()) {
        this.transform.setTRS(pos, rot, scale);
    }

    fromGeometry(geometry) {
        if (!geometry || !(geometry instanceof Geometry)) {
            throw new Error('No geometry provided.');
        }
        const positions = geometry.positions ?? [];
        const indices = geometry.indices ?? [];
        this.tris = [];
        for (let k = 0; k < indices.length; k += 3) {
            const i1 = indices[k];
            const i2 = indices[k + 1];
            const i3 = indices[k + 2];
            _triDataV0.set(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]);
            _triDataV1.set(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]);
            _triDataV2.set(positions[i3 * 3], positions[i3 * 3 + 1], positions[i3 * 3 + 2]);
            this.tris.push(new Tri(_triDataV0, _triDataV1, _triDataV2));
        }
    }
}

// ---------------------------------------------------------------------------
// Colors and the unlit shader used to render gizmo shapes.
// ---------------------------------------------------------------------------

const COLOR_RED = Object.freeze(new Color(1, 0.3, 0.3));
const COLOR_GREEN = Object.freeze(new Color(0.3, 1, 0.3));
const COLOR_BLUE = Object.freeze(new Color(0.3, 0.3, 1));
const COLOR_GRAY = Object.freeze(new Color(0.5, 0.5, 0.5, 0.5));

const unlitShader = {
    uniqueName: 'lfs-gizmo-unlit',
    attributes: {
        vertex_position: SEMANTIC_POSITION
    },
    vertexGLSL: /* glsl */ `
        attribute vec3 vertex_position;

        uniform mat4 matrix_model;
        uniform mat4 matrix_viewProjection;

        void main(void) {
            gl_Position = matrix_viewProjection * matrix_model * vec4(vertex_position, 1.0);
            gl_Position.z = clamp(gl_Position.z, -abs(gl_Position.w), abs(gl_Position.w));
        }
    `,
    fragmentGLSL: /* glsl */ `
        #include "gammaPS"

        precision highp float;

        uniform vec4 uColor;
        uniform float uDepth;

        void main(void) {
            if (uColor.a < 1.0 / 255.0) {
                discard;
            }
            gl_FragColor = vec4(gammaCorrectOutput(decodeGamma(uColor)), uColor.a);
            #if DEPTH_WRITE == 1
                gl_FragDepth = uDepth;
            #endif
        }
    `
};

// ---------------------------------------------------------------------------
// Shape - base class for a single gizmo handle (mesh + intersection tris).
// ---------------------------------------------------------------------------

class Shape {
    constructor(device, name, args) {
        this._position = new Vec3();
        this._rotation = new Vec3();
        this._scale = new Vec3(1, 1, 1);
        this._layers = [];
        this._material = new ShaderMaterial(unlitShader);
        this._disabled = false;
        this._visible = true;
        this._defaultColor = Color.WHITE;
        this._hoverColor = Color.BLACK;
        this._disabledColor = COLOR_GRAY;
        this._cull = CULLFACE_BACK;
        this._depth = -1;
        this.triData = [];
        this.meshInstances = [];

        this.device = device;
        this.axis = args.axis ?? 'x';
        if (args.position instanceof Vec3) this._position.copy(args.position);
        if (args.rotation instanceof Vec3) this._rotation.copy(args.rotation);
        if (args.scale instanceof Vec3) this._scale.copy(args.scale);
        this._disabled = args.disabled ?? this._disabled;
        this._visible = args.visible ?? this._visible;
        this._layers = args.layers ?? this._layers;
        if (args.defaultColor instanceof Color) this._defaultColor = args.defaultColor;
        if (args.hoverColor instanceof Color) this._hoverColor = args.hoverColor;
        if (args.disabledColor instanceof Color) this._disabledColor = args.disabledColor;
        this._cull = args.cull ?? this._cull;
        this._depth = args.depth ?? this._depth;

        this.entity = new Entity(`${name}:${this.axis}`);
        this.entity.setLocalPosition(this._position);
        this.entity.setLocalEulerAngles(this._rotation);
        this.entity.setLocalScale(this._scale);
    }

    set disabled(value) {
        this._disabled = value ?? false;
        this.hover(false);
    }

    get disabled() {
        return this._disabled;
    }

    set visible(value) {
        if (value === this._visible) return;
        for (let i = 0; i < this.meshInstances.length; i++) {
            this.meshInstances[i].visible = value;
        }
        this._visible = value;
    }

    get visible() {
        return this._visible;
    }

    _createRenderComponent(entity, meshes) {
        const color = this._disabled ? this._disabledColor : this._defaultColor;
        this._material.setDefine('DEPTH_WRITE', this._depth > 0 ? '1' : '0');
        this._material.setParameter('uDepth', this._depth);
        this._material.setParameter('uColor', color.toArray());
        this._material.cull = this._cull;
        this._material.blendType = BLEND_NORMAL;
        this._material.update();
        const meshInstances = [];
        for (let i = 0; i < meshes.length; i++) {
            const mi = new MeshInstance(meshes[i], this._material);
            mi.cull = false;
            meshInstances.push(mi);
            this.meshInstances.push(mi);
        }
        entity.addComponent('render', {
            meshInstances,
            layers: this._layers,
            castShadows: false
        });
    }

    _update() {
        this.entity.setLocalPosition(this._position);
        this.entity.setLocalEulerAngles(this._rotation);
        this.entity.setLocalScale(this._scale);
    }

    hover(state) {
        const color = this._disabled ? this._disabledColor : (state ? this._hoverColor : this._defaultColor);
        this._material.setParameter('uColor', color.toArray());
    }

    destroy() {
        this.entity.destroy();
    }
}

// ---------------------------------------------------------------------------
// Gizmo - base class handling pointer events, positioning, and scaling.
// ---------------------------------------------------------------------------

const _gizmoV = new Vec3();
const _gizmoPosition = new Vec3();
const _gizmoDir = new Vec3();
const _gizmoRotation = new Quat();
const _gizmoM1 = new Mat4();
const _gizmoM2 = new Mat4();
const _gizmoRay = new Ray();

const GIZMO_MIN_SCALE = 1e-4;
const GIZMO_PERS_SCALE_RATIO = 0.3;
const GIZMO_ORTHO_SCALE_RATIO = 0.32;
const GIZMO_UPDATE_EPSILON = 1e-6;
const GIZMO_DIST_EPSILON = 1e-4;

class Gizmo extends EventHandler {
    static createLayer(app, layerName = 'Gizmo', layerIndex = app.scene.layers.layerList.length) {
        const layer = new Layer({
            name: layerName,
            clearDepthBuffer: true,
            opaqueSortMode: SORTMODE_NONE,
            transparentSortMode: SORTMODE_NONE
        });
        app.scene.layers.insert(layer, layerIndex);
        return layer;
    }

    constructor(camera, layer, name = 'gizmo') {
        super();
        this._size = 1;
        this._scale = 1;
        this._coordSpace = 'world';
        this._handles = [];
        this._mouseButtons = [true, true, true];
        this._renderUpdate = false;
        this.nodes = [];
        this.intersectShapes = [];
        this.preventDefault = true;
        // True from a handle hit (pointerdown) through the matching pointerup,
        // even if the pointer strays off every shape's hit area mid-drag. Used
        // to keep swallowing input for the whole drag, not just while directly
        // over a shape.
        this._activeDrag = false;

        this._layer = layer;
        this._camera = camera;
        this._camera.layers = this._camera.layers.concat(this._layer.id);
        this._app = this._camera.system.app;
        this._device = this._app.graphicsDevice;
        this.root = new Entity(name);
        this._app.root.addChild(this.root);
        this.root.enabled = false;
        this._updateScale();

        this._onPointerDown = this._onPointerDown.bind(this);
        this._onPointerMove = this._onPointerMove.bind(this);
        this._onPointerUp = this._onPointerUp.bind(this);
        this._onPointerCancel = this._onPointerCancel.bind(this);
        // Capture phase: on the event target itself, capturing listeners run
        // before bubbling ones (see the file header note), so this reliably
        // runs before the viewer's own bubble-phase camera-control listeners.
        this._device.canvas.addEventListener('pointerdown', this._onPointerDown, true);
        this._device.canvas.addEventListener('pointermove', this._onPointerMove, true);
        this._device.canvas.addEventListener('pointerup', this._onPointerUp, true);
        this._device.canvas.addEventListener('pointercancel', this._onPointerCancel, true);
        this._handles.push(this._app.on('prerender', () => this.prerender()));
        this._handles.push(this._app.on('update', () => this.update()));
        this._handles.push(this._app.on('destroy', () => this.destroy()));
    }

    set enabled(state) {
        const cameraDist = this.root.getLocalPosition().distance(this.camera.entity.getPosition());
        const enabled = state ? (this.nodes.length > 0 && cameraDist > GIZMO_DIST_EPSILON) : false;
        if (enabled !== this.root.enabled) {
            this.root.enabled = enabled;
            this._renderUpdate = true;
        }
    }

    get enabled() {
        return this.root.enabled;
    }

    get mouseButtons() {
        return this._mouseButtons;
    }

    set layer(layer) {
        if (this._layer === layer) return;
        this._camera.layers = this._camera.layers.filter((id) => id !== this._layer.id);
        this._layer = layer;
        this._camera.layers = this._camera.layers.concat(this._layer.id);
        this.enabled = true;
    }

    get layer() {
        return this._layer;
    }

    set camera(camera) {
        if (this._camera === camera) return;
        this._camera.layers = this._camera.layers.filter((id) => id !== this._layer.id);
        this._camera = camera;
        this._camera.layers = this._camera.layers.concat(this._layer.id);
        this.enabled = true;
    }

    get camera() {
        return this._camera;
    }

    set coordSpace(value) {
        this._coordSpace = value ?? this._coordSpace;
        this._updateRotation();
    }

    get coordSpace() {
        return this._coordSpace;
    }

    set size(value) {
        this._size = value;
        this._updateScale();
    }

    get size() {
        return this._size;
    }

    get facingDir() {
        if (this._camera.projection === PROJECTION_PERSPECTIVE) {
            const gizmoPos = this.root.getLocalPosition();
            const cameraPos = this._camera.entity.getPosition();
            return _gizmoDir.sub2(cameraPos, gizmoPos).normalize();
        }
        return _gizmoDir.copy(this._camera.entity.forward).mulScalar(-1);
    }

    get cameraDir() {
        const cameraPos = this._camera.entity.getPosition();
        const gizmoPos = this.root.getLocalPosition();
        return _gizmoDir.sub2(cameraPos, gizmoPos).normalize();
    }

    _onPointerDown(e) {
        if (!this.enabled || document.pointerLockElement) return;
        if (!this.mouseButtons[e.button]) return;
        const selection = this._getSelection(e.offsetX, e.offsetY);
        if (selection[0]) {
            this._activeDrag = true;
            if (this.preventDefault) e.preventDefault();
            e.stopImmediatePropagation();
        }
        const { canvas } = this._device;
        canvas.setPointerCapture(e.pointerId);
        this.fire(Gizmo.EVENT_POINTERDOWN, e.offsetX, e.offsetY, selection[0]);
    }

    _onPointerMove(e) {
        if (!this.enabled || document.pointerLockElement) return;
        const selection = this._getSelection(e.offsetX, e.offsetY);
        if (selection[0] || this._activeDrag) {
            if (this.preventDefault) e.preventDefault();
            e.stopImmediatePropagation();
        }
        this.fire(Gizmo.EVENT_POINTERMOVE, e.offsetX, e.offsetY, selection[0]);
    }

    _onPointerUp(e) {
        if (!this.mouseButtons[e.button]) return;
        const wasDragging = this._activeDrag;
        this._activeDrag = false;
        if (!this.enabled || document.pointerLockElement) return;
        const selection = this._getSelection(e.offsetX, e.offsetY);
        if (selection[0] || wasDragging) {
            if (this.preventDefault) e.preventDefault();
            e.stopImmediatePropagation();
        }
        const { canvas } = this._device;
        canvas.releasePointerCapture(e.pointerId);
        this.fire(Gizmo.EVENT_POINTERUP, e.offsetX, e.offsetY, selection[0]);
    }

    _onPointerCancel(e) {
        this._activeDrag = false;
        const { canvas } = this._device;
        canvas.releasePointerCapture(e.pointerId);
        if (this.enabled) {
            this.fire(Gizmo.EVENT_POINTERUP, e.offsetX, e.offsetY, null);
        }
    }

    _updatePosition() {
        _gizmoPosition.set(0, 0, 0);
        if (this._coordSpace === 'local') {
            _gizmoPosition.copy(this.nodes[this.nodes.length - 1].getPosition());
        } else {
            for (let i = 0; i < this.nodes.length; i++) {
                _gizmoPosition.add(this.nodes[i].getPosition());
            }
            _gizmoPosition.mulScalar(1.0 / (this.nodes.length || 1));
        }
        if (_gizmoPosition.equalsApprox(this.root.getLocalPosition(), GIZMO_UPDATE_EPSILON)) return;
        this.root.setLocalPosition(_gizmoPosition);
        this.fire(Gizmo.EVENT_POSITIONUPDATE, _gizmoPosition);
        this._renderUpdate = true;
    }

    _updateRotation() {
        _gizmoRotation.set(0, 0, 0, 1);
        if (this._coordSpace === 'local' && this.nodes.length !== 0) {
            _gizmoRotation.copy(this.nodes[this.nodes.length - 1].getRotation());
        }
        if (_gizmoRotation.equalsApprox(this.root.getRotation(), GIZMO_UPDATE_EPSILON)) return;
        this.root.setRotation(_gizmoRotation);
        this.fire(Gizmo.EVENT_ROTATIONUPDATE, _gizmoRotation.getEulerAngles());
        this._renderUpdate = true;
    }

    _updateScale() {
        if (this._camera.projection === PROJECTION_PERSPECTIVE) {
            const gizmoPos = this.root.getLocalPosition();
            const cameraPos = this._camera.entity.getPosition();
            const dist = _gizmoV.sub2(gizmoPos, cameraPos).dot(this._camera.entity.forward);
            this._scale = Math.tan(0.5 * this._camera.fov * math.DEG_TO_RAD) * dist * GIZMO_PERS_SCALE_RATIO;
        } else {
            this._scale = this._camera.orthoHeight * GIZMO_ORTHO_SCALE_RATIO;
        }
        this._scale = Math.max(this._scale * this._size, GIZMO_MIN_SCALE);
        if (Math.abs(this._scale - this.root.getLocalScale().x) < GIZMO_UPDATE_EPSILON) return;
        this.root.setLocalScale(this._scale, this._scale, this._scale);
        this.fire(Gizmo.EVENT_SCALEUPDATE, this._scale);
        this._renderUpdate = true;
    }

    _getSelection(x, y) {
        const start = this._camera.screenToWorld(x, y, 0);
        const end = this._camera.screenToWorld(x, y, this._camera.farClip - this._camera.nearClip);
        const dir = _gizmoV.copy(end).sub(start).normalize();
        const selection = [];
        for (let i = 0; i < this.intersectShapes.length; i++) {
            const shape = this.intersectShapes[i];
            if (shape.disabled || !shape.entity.enabled) continue;
            const parentTM = shape.entity.getWorldTransform();
            for (let j = 0; j < shape.triData.length; j++) {
                const { tris, transform, priority } = shape.triData[j];
                const triWTM = _gizmoM1.copy(parentTM).mul(transform);
                const invTriWTM = _gizmoM2.copy(triWTM).invert();
                invTriWTM.transformPoint(start, _gizmoRay.origin);
                invTriWTM.transformVector(dir, _gizmoRay.direction);
                _gizmoRay.direction.normalize();
                for (let k = 0; k < tris.length; k++) {
                    if (tris[k].intersectsRay(_gizmoRay, _gizmoV)) {
                        selection.push({
                            dist: triWTM.transformPoint(_gizmoV).sub(start).length(),
                            meshInstances: shape.meshInstances,
                            priority
                        });
                    }
                }
            }
        }
        if (selection.length) {
            selection.sort((s0, s1) => {
                if (s0.priority !== 0 && s1.priority !== 0) return s1.priority - s0.priority;
                return s0.dist - s1.dist;
            });
            return selection[0].meshInstances;
        }
        return [];
    }

    attach(nodes = []) {
        if (Array.isArray(nodes)) {
            if (nodes.length === 0) return;
            this.nodes = nodes;
        } else {
            this.nodes = [nodes];
        }
        this._updatePosition();
        this._updateRotation();
        this._updateScale();
        this.fire(Gizmo.EVENT_NODESATTACH);
        this.enabled = true;
    }

    detach() {
        this.enabled = false;
        this._activeDrag = false;
        this.fire(Gizmo.EVENT_NODESDETACH);
        this.nodes = [];
    }

    prerender() {}

    update() {
        if (this._renderUpdate) {
            this._renderUpdate = false;
            this.fire(Gizmo.EVENT_RENDERUPDATE);
        }
        if (!this.enabled) return;
        this._updatePosition();
        this._updateRotation();
        this._updateScale();
    }

    destroy() {
        this.detach();
        this._device.canvas.removeEventListener('pointerdown', this._onPointerDown, true);
        this._device.canvas.removeEventListener('pointermove', this._onPointerMove, true);
        this._device.canvas.removeEventListener('pointerup', this._onPointerUp, true);
        this._device.canvas.removeEventListener('pointercancel', this._onPointerCancel, true);
        this._handles.forEach((handle) => handle.off());
        this.root.destroy();
    }
}
Gizmo.EVENT_POINTERDOWN = 'pointer:down';
Gizmo.EVENT_POINTERMOVE = 'pointer:move';
Gizmo.EVENT_POINTERUP = 'pointer:up';
Gizmo.EVENT_POSITIONUPDATE = 'position:update';
Gizmo.EVENT_ROTATIONUPDATE = 'rotation:update';
Gizmo.EVENT_SCALEUPDATE = 'scale:update';
Gizmo.EVENT_NODESATTACH = 'nodes:attach';
Gizmo.EVENT_NODESDETACH = 'nodes:detach';
Gizmo.EVENT_RENDERUPDATE = 'render:update';

// ---------------------------------------------------------------------------
// TransformGizmo - shared drag/hover/theme logic for transform gizmos.
// ---------------------------------------------------------------------------

const _tgV1 = new Vec3();
const _tgV2 = new Vec3();
const _tgPoint = new Vec3();
const _tgRay = new Ray();
const _tgPlane = new Plane();
const _tgColor = new Color();
const AXES_XYZ = ['x', 'y', 'z'];

class TransformGizmo extends Gizmo {
    constructor(camera, layer, name = 'gizmo:transform') {
        super(camera, layer, name);
        this._theme = {
            shapeBase: {
                x: COLOR_RED.clone(),
                y: COLOR_GREEN.clone(),
                z: COLOR_BLUE.clone(),
                xyz: new Color(0.8, 0.8, 0.8, 1),
                f: new Color(0.8, 0.8, 0.8, 1)
            },
            shapeHover: {
                x: new Color().lerp(COLOR_RED, Color.WHITE, 0.75),
                y: new Color().lerp(COLOR_GREEN, Color.WHITE, 0.75),
                z: new Color().lerp(COLOR_BLUE, Color.WHITE, 0.75),
                xyz: Color.WHITE.clone(),
                f: Color.WHITE.clone()
            },
            guideBase: {
                x: COLOR_RED.clone(),
                y: COLOR_GREEN.clone(),
                z: COLOR_BLUE.clone()
            },
            guideOcclusion: 0.8,
            disabled: COLOR_GRAY.clone()
        };
        this._rootStartPos = new Vec3();
        this._rootStartRot = new Quat();
        this._shapes = {};
        this._shapeMap = new Map();
        this._hovering = new Set();
        this._hoverAxis = '';
        this._hoverIsPlane = false;
        this._selectedAxis = '';
        this._selectedIsPlane = false;
        this._selectionStartPoint = new Vec3();
        this.snap = false;
        this.snapIncrement = 1;
        this.dragMode = 'selected';

        this.on(Gizmo.EVENT_POINTERDOWN, (x, y, meshInstance) => {
            const shape = this._shapeMap.get(meshInstance);
            if (shape?.disabled) return;
            if (this._dragging) return;
            if (!meshInstance) return;
            this._hoverAxis = '';
            this._hoverIsPlane = false;
            this._selectedAxis = this._getAxis(meshInstance);
            this._selectedIsPlane = this._getIsPlane(meshInstance);
            this._rootStartPos.copy(this.root.getLocalPosition());
            this._rootStartRot.copy(this.root.getRotation());
            const point = this._screenToPoint(x, y);
            this._selectionStartPoint.copy(point);
            this.fire(TransformGizmo.EVENT_TRANSFORMSTART, point, x, y);
        });
        this.on(Gizmo.EVENT_POINTERMOVE, (x, y, meshInstance) => {
            const shape = this._shapeMap.get(meshInstance);
            if (shape?.disabled) return;
            this._hover(meshInstance);
            if (!this._dragging) return;
            const point = this._screenToPoint(x, y);
            this.fire(TransformGizmo.EVENT_TRANSFORMMOVE, point, x, y);
        });
        this.on(Gizmo.EVENT_POINTERUP, (_x, _y, meshInstance) => {
            this._hover(meshInstance);
            if (!this._dragging) return;
            if (meshInstance) {
                this._hoverAxis = this._selectedAxis;
                this._hoverIsPlane = this._selectedIsPlane;
            }
            this._selectedAxis = '';
            this._selectedIsPlane = false;
            this.fire(TransformGizmo.EVENT_TRANSFORMEND);
        });
        this.on(Gizmo.EVENT_NODESDETACH, () => {
            this._hoverAxis = '';
            this._hoverIsPlane = false;
            this._hover();
            this.fire(Gizmo.EVENT_POINTERUP);
        });
    }

    get theme() {
        return this._theme;
    }

    get _dragging() {
        return !this._hoverAxis && !!this._selectedAxis;
    }

    _getAxis(meshInstance) {
        if (!meshInstance) return '';
        return meshInstance.node.name.split(':')[1];
    }

    _getIsPlane(meshInstance) {
        if (!meshInstance) return false;
        return meshInstance.node.name.indexOf('plane') !== -1;
    }

    _hover(meshInstance) {
        if (this._dragging) return;
        const remove = new Set(this._hovering);
        let changed = false;
        const add = (axis) => {
            if (remove.has(axis)) {
                remove.delete(axis);
            } else {
                this._hovering.add(axis);
                this._shapes[axis]?.hover(true);
                changed = true;
            }
        };
        this._hoverAxis = this._getAxis(meshInstance);
        this._hoverIsPlane = this._getIsPlane(meshInstance);
        if (this._hoverAxis) {
            if (this._hoverAxis === 'xyz') {
                add('x'); add('y'); add('z'); add('xyz');
            } else if (this._hoverIsPlane) {
                switch (this._hoverAxis) {
                    case 'x': add('y'); add('z'); add('yz'); break;
                    case 'y': add('x'); add('z'); add('xz'); break;
                    case 'z': add('x'); add('y'); add('xy'); break;
                }
            } else {
                add(this._hoverAxis);
            }
        }
        for (const axis of remove) {
            this._hovering.delete(axis);
            this._shapes[axis]?.hover(false);
            changed = true;
        }
        if (changed) this._renderUpdate = true;
    }

    _createRay(mouseWPos) {
        if (this._camera.projection === PROJECTION_PERSPECTIVE) {
            _tgRay.origin.copy(this._camera.entity.getPosition());
            _tgRay.direction.sub2(mouseWPos, _tgRay.origin).normalize();
            return _tgRay;
        }
        const orthoDepth = this._camera.farClip - this._camera.nearClip;
        _tgRay.origin.sub2(mouseWPos, _tgV1.copy(this._camera.entity.forward).mulScalar(orthoDepth));
        _tgRay.direction.copy(this._camera.entity.forward);
        return _tgRay;
    }

    _createPlane(axis, isFacing, isLine) {
        const facingDir = _tgV1.copy(this.facingDir);
        const normal = _tgPlane.normal.set(0, 0, 0);
        if (isFacing) {
            normal.copy(this._camera.entity.forward).mulScalar(-1);
        } else {
            normal[axis] = 1;
            this._rootStartRot.transformVector(normal, normal);
            if (isLine) {
                _tgV2.cross(normal, facingDir).normalize();
                normal.cross(_tgV2, normal).normalize();
            }
        }
        return _tgPlane.setFromPointNormal(this._rootStartPos, normal);
    }

    _dirFromAxis(axis, dir) {
        if (axis === 'f') {
            dir.copy(this._camera.entity.forward).mulScalar(-1);
        } else {
            dir.set(0, 0, 0);
            dir[axis] = 1;
        }
        return dir;
    }

    _projectToAxis(point, axis) {
        _tgV1.set(0, 0, 0);
        _tgV1[axis] = 1;
        point.copy(_tgV1.mulScalar(_tgV1.dot(point)));
        const v = point[axis];
        point.set(0, 0, 0);
        point[axis] = v;
    }

    _screenToPoint(x, y, isFacing = false, isLine = false) {
        const mouseWPos = this._camera.screenToWorld(x, y, 1);
        const axis = this._selectedAxis;
        const ray = this._createRay(mouseWPos);
        const plane = this._createPlane(axis, isFacing, isLine);
        plane.intersectsRay(ray, _tgPoint);
        return _tgPoint;
    }

    _drawGuideLines(pos, rot, activeAxis, activeIsPlane) {
        for (const axis of AXES_XYZ) {
            if (activeAxis === 'xyz') {
                this._drawSpanLine(pos, rot, axis);
                continue;
            }
            if (activeIsPlane) {
                if (axis !== activeAxis) {
                    this._drawSpanLine(pos, rot, axis);
                }
            } else {
                if (axis === activeAxis) {
                    this._drawSpanLine(pos, rot, axis);
                }
            }
        }
    }

    _drawSpanLine(pos, rot, axis) {
        const dir = this._dirFromAxis(axis, _tgV1);
        const base = this._theme.guideBase[axis];
        const from = _tgV1.copy(dir).mulScalar(this._camera.farClip - this._camera.nearClip);
        const to = _tgV2.copy(from).mulScalar(-1);
        rot.transformVector(from, from).add(pos);
        rot.transformVector(to, to).add(pos);
        if (this._theme.guideOcclusion < 1) {
            const occluded = _tgColor.copy(base);
            occluded.a *= 1 - this._theme.guideOcclusion;
            this._app.drawLine(from, to, occluded, false, this._layer);
        }
        if (base.a !== 0) {
            this._app.drawLine(from, to, base, true);
        }
    }

    _createTransform() {
        for (const key in this._shapes) {
            const shape = this._shapes[key];
            this.root.addChild(shape.entity);
            this.intersectShapes.push(shape);
            for (let i = 0; i < shape.meshInstances.length; i++) {
                this._shapeMap.set(shape.meshInstances[i], shape);
            }
        }
    }

    enableShape(shapeAxis, enabled) {
        const shape = this._shapes[shapeAxis];
        if (!shape) return;
        shape.disabled = !enabled;
    }

    isShapeEnabled(shapeAxis) {
        const shape = this._shapes[shapeAxis];
        if (!shape) return false;
        return !shape.disabled;
    }

    setTheme(partial) {
        const theme = { ...this._theme, ...partial };
        if (typeof theme !== 'object' || typeof theme.shapeBase !== 'object' ||
            typeof theme.shapeHover !== 'object' || typeof theme.guideBase !== 'object') {
            return;
        }
        for (const axis in theme.shapeBase) {
            if (theme.shapeBase[axis] instanceof Color) this._theme.shapeBase[axis].copy(theme.shapeBase[axis]);
        }
        for (const axis in theme.shapeHover) {
            if (theme.shapeHover[axis] instanceof Color) this._theme.shapeHover[axis].copy(theme.shapeHover[axis]);
        }
        for (const axis in theme.guideBase) {
            if (theme.guideBase[axis] instanceof Color) this._theme.guideBase[axis].copy(theme.guideBase[axis]);
        }
        if (typeof theme.guideOcclusion === 'number') {
            this._theme.guideOcclusion = math.clamp(theme.guideOcclusion, 0, 1);
        }
        if (theme.disabled instanceof Color) this._theme.disabled.copy(theme.disabled);
        for (const name in this._shapes) {
            this._shapes[name].hover(!!this._hoverAxis);
        }
    }

    prerender() {
        super.prerender();
        if (!this.enabled) return;
        const gizmoPos = this.root.getLocalPosition();
        const gizmoRot = this.root.getRotation();
        const activeAxis = this._hoverAxis || this._selectedAxis;
        const activeIsPlane = this._hoverIsPlane || this._selectedIsPlane;
        this._drawGuideLines(gizmoPos, gizmoRot, activeAxis, activeIsPlane);
    }

    destroy() {
        super.destroy();
        for (const key in this._shapes) {
            this._shapes[key].destroy();
        }
    }
}
TransformGizmo.EVENT_TRANSFORMSTART = 'transform:start';
TransformGizmo.EVENT_TRANSFORMMOVE = 'transform:move';
TransformGizmo.EVENT_TRANSFORMEND = 'transform:end';

// ---------------------------------------------------------------------------
// Shape helpers used by TranslateGizmo: sphere (center), plane, arrow.
// ---------------------------------------------------------------------------

class SphereShape extends Shape {
    constructor(device, args = {}) {
        super(device, 'sphereCenter', args);
        this._radius = 0.03;
        this._radius = args.radius ?? this._radius;
        this.triData = [new TriData(new SphereGeometry(), 2)];
        this._createRenderComponent(this.entity, [
            Mesh.fromGeometry(this.device, new SphereGeometry({ latitudeBands: 32, longitudeBands: 32 }))
        ]);
        this._update();
    }

    set radius(value) {
        this._radius = value ?? this._radius;
        this._update();
    }

    get radius() {
        return this._radius;
    }

    _update() {
        const scale = this._radius * 2;
        this.entity.setLocalScale(scale, scale, scale);
    }
}

class PlaneShape extends Shape {
    constructor(device, args = {}) {
        super(device, 'plane', args);
        this._cull = CULLFACE_NONE;
        this._size = 0.16;
        this._gap = 0;
        this._flipped = new Vec3();
        this._size = args.size ?? this._size;
        this._gap = args.gap ?? this._gap;
        this.triData = [new TriData(new PlaneGeometry())];
        this._createRenderComponent(this.entity, [Mesh.fromGeometry(this.device, new PlaneGeometry())]);
        this._update();
    }

    set size(value) {
        this._size = value ?? this._size;
        this._update();
    }

    get size() {
        return this._size;
    }

    set gap(value) {
        this._gap = value ?? this._gap;
        this._update();
    }

    get gap() {
        return this._gap;
    }

    set flipped(value) {
        if (this._flipped.distance(value) < GIZMO_UPDATE_EPSILON) return;
        this._flipped.copy(value);
        this._update();
    }

    get flipped() {
        return this._flipped;
    }

    _update() {
        const offset = this._size / 2 + this._gap;
        this._position.set(
            this._flipped.x ? -offset : offset,
            this._flipped.y ? -offset : offset,
            this._flipped.z ? -offset : offset
        );
        this._position[this.axis] = 0;
        this.entity.setLocalPosition(this._position);
        this.entity.setLocalEulerAngles(this._rotation);
        this.entity.setLocalScale(this._size, this._size, this._size);
    }
}

class ArrowShape extends Shape {
    constructor(device, args = {}) {
        super(device, 'arrow', args);
        this._gap = 0;
        this._lineThickness = 0.02;
        this._lineLength = 0.5;
        this._arrowThickness = 0.12;
        this._arrowLength = 0.18;
        this._tolerance = 0.1;
        this._gap = args.gap ?? this._gap;
        this._lineThickness = args.lineThickness ?? this._lineThickness;
        this._lineLength = args.lineLength ?? this._lineLength;
        this._arrowThickness = args.arrowThickness ?? this._arrowThickness;
        this._arrowLength = args.arrowLength ?? this._arrowLength;
        this._tolerance = args.tolerance ?? this._tolerance;

        this.triData = [new TriData(new ConeGeometry()), new TriData(new CylinderGeometry(), 1)];

        this._head = new Entity(`head:${this.axis}`);
        this.entity.addChild(this._head);
        this._createRenderComponent(this._head, [Mesh.fromGeometry(this.device, new ConeGeometry())]);

        this._line = new Entity(`line:${this.axis}`);
        this.entity.addChild(this._line);
        this._createRenderComponent(this._line, [Mesh.fromGeometry(this.device, new CylinderGeometry())]);

        this._update();
    }

    set gap(value) {
        this._gap = value ?? this._gap;
        this._update();
    }

    get gap() {
        return this._gap;
    }

    set lineThickness(value) {
        this._lineThickness = value ?? this._lineThickness;
        this._update();
    }

    get lineThickness() {
        return this._lineThickness;
    }

    set lineLength(value) {
        this._lineLength = value ?? this._lineLength;
        this._update();
    }

    get lineLength() {
        return this._lineLength;
    }

    set arrowThickness(value) {
        this._arrowThickness = value ?? this._arrowThickness;
        this._update();
    }

    get arrowThickness() {
        return this._arrowThickness;
    }

    set arrowLength(value) {
        this._arrowLength = value ?? this._arrowLength;
        this._update();
    }

    get arrowLength() {
        return this._arrowLength;
    }

    set tolerance(value) {
        this._tolerance = value;
        this._update();
    }

    get tolerance() {
        return this._tolerance;
    }

    _update() {
        _tgV1.set(0, this._gap + this._arrowLength * 0.5 + this._lineLength, 0);
        this.triData[0].setTransform(_tgV1, Quat.IDENTITY, _tgV2.set(this._arrowThickness, this._arrowLength, this._arrowThickness));
        _tgV1.set(0, this._gap + this._lineLength * 0.5, 0);
        this.triData[1].setTransform(_tgV1, Quat.IDENTITY, _tgV2.set(this._lineThickness + this._tolerance, this._lineLength, this._lineThickness + this._tolerance));

        this._head.setLocalPosition(0, this._gap + this._arrowLength * 0.5 + this._lineLength, 0);
        this._head.setLocalScale(this._arrowThickness, this._arrowLength, this._arrowThickness);
        this._line.setLocalPosition(0, this._gap + this._lineLength * 0.5, 0);
        this._line.setLocalScale(this._lineThickness, this._lineLength, this._lineThickness);
    }
}

// ---------------------------------------------------------------------------
// TranslateGizmo - arrows + planes + center sphere for dragging a point.
// ---------------------------------------------------------------------------

const _tlV1 = new Vec3();
const _tlV2 = new Vec3();
const _tlPoint = new Vec3();
const _tlDelta = new Vec3();
const _tlQ = new Quat();
const TRANSLATE_GLANCE_EPSILON = 0.01;

class TranslateGizmo extends TransformGizmo {
    constructor(camera, layer) {
        super(camera, layer, 'gizmo:translate');

        this._shapes = {
            xyz: new SphereShape(this._device, {
                axis: 'xyz', layers: [this._layer.id],
                defaultColor: this._theme.shapeBase.xyz, hoverColor: this._theme.shapeHover.xyz, disabledColor: this._theme.disabled
            }),
            yz: new PlaneShape(this._device, {
                axis: 'x', layers: [this._layer.id], rotation: new Vec3(0, 0, -90),
                defaultColor: this._theme.shapeBase.x, hoverColor: this._theme.shapeHover.x, disabledColor: this._theme.disabled, depth: 1
            }),
            xz: new PlaneShape(this._device, {
                axis: 'y', layers: [this._layer.id], rotation: new Vec3(0, 0, 0),
                defaultColor: this._theme.shapeBase.y, hoverColor: this._theme.shapeHover.y, disabledColor: this._theme.disabled, depth: 1
            }),
            xy: new PlaneShape(this._device, {
                axis: 'z', layers: [this._layer.id], rotation: new Vec3(90, 0, 0),
                defaultColor: this._theme.shapeBase.z, hoverColor: this._theme.shapeHover.z, disabledColor: this._theme.disabled, depth: 1
            }),
            x: new ArrowShape(this._device, {
                axis: 'x', layers: [this._layer.id], rotation: new Vec3(0, 0, -90),
                defaultColor: this._theme.shapeBase.x, hoverColor: this._theme.shapeHover.x, disabledColor: this._theme.disabled
            }),
            y: new ArrowShape(this._device, {
                axis: 'y', layers: [this._layer.id], rotation: new Vec3(0, 0, 0),
                defaultColor: this._theme.shapeBase.y, hoverColor: this._theme.shapeHover.y, disabledColor: this._theme.disabled
            }),
            z: new ArrowShape(this._device, {
                axis: 'z', layers: [this._layer.id], rotation: new Vec3(90, 0, 0),
                defaultColor: this._theme.shapeBase.z, hoverColor: this._theme.shapeHover.z, disabledColor: this._theme.disabled
            })
        };
        this._nodeLocalPositions = new Map();
        this._nodePositions = new Map();
        this.snapIncrement = 1;
        this.flipPlanes = true;

        this._createTransform();

        this.on(TransformGizmo.EVENT_TRANSFORMSTART, () => {
            this._storeNodePositions();
            this._drag(true);
        });
        this.on(TransformGizmo.EVENT_TRANSFORMMOVE, (point) => {
            const translateDelta = _tlDelta.copy(point).sub(this._selectionStartPoint);
            if (this.snap) {
                translateDelta.mulScalar(1 / this.snapIncrement);
                translateDelta.round();
                translateDelta.mulScalar(this.snapIncrement);
            }
            this._setNodePositions(translateDelta);
        });
        this.on(TransformGizmo.EVENT_TRANSFORMEND, () => {
            this._drag(false);
        });
        this.on(Gizmo.EVENT_NODESDETACH, () => {
            this._nodeLocalPositions.clear();
            this._nodePositions.clear();
        });
    }

    _shapesLookAtCamera() {
        const cameraDir = this.cameraDir;
        let changed = false;
        let dot, enabled;

        dot = cameraDir.dot(this.root.right);
        enabled = 1 - Math.abs(dot) > TRANSLATE_GLANCE_EPSILON;
        if (this._shapes.x.entity.enabled !== enabled) { this._shapes.x.entity.enabled = enabled; changed = true; }

        dot = cameraDir.dot(this.root.up);
        enabled = 1 - Math.abs(dot) > TRANSLATE_GLANCE_EPSILON;
        if (this._shapes.y.entity.enabled !== enabled) { this._shapes.y.entity.enabled = enabled; changed = true; }

        dot = cameraDir.dot(this.root.forward);
        enabled = 1 - Math.abs(dot) > TRANSLATE_GLANCE_EPSILON;
        if (this._shapes.z.entity.enabled !== enabled) { this._shapes.z.entity.enabled = enabled; changed = true; }

        let flipped;
        _tlV1.cross(cameraDir, this.root.right);
        enabled = 1 - _tlV1.length() > TRANSLATE_GLANCE_EPSILON;
        if (this._shapes.yz.entity.enabled !== enabled) { this._shapes.yz.entity.enabled = enabled; changed = true; }
        flipped = this.flipPlanes ? _tlV2.set(0, +(_tlV1.dot(this.root.forward) < 0), +(_tlV1.dot(this.root.up) < 0)) : Vec3.ZERO;
        if (!this._shapes.yz.flipped.equals(flipped)) { this._shapes.yz.flipped = flipped; changed = true; }

        _tlV1.cross(cameraDir, this.root.forward);
        enabled = 1 - _tlV1.length() > TRANSLATE_GLANCE_EPSILON;
        if (this._shapes.xy.entity.enabled !== enabled) { this._shapes.xy.entity.enabled = enabled; changed = true; }
        flipped = this.flipPlanes ? _tlV2.set(+(_tlV1.dot(this.root.up) < 0), +(_tlV1.dot(this.root.right) > 0), 0) : Vec3.ZERO;
        if (!this._shapes.xy.flipped.equals(flipped)) { this._shapes.xy.flipped = flipped; changed = true; }

        _tlV1.cross(cameraDir, this.root.up);
        enabled = 1 - _tlV1.length() > TRANSLATE_GLANCE_EPSILON;
        if (this._shapes.xz.entity.enabled !== enabled) { this._shapes.xz.entity.enabled = enabled; changed = true; }
        flipped = this.flipPlanes ? _tlV2.set(+(_tlV1.dot(this.root.forward) > 0), 0, +(_tlV1.dot(this.root.right) > 0)) : Vec3.ZERO;
        if (!this._shapes.xz.flipped.equals(flipped)) { this._shapes.xz.flipped = flipped; changed = true; }

        if (changed) this._renderUpdate = true;
    }

    _drag(state) {
        for (const axis in this._shapes) {
            const shape = this._shapes[axis];
            switch (this.dragMode) {
                case 'show':
                    continue;
                case 'hide':
                    shape.visible = !state;
                    continue;
                case 'selected':
                    if (this._selectedAxis === 'xyz') {
                        shape.visible = state ? axis.length === 1 : true;
                        continue;
                    }
                    if (this._selectedIsPlane) {
                        shape.visible = state ? (axis.length === 1 && !axis.includes(this._selectedAxis)) : true;
                        continue;
                    }
                    shape.visible = state ? axis === this._selectedAxis : true;
            }
        }
        this._renderUpdate = true;
    }

    _storeNodePositions() {
        for (let i = 0; i < this.nodes.length; i++) {
            const node = this.nodes[i];
            this._nodeLocalPositions.set(node, node.getLocalPosition().clone());
            this._nodePositions.set(node, node.getPosition().clone());
        }
    }

    _setNodePositions(translateDelta) {
        for (let i = 0; i < this.nodes.length; i++) {
            const node = this.nodes[i];
            if (this._coordSpace === 'local') {
                const pos = this._nodeLocalPositions.get(node);
                if (!pos) continue;
                _tlV1.copy(translateDelta);
                node.parent?.getWorldTransform().getScale(_tlV2);
                _tlV2.x = 1 / _tlV2.x; _tlV2.y = 1 / _tlV2.y; _tlV2.z = 1 / _tlV2.z;
                _tlQ.copy(node.getLocalRotation()).transformVector(_tlV1, _tlV1);
                _tlV1.mul(_tlV2);
                node.setLocalPosition(_tlV1.add(pos));
            } else {
                const pos = this._nodePositions.get(node);
                if (!pos) continue;
                node.setPosition(_tlV1.copy(translateDelta).add(pos));
            }
        }
        this._updatePosition();
    }

    _screenToPoint(x, y) {
        const mouseWPos = this._camera.screenToWorld(x, y, 1);
        const axis = this._selectedAxis;
        const isPlane = this._selectedIsPlane;
        const ray = this._createRay(mouseWPos);
        const plane = this._createPlane(axis, axis === 'xyz', !isPlane);
        if (!plane.intersectsRay(ray, _tlPoint)) {
            return _tlPoint;
        }
        _tlQ.copy(this._rootStartRot).invert().transformVector(_tlPoint, _tlPoint);
        if (!isPlane && axis !== 'xyz') {
            this._projectToAxis(_tlPoint, axis);
        }
        return _tlPoint;
    }

    // While actively dragging any axis, show guide lines for all three axes
    // (not just the one being dragged) so the full 3D reference is visible.
    _drawGuideLines(pos, rot, activeAxis, activeIsPlane) {
        for (const axis of AXES_XYZ) {
            if (this._dragging || activeAxis === 'xyz') {
                this._drawSpanLine(pos, rot, axis);
                continue;
            }
            if (activeIsPlane) {
                if (axis !== activeAxis) {
                    this._drawSpanLine(pos, rot, axis);
                }
            } else {
                if (axis === activeAxis) {
                    this._drawSpanLine(pos, rot, axis);
                }
            }
        }
    }

    prerender() {
        super.prerender();
        if (!this.enabled) return;
        this._shapesLookAtCamera();
    }
}

export { Gizmo, TranslateGizmo };
