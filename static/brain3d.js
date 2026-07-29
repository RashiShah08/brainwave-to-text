/**
 * The page as an anatomical plate.
 *
 * This is drawn the way Santiago Ramón y Cajal drew cortex in the 1890s: iron-
 * gall ink on laid paper, somata as filled discs, processes as tapered strokes,
 * shading by stipple. Nothing glows and nothing is additively blended — on
 * paper, "brighter" means *more ink*, so activity darkens and thickens rather
 * than lighting up.
 *
 * Two consequences worth stating, because they drive every choice below:
 *
 *   - Depth is aerial perspective, not occlusion. Distant strokes carry less
 *     ink, exactly as they would in a drawing, which is why every shader takes
 *     a fog term and why nothing writes to the depth buffer.
 *   - Strokes wobble. A perfectly straight line between two electrodes reads as
 *     a computer plot; a stroke that bows slightly and tapers at both ends
 *     reads as a pen. The wobble is deterministic per edge, so the plate is the
 *     same drawing every time it loads.
 *
 * The data underneath is unchanged and real: 64 contacts at their true montage
 * coordinates, ink weight driven by measured mu/beta power, and a committed
 * decision washing the responding hemisphere in oxblood.
 */

import * as THREE from './three.module.min.js';

const INK = new THREE.Color('#2a2018');      // iron gall, warm near-black
const SEPIA = new THREE.Color('#6d5333');    // faded ink, construction lines
const OXBLOOD = new THREE.Color('#8f3320');  // the accent, used sparingly
const WASH = new THREE.Color('#3f3324');     // mid-tone for stipple

/** Software rasterisers cannot hold a full-viewport multisampled buffer. */
function isSoftwareGL() {
  try {
    const gl = document.createElement('canvas').getContext('webgl');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const name = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
    return /swiftshader|llvmpipe|software|basic render|microsoft/i.test(name);
  } catch {
    return true;
  }
}

/** A pen dot: solid core out to `hard`, then a quick falloff. Not a glow. */
function inkDot(hard) {
  const s = 64;
  const cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const ctx = cv.getContext('2d');
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(hard, 'rgba(255,255,255,0.95)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const t = new THREE.CanvasTexture(cv);
  t.needsUpdate = true;
  return t;
}

/** Distance fade, shared by every material so the whole plate recedes alike. */
const FOG = `
  uniform float uNear;
  uniform float uFar;
  float aerial(float viewZ) {
    return clamp((uFar + viewZ) / (uFar - uNear), 0.0, 1.0);
  }`;

const STROKE_VERT = `
  ${FOG}
  attribute float aInk;
  varying float vInk;
  varying vec3 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vInk = aInk * aerial(mv.z);
    vColor = color;
    gl_Position = projectionMatrix * mv;
  }`;

const STROKE_FRAG = `
  varying float vInk;
  varying vec3 vColor;
  void main() {
    if (vInk < 0.004) discard;
    gl_FragColor = vec4(vColor, vInk);
  }`;

const DOT_VERT = `
  ${FOG}
  uniform float uScale;
  attribute float aSize;
  attribute float aInk;
  varying float vInk;
  varying vec3 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vInk = aInk * aerial(mv.z);
    vColor = color;
    gl_PointSize = aSize * uScale / max(0.15, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const DOT_FRAG = `
  uniform sampler2D uMap;
  varying float vInk;
  varying vec3 vColor;
  void main() {
    float a = texture2D(uMap, gl_PointCoord).a * vInk;
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor, a);
  }`;

/** Deterministic hash so the plate redraws identically on every load. */
function hash(n) {
  const x = Math.sin(n * 127.1) * 43758.5453;
  return x - Math.floor(x);
}

export class NeuralEnvironment {
  constructor(canvas, geometry) {
    this.canvas = canvas;
    this.electrodes = geometry.electrodes;
    this.n = this.electrodes.length;
    this.power = new Float32Array(this.n).fill(0.42);
    this.target = new Float32Array(this.n).fill(0.42);
    this.vis = new Float32Array(this.n).fill(0.42);
    this.flashAmount = 0;
    this.flashSide = null;

    this.scroll = 0;
    this.scrollEased = 0;
    this.yaw = -0.55;
    this.pitch = 0.10;
    this.drag = { active: false, x: 0, y: 0 };
    this.pointer = { x: 0, y: 0 };

    this.clock = new THREE.Clock();
    this._col = new THREE.Color();
    this.soft = isSoftwareGL();

    this.uNear = { value: 0.6 };
    this.uFar = { value: 6.5 };

    this._renderer();
    this._scene();
    this._contours();
    this._processes();
    this._somata();
    this._stipple();
    this._beads();
    this._spatter();
    this._input();

    addEventListener('resize', () => this._resize(), { passive: true });
    addEventListener('scroll', () => this._onScroll(), { passive: true });
    this._resize();
    this._onScroll();
    this._loop();
  }

  _renderer() {
    this.r = new THREE.WebGLRenderer({
      canvas: this.canvas, antialias: !this.soft, alpha: true,
      powerPreference: 'high-performance',
    });
    this.r.setPixelRatio(this.soft ? 1 : Math.min(devicePixelRatio, 2));
    this.r.setClearColor(0x000000, 0);
    this.canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      document.documentElement.dataset.glLost = '1';
    });
    this.canvas.addEventListener('webglcontextrestored', () => {
      delete document.documentElement.dataset.glLost;
    });
  }

  _scene() {
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(46, 1, 0.1, 60);
    this.rig = new THREE.Group();
    this.scene.add(this.rig);
    // Crisp cores: a soft falloff at these sizes reads as a smudge, not a nib.
    this.somaTex = inkDot(0.74);
    this.speckTex = inkDot(0.52);
  }

  _strokeMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: { uNear: this.uNear, uFar: this.uFar },
      vertexShader: STROKE_VERT, fragmentShader: STROKE_FRAG,
      transparent: true, depthWrite: false, vertexColors: true,
    });
  }

  _dotMaterial(map) {
    return new THREE.ShaderMaterial({
      uniforms: { uNear: this.uNear, uFar: this.uFar, uMap: { value: map }, uScale: { value: 340 } },
      vertexShader: DOT_VERT, fragmentShader: DOT_FRAG,
      transparent: true, depthWrite: false, vertexColors: true,
    });
  }

  /**
   * Construction lines — the faint sepia arcs an anatomist rules before
   * inking. Three great circles, wobbled so they read as drawn, not plotted.
   */
  _contours() {
    const pos = [];
    const col = [];
    const ink = [];
    const planes = [
      (a) => [Math.sin(a) * 1.06, Math.cos(a) * 1.15, 0],           // coronal
      (a) => [0, Math.cos(a) * 1.15, Math.sin(a) * 1.22],           // sagittal
      (a) => [Math.sin(a) * 1.06, 0.12, Math.cos(a) * 1.22],        // axial
    ];
    planes.forEach((f, p) => {
      const N = 150;
      let prev = null;
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2;
        const w = 1 + 0.011 * Math.sin(a * 7 + p * 2.3) + 0.007 * Math.sin(a * 13 + p);
        const [x, y, z] = f(a);
        const cur = [x * w, y * w, z * w];
        if (prev) {
          pos.push(...prev, ...cur);
          for (let k = 0; k < 2; k++) {
            col.push(SEPIA.r, SEPIA.g, SEPIA.b);
            // Broken line: the pen lifts, the way a ruled guide does.
            ink.push(0.19 * (0.4 + 0.6 * Math.abs(Math.sin(a * 9 + p))));
          }
        }
        prev = cur;
      }
    });
    this.contours = this._lines(pos, col, ink);
    this.rig.add(this.contours);
  }

  _lines(pos, col, ink) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.Float32BufferAttribute(ink, 1));
    return new THREE.LineSegments(g, this._strokeMaterial());
  }

  /**
   * Processes between contacts. Each is a bowed, tapered stroke built from
   * `SEG` sub-segments; `strokes[k]` records which vertices belong to edge k
   * and the taper at each, so the loop can re-ink a whole stroke from the
   * power at its two ends without recomputing geometry.
   */
  _processes() {
    const SEG = 7;
    const pairs = [];
    for (let i = 0; i < this.n; i++) {
      const a = this.electrodes[i];
      for (let j = i + 1; j < this.n; j++) {
        const b = this.electrodes[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (d < 0.52) pairs.push([i, j, 1]);
        else if (d < 1.15 && (i * 31 + j * 17) % 13 === 0) pairs.push([i, j, 0.42]);
      }
    }
    this.strokes = [];
    this.paths = pairs.map(([i, j]) => [i, j]);

    const pos = [];
    const col = [];
    const ink = [];
    const up = new THREE.Vector3(0, 1, 0);
    const A = new THREE.Vector3();
    const B = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const perp = new THREE.Vector3();
    const cur = new THREE.Vector3();

    pairs.forEach(([i, j, weight], k) => {
      const a = this.electrodes[i];
      const b = this.electrodes[j];
      A.set(a.x, a.y, a.z);
      B.set(b.x, b.y, b.z);
      dir.subVectors(B, A);
      perp.crossVectors(dir, up).normalize();
      if (!isFinite(perp.x)) perp.set(1, 0, 0);

      const bow = (hash(k) - 0.5) * 0.09 + 0.02;
      const verts = [];
      for (let s = 0; s <= SEG; s++) {
        const t = s / SEG;
        const swell = Math.sin(t * Math.PI);
        cur.copy(A).addScaledVector(dir, t)
          .addScaledVector(perp, bow * swell)
          .multiplyScalar(1 + 0.012 * swell);
        verts.push([cur.x, cur.y, cur.z, swell]);
      }

      const vertexIds = [];
      for (let s = 0; s < SEG; s++) {
        for (const v of [verts[s], verts[s + 1]]) {
          vertexIds.push({ index: pos.length / 3, taper: 0.35 + 0.65 * v[3] });
          pos.push(v[0], v[1], v[2]);
          col.push(INK.r, INK.g, INK.b);
          ink.push(0.3);
        }
      }
      this.strokes.push({ i, j, weight, vertexIds });
    });

    this.processes = this._lines(pos, col, ink);
    this.rig.add(this.processes);
  }

  /** Cell bodies: filled ink discs, the darkest thing on the plate. */
  _somata() {
    const pos = new Float32Array(this.n * 3);
    const col = new Float32Array(this.n * 3);
    const size = new Float32Array(this.n);
    const ink = new Float32Array(this.n).fill(1);
    for (let i = 0; i < this.n; i++) {
      const e = this.electrodes[i];
      pos[i * 3] = e.x; pos[i * 3 + 1] = e.y; pos[i * 3 + 2] = e.z;
      col[i * 3] = INK.r; col[i * 3 + 1] = INK.g; col[i * 3 + 2] = INK.b;
      size[i] = 0.4;
    }
    this.somata = this._points(pos, col, size, ink, this.somaTex);
    this.rig.add(this.somata);
  }

  _points(pos, col, size, ink, map) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));
    return new THREE.Points(g, this._dotMaterial(map));
  }

  /**
   * Stipple. Cajal shaded by dotting, and so does this: each speck belongs to
   * one contact and takes its ink from that contact's power, so an active
   * region visibly darkens instead of changing hue.
   */
  _stipple() {
    const per = this.soft ? 14 : 34;
    const n = this.n * per;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const ink = new Float32Array(n);
    this.stippleOwner = new Int32Array(n);
    this.stippleBias = new Float32Array(n);

    for (let i = 0; i < this.n; i++) {
      const e = this.electrodes[i];
      for (let s = 0; s < per; s++) {
        const k = i * per + s;
        const h1 = hash(k * 1.7);
        const h2 = hash(k * 3.1 + 11);
        const h3 = hash(k * 5.9 + 23);
        // Cluster tightly around the soma and thin out with distance, the way
        // stipple shading falls off.
        const rad = 0.055 + Math.pow(h1, 1.7) * 0.20;
        const th = h2 * Math.PI * 2;
        const ph = Math.acos(2 * h3 - 1);
        pos[k * 3] = e.x + rad * Math.sin(ph) * Math.cos(th);
        pos[k * 3 + 1] = e.y + rad * Math.cos(ph);
        pos[k * 3 + 2] = e.z + rad * Math.sin(ph) * Math.sin(th);
        col[k * 3] = WASH.r; col[k * 3 + 1] = WASH.g; col[k * 3 + 2] = WASH.b;
        size[k] = 0.055 + h2 * 0.05;
        ink[k] = 0.2;
        this.stippleOwner[k] = i;
        this.stippleBias[k] = 1 - Math.pow(h1, 1.7);   // near specks ink first
      }
    }
    this.stipple = this._points(pos, col, size, ink, this.speckTex);
    this.rig.add(this.stipple);
  }

  /** Travelling ink beads — signal moving along a process. */
  _beads() {
    this.nBead = Math.min(this.soft ? 55 : 110, this.paths.length);
    this.bead = [];
    const pos = new Float32Array(this.nBead * 3);
    const col = new Float32Array(this.nBead * 3);
    const size = new Float32Array(this.nBead);
    const ink = new Float32Array(this.nBead);
    for (let k = 0; k < this.nBead; k++) {
      this.bead.push({
        path: Math.floor(hash(k * 7.3) * this.paths.length),
        t: hash(k * 2.9),
        speed: 0.16 + hash(k * 4.1) * 0.34,
      });
      size[k] = 0.10 + hash(k * 6.7) * 0.06;
      ink[k] = 0.6;
    }
    this.beads = this._points(pos, col, size, ink, this.somaTex);
    this.rig.add(this.beads);
  }

  /** Sparse spatter, the way ink flecks a plate. Pure texture, no meaning. */
  _spatter() {
    const n = this.soft ? 120 : 300;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const ink = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = 2.2 + Math.pow(hash(i * 1.31), 0.7) * 9;
      const th = hash(i * 2.71) * Math.PI * 2;
      const ph = Math.acos(2 * hash(i * 3.77) - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
      pos[i * 3 + 1] = r * Math.cos(ph) * 0.8;
      pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      col[i * 3] = SEPIA.r; col[i * 3 + 1] = SEPIA.g; col[i * 3 + 2] = SEPIA.b;
      size[i] = 0.03 + hash(i * 5.11) * 0.05;
      ink[i] = 0.10 + hash(i * 7.13) * 0.22;
    }
    this.spatter = this._points(pos, col, size, ink, this.speckTex);
    this.scene.add(this.spatter);
  }

  _input() {
    addEventListener('pointermove', (e) => {
      this.pointer.x = (e.clientX / innerWidth) * 2 - 1;
      this.pointer.y = (e.clientY / innerHeight) * 2 - 1;
      if (!this.drag.active) return;
      this.yaw += (e.clientX - this.drag.x) * 0.005;
      this.pitch = Math.max(-0.9, Math.min(0.9, this.pitch + (e.clientY - this.drag.y) * 0.004));
      this.drag.x = e.clientX; this.drag.y = e.clientY;
    }, { passive: true });

    addEventListener('pointerdown', (e) => {
      if (e.target.closest('a, button, select, input, label, table')) return;
      this.drag = { active: true, x: e.clientX, y: e.clientY };
      document.body.classList.add('grabbing');
    });
    const stop = () => {
      this.drag.active = false;
      document.body.classList.remove('grabbing');
    };
    addEventListener('pointerup', stop);
    addEventListener('pointercancel', stop);
  }

  _onScroll() {
    const max = Math.max(1, document.body.scrollHeight - innerHeight);
    this.scroll = Math.min(1, Math.max(0, scrollY / max));
  }

  _resize() {
    const w = innerWidth;
    const h = innerHeight;
    this.r.setSize(w, h, false);
    this.cam.aspect = w / h;
    this.cam.updateProjectionMatrix();
    // Sets the pen. A soma lands around 6–16 px at reading distance; stipple
    // specks stay at 2–3 px so they read as dots rather than overlapping into
    // a wash.
    const s = Math.max(64, h * 0.10);
    for (const p of [this.somata, this.stipple, this.beads, this.spatter]) {
      p.material.uniforms.uScale.value = s;
    }
  }

  // -- public ---------------------------------------------------------------

  update(bandPower) {
    if (!bandPower || bandPower.length !== this.n) return;
    for (let i = 0; i < this.n; i++) this.target[i] = bandPower[i];
  }

  flash(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('right')) this.flashSide = 'left_motor';
    else if (l.includes('left')) this.flashSide = 'right_motor';
    else this.flashSide = 'midline_motor';
    this.flashAmount = 1;
  }

  reset() {
    this.power.fill(0.42);
    this.target.fill(0.42);
    this.flashAmount = 0;
  }

  // -- loop -----------------------------------------------------------------

  _loop() {
    requestAnimationFrame(() => this._loop());
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    this.scrollEased += (this.scroll - this.scrollEased) * (1 - Math.exp(-dt * 4));
    const s = this.scrollEased;

    if (!this.drag.active) this.yaw += dt * 0.05;
    this.rig.position.x = 0.40 * (1 - s);
    this.rig.rotation.set(this.pitch + s * 0.35, this.yaw, 0);
    this.spatter.rotation.y = -this.yaw * 0.2;

    this.cam.position.set(
      this.pointer.x * 0.20,
      0.25 + s * 0.55 - this.pointer.y * 0.14,
      3.15 - s * 1.55,
    );
    this.cam.lookAt(0, s * 0.12, 0);

    const ease = 1 - Math.exp(-dt * 7);
    this.flashAmount = Math.max(0, this.flashAmount - dt * 1.1);

    // -- somata + stipple ---------------------------------------------------
    const scol = this.somata.geometry.attributes.color.array;
    const ssize = this.somata.geometry.attributes.aSize.array;
    const sink = this.somata.geometry.attributes.aInk.array;

    for (let i = 0; i < this.n; i++) {
      this.power[i] += (this.target[i] - this.power[i]) * ease;
      const e = this.electrodes[i];
      const v = Math.min(1, Math.max(0,
        this.power[i] + 0.10 * Math.sin(t * 0.8 + e.y * 3.1 + e.x * 2.2)));
      this.vis[i] = v;

      const lit = this.flashAmount > 0 && e.region === this.flashSide
        ? this.flashAmount : 0;
      this._col.copy(INK).lerp(OXBLOOD, lit * 0.9);

      const k = i * 3;
      scol[k] = this._col.r; scol[k + 1] = this._col.g; scol[k + 2] = this._col.b;
      ssize[i] = 0.20 + v * 0.34 + lit * 0.30;
      sink[i] = 0.42 + v * 0.58;
    }

    const pink = this.stipple.geometry.attributes.aInk.array;
    const pcol = this.stipple.geometry.attributes.color.array;
    for (let k = 0; k < this.stippleOwner.length; k++) {
      const owner = this.stippleOwner[k];
      const v = this.vis[owner];
      const lit = this.flashAmount > 0
        && this.electrodes[owner].region === this.flashSide ? this.flashAmount : 0;
      pink[k] = Math.max(0, (v * 1.5 - 0.34) * this.stippleBias[k] + lit * 0.55);
      if (lit > 0) {
        this._col.copy(WASH).lerp(OXBLOOD, lit * 0.8);
        pcol[k * 3] = this._col.r;
        pcol[k * 3 + 1] = this._col.g;
        pcol[k * 3 + 2] = this._col.b;
      } else {
        pcol[k * 3] = WASH.r; pcol[k * 3 + 1] = WASH.g; pcol[k * 3 + 2] = WASH.b;
      }
    }

    // -- processes ----------------------------------------------------------
    const lcol = this.processes.geometry.attributes.color.array;
    const link = this.processes.geometry.attributes.aInk.array;
    for (const st of this.strokes) {
      const v = (this.vis[st.i] + this.vis[st.j]) * 0.5;
      const lit = this.flashAmount > 0
        && (this.electrodes[st.i].region === this.flashSide
          || this.electrodes[st.j].region === this.flashSide)
        ? this.flashAmount : 0;
      this._col.copy(INK).lerp(OXBLOOD, lit * 0.85);
      const base = (0.20 + v * 0.62) * st.weight;
      for (const vx of st.vertexIds) {
        link[vx.index] = base * vx.taper;
        const c = vx.index * 3;
        lcol[c] = this._col.r; lcol[c + 1] = this._col.g; lcol[c + 2] = this._col.b;
      }
    }

    // -- beads --------------------------------------------------------------
    const bpos = this.beads.geometry.attributes.position.array;
    const bcol = this.beads.geometry.attributes.color.array;
    const bink = this.beads.geometry.attributes.aInk.array;
    for (let k = 0; k < this.nBead; k++) {
      const b = this.bead[k];
      b.t += dt * b.speed;
      if (b.t > 1) { b.t = 0; b.path = Math.floor(hash(t * 13 + k) * this.paths.length); }
      const [ia, ib] = this.paths[b.path];
      const a = this.electrodes[ia];
      const c = this.electrodes[ib];
      const u = b.t;
      const j = k * 3;
      bpos[j] = a.x + (c.x - a.x) * u;
      bpos[j + 1] = a.y + (c.y - a.y) * u;
      bpos[j + 2] = a.z + (c.z - a.z) * u;
      const energy = (this.vis[ia] + this.vis[ib]) * 0.5;
      this._col.copy(INK).lerp(OXBLOOD, 0.35 + energy * 0.5);
      bcol[j] = this._col.r; bcol[j + 1] = this._col.g; bcol[j + 2] = this._col.b;
      bink[k] = Math.sin(u * Math.PI) * (0.35 + energy * 0.65);
    }

    for (const [obj, attrs] of [
      [this.somata, ['color', 'aSize', 'aInk']],
      [this.stipple, ['color', 'aInk']],
      [this.processes, ['color', 'aInk']],
      [this.beads, ['position', 'color', 'aInk']],
    ]) {
      for (const a of attrs) obj.geometry.attributes[a].needsUpdate = true;
    }

    this.r.render(this.scene, this.cam);
  }
}

export { NeuralEnvironment as BrainScene };
