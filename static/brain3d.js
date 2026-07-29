/**
 * The page as an anatomical plate.
 *
 * Drawn the way Santiago Ramón y Cajal drew cortex in the 1890s: iron-gall ink
 * on laid paper, somata as filled discs, dendrites as tapered strokes, shading
 * by stipple. Nothing glows and nothing is additively blended — on paper,
 * "brighter" means *more ink*, so activity darkens and thickens rather than
 * lighting up.
 *
 * The form is an actual brain, not a sphere: an ellipsoid worked into shape by
 * `brainSurface` (narrowed poles, flattened skull base, longitudinal fissure,
 * Sylvian groove) and then folded into gyri, with a separate cerebellum and
 * brainstem. The cortex is rendered as engraver's hatching that follows the
 * folds — dense in the sulci, open on the gyral crowns — because that is how
 * an anatomical plate renders a curved surface, and because it gives every
 * region a place to hold ink.
 *
 * Three things fall out of drawing on paper rather than on black:
 *
 *   - Depth is aerial perspective, not occlusion. Every shader takes a fog
 *     term that fades distant strokes toward the paper, and nothing writes to
 *     the depth buffer.
 *   - Strokes bow and taper. A straight segment reads as a computer plot; a
 *     stroke that bows and thins at both ends reads as a pen. All wobble is
 *     seeded deterministically, so it is the same drawing every load.
 *   - Ink weight is the only channel for magnitude. Hatching, dendrites,
 *     stipple and somata all take theirs from the measured mu/beta power at
 *     the nearest contact, so an active region visibly darkens.
 *
 * The data underneath is real: 64 contacts at their true montage coordinates,
 * projected onto the cortical surface, and a committed decision washing the
 * responding hemisphere in oxblood.
 */

import * as THREE from './three.module.min.js';
import { strokes } from './inkfont.js';

const MAX_CHARS = 22;   // margin line length before the oldest letter is dropped
const MAX_PINS = 9;     // annotations kept on the specimen at once
const SEG_PER_GLYPH = 18;
// Sized so a full margin line spans well inside the frustum at its held depth.
const GLYPH_SIZE = 0.085;

const INK = new THREE.Color('#2a2018');      // iron gall, warm near-black
const SEPIA = new THREE.Color('#6d5333');    // faded ink, secondary structure
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

/** Deterministic hash so the plate redraws identically on every load. */
function hash(n) {
  const x = Math.sin(n * 127.1) * 43758.5453;
  return x - Math.floor(x);
}

function smoothstep(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// ── anatomy ───────────────────────────────────────────────────────────────
// Frame matches /api/v1/geometry: +x right, +y up, +z posterior.

/**
 * Gyral folding. Sines driven by other sines meander instead of forming a
 * grid, which is what makes the pattern read as convolutions rather than
 * corduroy. Negative is a sulcus, positive a gyral crown.
 */
function gyri(x, y, z) {
  const a = Math.sin(6.1 * x + 1.9 * Math.sin(3.1 * z + 0.7));
  const b = Math.sin(5.3 * z + 2.2 * Math.sin(3.7 * y + 1.3));
  const c = Math.sin(6.9 * y + 1.6 * Math.sin(4.3 * x + 2.1));
  return (a + b + c) / 3;
}

/**
 * Maps a unit direction to a point on the cerebral surface. `fold` scales the
 * gyral displacement — the hatching wants the folded surface, the landmark
 * strokes want the smooth one underneath.
 */
function brainSurface(dx, dy, dz, out, fold = 1) {
  // Roughly 165 mm long, 140 wide, 115 tall.
  let px = dx * 1.00;
  let py = dy * 0.84;
  let pz = dz * 1.22;

  // Frontal pole narrows and drops; occipital tapers less.
  const front = Math.max(0, -dz);
  const back = Math.max(0, dz);
  px *= 1 - 0.22 * front * front - 0.12 * back * back;
  py *= 1 - 0.12 * front * front;

  // Temporal lobe: a lateral bulge, low and forward of centre.
  const temp = Math.exp(-((dy + 0.42) ** 2) / 0.10) * Math.exp(-((dz + 0.05) ** 2) / 0.55);
  px *= 1 + 0.20 * temp;

  // The brain sits on the skull base, so the inferior surface is flat.
  if (py < -0.30) py = -0.30 + (py + 0.30) * 0.40;

  // Longitudinal fissure: the deep sagittal groove between the hemispheres.
  const fissure = smoothstep(0.02, 0.45, dy) * Math.exp(-(dx * dx) / 0.012);
  py -= fissure * 0.20;

  // Sylvian fissure: an oblique groove above the temporal lobe.
  const syl = Math.exp(-((dy + 0.16 + 0.22 * dz) ** 2) / 0.014)
    * smoothstep(0.15, 0.5, Math.abs(dx));
  const shrink = 1 - syl * 0.09;
  px *= shrink; py *= shrink; pz *= shrink;

  // Convolutions, displaced along the outward normal.
  if (fold !== 0) {
    const g = gyri(px, py, pz);
    const len = Math.hypot(px, py, pz) || 1;
    const k = 1 + fold * 0.036 * g;
    px *= k; py *= k; pz *= k;
    void len;
  }
  return out.set(px, py, pz);
}

// ── shaders ───────────────────────────────────────────────────────────────

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

export class NeuralEnvironment {
  constructor(canvas, geometry) {
    this.canvas = canvas;
    this.electrodes = geometry.electrodes;
    this.n = this.electrodes.length;
    this.power = new Float32Array(this.n).fill(0.42);
    this.target = new Float32Array(this.n).fill(0.42);
    this.vis = new Float32Array(this.n).fill(0.42);
    this.flashAmount = 0;
    this.flashWas = 0;
    this.flashSide = null;

    // Posterior wash, per hemisphere region, and the lettering the plate is
    // currently carrying.
    this.wash = { left_motor: 0, right_motor: 0, midline_motor: 0 };
    this.washEased = { left_motor: 0, right_motor: 0, midline_motor: 0 };
    this.text = [];
    this.glyphs = [];
    this.pins = [];

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
    this._project();
    this._cortex();
    this._landmarks();
    this._cerebellum();
    this._stem();
    this._arbors();
    this._processes();
    this._somata();
    this._stipple();
    this._beads();
    this._margin();
    this._pinwork();
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
    // The margin is parented to the camera, so it must be in the graph.
    this.scene.add(this.cam);
    this.rig = new THREE.Group();
    this.scene.add(this.rig);
    // Slight anatomical tilt, as a specimen is mounted.
    this.rig.rotation.z = 0.04;
    this.somaTex = inkDot(0.74);
    this.speckTex = inkDot(0.52);
    this._v = new THREE.Vector3();
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
      uniforms: {
        uNear: this.uNear, uFar: this.uFar,
        uMap: { value: map }, uScale: { value: 90 },
      },
      vertexShader: DOT_VERT, fragmentShader: DOT_FRAG,
      transparent: true, depthWrite: false, vertexColors: true,
    });
  }

  _lines(pos, col, ink) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.Float32BufferAttribute(ink, 1));
    return new THREE.LineSegments(g, this._strokeMaterial());
  }

  _points(pos, col, size, ink, map) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));
    return new THREE.Points(g, this._dotMaterial(map));
  }

  /** Contacts are scalp positions; drop each onto the cortex beneath it. */
  _project() {
    this.site = [];
    this.normal = [];
    const p = new THREE.Vector3();
    for (const e of this.electrodes) {
      const len = Math.hypot(e.x, e.y, e.z) || 1;
      brainSurface(e.x / len, e.y / len, e.z / len, p);
      this.site.push(p.clone().multiplyScalar(1.012));
      this.normal.push(p.clone().normalize());
    }
  }

  /** Nearest contact to an arbitrary surface point, for ink attribution. */
  _owner(x, y, z) {
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < this.n; i++) {
      const s = this.site[i];
      const d = (s.x - x) ** 2 + (s.y - y) ** 2 + (s.z - z) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  /**
   * Engraver's hatching over the cortex. Each stroke runs *along* a fold
   * (perpendicular to the gradient of the gyral field), so the marks describe
   * the surface instead of sitting on it. Ink is heaviest in the sulci and is
   * further weighted, per frame, by the power at the nearest contact.
   */
  _cortex() {
    const N = this.soft ? 1800 : 5400;
    const pos = [];
    const col = [];
    const ink = [];
    this.hatchOwner = new Int32Array(N);
    this.hatchBase = new Float32Array(N);

    const p = new THREE.Vector3();
    const q = new THREE.Vector3();
    const nrm = new THREE.Vector3();
    const grad = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const EPS = 0.02;
    let k = 0;

    for (let i = 0; i < N && k < N; i++) {
      // Uniform on the sphere, then pushed to the cortical surface.
      const u = 2 * hash(i * 1.37) - 1;
      const th = hash(i * 2.53 + 4) * Math.PI * 2;
      const s = Math.sqrt(Math.max(0, 1 - u * u));
      const dx = s * Math.cos(th);
      const dy = u;
      const dz = s * Math.sin(th);

      brainSurface(dx, dy, dz, p);
      // Leave the cerebellum and stem their own territory.
      if (dz > 0.42 && dy < -0.18) continue;

      nrm.copy(p).normalize();
      const g0 = gyri(p.x, p.y, p.z);
      grad.set(
        gyri(p.x + EPS, p.y, p.z) - g0,
        gyri(p.x, p.y + EPS, p.z) - g0,
        gyri(p.x, p.y, p.z + EPS) - g0,
      );
      grad.addScaledVector(nrm, -grad.dot(nrm));      // project to tangent
      if (grad.lengthSq() < 1e-9) grad.set(nrm.z, 0, -nrm.x);
      dir.crossVectors(nrm, grad).normalize();        // run along the fold

      const len = 0.030 + hash(i * 3.91) * 0.032;
      q.copy(p).addScaledVector(dir, len);
      // Stay on the same radial shell. Re-projecting through brainSurface
      // instead would send strokes flying: near the fissure its gradient is
      // steep enough that two adjacent directions land far apart, which drew
      // long scratches across the vertex.
      q.multiplyScalar((p.length() || 1) / (q.length() || 1));

      pos.push(p.x, p.y, p.z, q.x, q.y, q.z);
      for (let v = 0; v < 2; v++) col.push(INK.r, INK.g, INK.b);
      ink.push(0, 0);

      this.hatchOwner[k] = this._owner(p.x, p.y, p.z);
      // Sulci hold ink; crowns stay open. Plus a little stroke-to-stroke
      // variation so the hatching is not mechanically even.
      this.hatchBase[k] = (0.22 + 0.62 * Math.max(0, -g0))
        * (0.72 + 0.28 * hash(i * 7.77));
      k++;
    }
    this.nHatch = k;
    this.hatch = this._lines(pos, col, ink);
    this.rig.add(this.hatch);
  }

  /** A stroke traced across the smooth surface, used for the fissures. */
  _trace(fn, steps, wobble, seed) {
    const pts = [];
    const p = new THREE.Vector3();
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const [dx, dy, dz] = fn(t);
      const len = Math.hypot(dx, dy, dz) || 1;
      brainSurface(dx / len, dy / len, dz / len, p, 0);
      const w = 1 + wobble * Math.sin(t * 21 + seed) + wobble * 0.6 * Math.sin(t * 37 + seed * 2);
      pts.push([p.x * w, p.y * w, p.z * w]);
    }
    return pts;
  }

  /**
   * The landmarks that make a brain legible at a glance: the longitudinal
   * fissure, both Sylvian fissures, and the central sulcus on each side.
   */
  _landmarks() {
    const pos = [];
    const col = [];
    const ink = [];

    const add = (pts, weight) => {
      for (let s = 0; s < pts.length - 1; s++) {
        const t = s / (pts.length - 1);
        pos.push(...pts[s], ...pts[s + 1]);
        for (let v = 0; v < 2; v++) {
          col.push(INK.r, INK.g, INK.b);
          ink.push(weight * (0.45 + 0.55 * Math.sin(t * Math.PI)));
        }
      }
    };

    // Longitudinal fissure, front to back over the vertex.
    add(this._trace((t) => {
      const a = (t - 0.5) * Math.PI * 1.06;
      return [0, Math.cos(a), Math.sin(a)];
    }, 90, 0.006, 1.1), 0.7);

    for (const side of [-1, 1]) {
      // Sylvian fissure: oblique, rising toward the back.
      add(this._trace((t) => {
        const z = -0.72 + t * 1.5;
        return [side * (0.85 - 0.2 * t), -0.30 + 0.30 * t, z];
      }, 60, 0.007, side * 2.3), 0.62);

      // Central sulcus: down and forward from the vertex.
      add(this._trace((t) => {
        const a = t * 1.15;
        return [side * Math.sin(a) * 0.95, Math.cos(a), 0.12 - 0.42 * t];
      }, 50, 0.006, side * 3.7), 0.5);
    }

    this.landmarks = this._lines(pos, col, ink);
    this.rig.add(this.landmarks);
  }

  /**
   * Cerebellum. Its folia are far finer and more regular than cortical gyri,
   * so it is drawn as stacked parallel rings — which is exactly what makes it
   * recognisable next to the cerebrum.
   */
  _cerebellum() {
    const C = new THREE.Vector3(0, -0.54, 0.88);
    const R = { x: 0.46, y: 0.25, z: 0.30 };
    const rows = this.soft ? 15 : 26;
    const pos = [];
    const col = [];
    const ink = [];

    for (let r = 0; r < rows; r++) {
      const v = (r / (rows - 1)) * 2 - 1;                 // -1 .. 1 vertically
      const ring = Math.sqrt(Math.max(0, 1 - v * v));
      const y = C.y + v * R.y;
      let prev = null;
      const steps = 46;
      for (let s = 0; s <= steps; s++) {
        // Posterior arc only: the front of it is hidden by the cerebrum.
        const a = -Math.PI * 0.62 + (s / steps) * Math.PI * 1.24;
        const w = 1 + 0.02 * Math.sin(s * 0.9 + r);
        const cur = [
          C.x + Math.sin(a) * R.x * ring * w,
          y,
          C.z + Math.cos(a) * R.z * ring * w,
        ];
        if (prev) {
          pos.push(...prev, ...cur);
          for (let k = 0; k < 2; k++) {
            col.push(SEPIA.r, SEPIA.g, SEPIA.b);
            ink.push(0.22 * (0.5 + 0.5 * ring));
          }
        }
        prev = cur;
      }
    }
    this.cerebellum = this._lines(pos, col, ink);
    this.rig.add(this.cerebellum);
  }

  /**
   * Brainstem, plus the cranial nerve filaments leaving it. These carry no
   * data — they are here because a brain without them reads as a diagram.
   */
  _stem() {
    const pos = [];
    const col = [];
    const ink = [];
    const top = new THREE.Vector3(0, -0.34, 0.30);
    const bot = new THREE.Vector3(0, -0.92, 0.50);

    // Longitudinal strokes down the stem, tapering as it descends.
    const fibres = 9;
    for (let f = 0; f < fibres; f++) {
      const a = (f / fibres) * Math.PI * 2;
      let prev = null;
      const steps = 22;
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        const rad = 0.20 * (1 - 0.45 * t) * (1 + 0.06 * Math.sin(t * 9 + f));
        const cur = [
          top.x + (bot.x - top.x) * t + Math.cos(a) * rad,
          top.y + (bot.y - top.y) * t,
          top.z + (bot.z - top.z) * t + Math.sin(a) * rad * 0.8,
        ];
        if (prev) {
          pos.push(...prev, ...cur);
          for (let k = 0; k < 2; k++) {
            col.push(INK.r, INK.g, INK.b);
            ink.push(0.34 * (1 - 0.5 * t));
          }
        }
        prev = cur;
      }
    }

    // Cranial nerves: fine filaments peeling away from the stem.
    for (let nvi = 0; nvi < 14; nvi++) {
      const t0 = 0.12 + hash(nvi * 4.7) * 0.7;
      const side = nvi % 2 ? 1 : -1;
      const ox = top.x + (bot.x - top.x) * t0;
      const oy = top.y + (bot.y - top.y) * t0;
      const oz = top.z + (bot.z - top.z) * t0;
      const spread = 0.4 + hash(nvi * 6.1) * 0.6;
      let prev = [ox + side * 0.16, oy, oz];
      const steps = 14;
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        const cur = [
          prev[0] + side * 0.035 * spread,
          oy - 0.055 * t * spread + 0.03 * Math.sin(t * 6 + nvi),
          oz - 0.04 * t * spread,
        ];
        pos.push(...prev, ...cur);
        for (let k = 0; k < 2; k++) {
          col.push(SEPIA.r, SEPIA.g, SEPIA.b);
          ink.push(0.30 * (1 - t) + 0.04);
        }
        prev = cur;
      }
    }

    this.stem = this._lines(pos, col, ink);
    this.rig.add(this.stem);
  }

  /**
   * A dendritic arbor at every contact: three primary trunks leaving the soma
   * tangentially, each branching twice, tapering throughout. This is the
   * "nerve" of the drawing — the whole tree takes its ink from that contact's
   * power, so activity spreads through a structure rather than a dot.
   */
  _arbors() {
    const pos = [];
    const col = [];
    const ink = [];
    this.arbor = [];

    const nrm = new THREE.Vector3();
    const t1 = new THREE.Vector3();
    const t2 = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const from = new THREE.Vector3();
    const to = new THREE.Vector3();
    const axis = new THREE.Vector3();
    let seed = 0;

    const segment = (a, b, depth, ids) => {
      const SUB = 3;
      const bow = (hash(seed++) - 0.5) * 0.05;
      axis.subVectors(b, a);
      const perp = new THREE.Vector3(axis.y, -axis.x, axis.z).normalize();
      let prev = null;
      for (let s = 0; s <= SUB; s++) {
        const t = s / SUB;
        const swell = Math.sin(t * Math.PI);
        const cur = new THREE.Vector3()
          .copy(a).addScaledVector(axis, t).addScaledVector(perp, bow * swell);
        if (prev) {
          for (const v of [prev, cur]) {
            ids.push({ index: pos.length / 3, taper: (0.30 + 0.70 * swell) * (1 - depth * 0.26) });
            pos.push(v.x, v.y, v.z);
            col.push(INK.r, INK.g, INK.b);
            ink.push(0);
          }
        }
        prev = cur;
      }
    };

    const grow = (base, heading, len, depth, ids) => {
      to.copy(base).addScaledVector(heading, len);
      segment(base.clone(), to.clone(), depth, ids);
      if (depth >= 2) return;
      const next = to.clone();
      for (const turn of [-1, 1]) {
        const h = heading.clone();
        // Splay in the tangent plane and lift slightly off the surface.
        const side = new THREE.Vector3().crossVectors(h, nrm).normalize();
        h.addScaledVector(side, turn * (0.55 + hash(seed++) * 0.4))
          .addScaledVector(nrm, 0.10 + hash(seed++) * 0.14)
          .normalize();
        grow(next, h, len * 0.62, depth + 1, ids);
      }
    };

    for (let i = 0; i < this.n; i++) {
      const site = this.site[i];
      nrm.copy(this.normal[i]);
      // Any two vectors spanning the tangent plane at this contact.
      t1.set(-nrm.z, 0, nrm.x);
      if (t1.lengthSq() < 1e-6) t1.set(1, 0, 0);
      t1.normalize();
      t2.crossVectors(nrm, t1).normalize();

      const ids = [];
      const trunks = 3;
      const phase = hash(i * 9.13) * Math.PI * 2;
      for (let b = 0; b < trunks; b++) {
        const a = phase + (b / trunks) * Math.PI * 2;
        dir.copy(t1).multiplyScalar(Math.cos(a))
          .addScaledVector(t2, Math.sin(a))
          .addScaledVector(nrm, 0.30)
          .normalize();
        from.copy(site);
        grow(from, dir, 0.078 + hash(i * 3.3 + b) * 0.038, 0, ids);
      }
      this.arbor.push(ids);
    }

    this.arbors = this._lines(pos, col, ink);
    this.rig.add(this.arbors);
  }

  /**
   * Long processes between neighbouring contacts. Kept faint — the arbors are
   * the anatomy; these carry the travelling beads and the sense of a connected
   * sheet.
   */
  _processes() {
    const SEG = 7;
    const pairs = [];
    for (let i = 0; i < this.n; i++) {
      for (let j = i + 1; j < this.n; j++) {
        const a = this.site[i];
        const b = this.site[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (d < 0.45) pairs.push([i, j]);
      }
    }
    this.paths = pairs;
    this.strokes = [];

    const pos = [];
    const col = [];
    const ink = [];
    const dir = new THREE.Vector3();
    const perp = new THREE.Vector3();
    const cur = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);

    pairs.forEach(([i, j], k) => {
      const A = this.site[i];
      const B = this.site[j];
      dir.subVectors(B, A);
      perp.crossVectors(dir, up).normalize();
      if (!isFinite(perp.x)) perp.set(1, 0, 0);
      const bow = (hash(k * 1.7) - 0.5) * 0.07;

      const verts = [];
      for (let s = 0; s <= SEG; s++) {
        const t = s / SEG;
        const swell = Math.sin(t * Math.PI);
        cur.copy(A).addScaledVector(dir, t).addScaledVector(perp, bow * swell)
          .multiplyScalar(1 + 0.02 * swell);
        verts.push([cur.x, cur.y, cur.z, swell]);
      }
      const ids = [];
      for (let s = 0; s < SEG; s++) {
        for (const v of [verts[s], verts[s + 1]]) {
          ids.push({ index: pos.length / 3, taper: 0.3 + 0.7 * v[3] });
          pos.push(v[0], v[1], v[2]);
          col.push(INK.r, INK.g, INK.b);
          ink.push(0);
        }
      }
      this.strokes.push({ i, j, vertexIds: ids });
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
      const s = this.site[i];
      pos[i * 3] = s.x; pos[i * 3 + 1] = s.y; pos[i * 3 + 2] = s.z;
      col[i * 3] = INK.r; col[i * 3 + 1] = INK.g; col[i * 3 + 2] = INK.b;
      size[i] = 0.4;
    }
    this.somata = this._points(pos, col, size, ink, this.somaTex);
    this.rig.add(this.somata);
  }

  /** Stipple shading, hugging each soma and thinning with distance. */
  _stipple() {
    const per = this.soft ? 12 : 26;
    const n = this.n * per;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const ink = new Float32Array(n);
    this.stippleOwner = new Int32Array(n);
    this.stippleBias = new Float32Array(n);

    for (let i = 0; i < this.n; i++) {
      const e = this.site[i];
      for (let s = 0; s < per; s++) {
        const k = i * per + s;
        const h1 = hash(k * 1.7);
        const h2 = hash(k * 3.1 + 11);
        const h3 = hash(k * 5.9 + 23);
        const rad = 0.05 + Math.pow(h1, 1.7) * 0.17;
        const th = h2 * Math.PI * 2;
        const ph = Math.acos(2 * h3 - 1);
        pos[k * 3] = e.x + rad * Math.sin(ph) * Math.cos(th);
        pos[k * 3 + 1] = e.y + rad * Math.cos(ph);
        pos[k * 3 + 2] = e.z + rad * Math.sin(ph) * Math.sin(th);
        col[k * 3] = WASH.r; col[k * 3 + 1] = WASH.g; col[k * 3 + 2] = WASH.b;
        size[k] = 0.05 + h2 * 0.045;
        this.stippleOwner[k] = i;
        this.stippleBias[k] = 1 - Math.pow(h1, 1.7);
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
      size[k] = 0.09 + hash(k * 6.7) * 0.055;
      ink[k] = 0.6;
    }
    this.beads = this._points(pos, col, size, ink, this.somaTex);
    this.rig.add(this.beads);
  }

  /**
   * The margin line the decoded text is written into.
   *
   * Buffers are preallocated to the worst case and the draw count is moved
   * instead of reallocating, because a letter arrives roughly every second and
   * rebuilding geometry that often would churn. It sits in world space below
   * the specimen, in the plane the camera faces, so the lettering reads flat
   * on the paper while the brain turns behind it.
   */
  _margin() {
    const max = MAX_CHARS * SEG_PER_GLYPH;
    const pos = new Float32Array(max * 6);
    const col = new Float32Array(max * 6);
    const ink = new Float32Array(max * 2);
    this.marginAt = new Float32Array(max * 2);   // pen travel, 0..1 per glyph
    for (let i = 0; i < max * 2; i++) {
      col[i * 3] = INK.r; col[i * 3 + 1] = INK.g; col[i * 3 + 2] = INK.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));
    g.setDrawRange(0, 0);
    this.marginCount = 0;
    this.margin = new THREE.LineSegments(g, this._strokeMaterial());
    // Held in camera space: the specimen turns and the page scrolls, but the
    // written line stays on the paper in front of the reader.
    this.margin.position.set(0, -0.56, -2.0);
    this.cam.add(this.margin);

    // A ruled line to write on, as a plate has.
    const rp = [];
    const rc = [];
    const ri = [];
    const half = (MAX_CHARS * 0.86 * GLYPH_SIZE) / 2;
    for (let s = 0; s < 60; s++) {
      const x0 = -half + (s / 60) * half * 2;
      const x1 = -half + ((s + 1) / 60) * half * 2;
      rp.push(x0, -0.055, 0, x1, -0.055, 0);
      for (let v = 0; v < 2; v++) {
        rc.push(SEPIA.r, SEPIA.g, SEPIA.b);
        ri.push(0.16 + 0.1 * Math.sin(s * 0.7));
      }
    }
    this.rule = this._lines(rp, rc, ri);
    this.rule.position.copy(this.margin.position);
    this.cam.add(this.rule);
  }

  /**
   * Rebuild the margin buffer from `this.text`. Called only when a letter is
   * added or the oldest is dropped, so per-glyph draw progress is preserved
   * across the rebuild rather than restarting every animation.
   */
  _layoutMargin() {
    const g = this.margin.geometry;
    const pos = g.attributes.position.array;
    const size = GLYPH_SIZE;
    let n = 0;

    // Centre the line on the advance width actually used.
    let width = 0;
    for (const ch of this.text) width += strokes(ch).advance * size;
    let x = -width / 2;

    for (let gi = 0; gi < this.text.length; gi++) {
      const { segments, advance } = strokes(this.text[gi]);
      const glyph = this.glyphs[gi];
      glyph.start = n;
      for (const [x1, y1, x2, y2, at] of segments) {
        if (n >= MAX_CHARS * SEG_PER_GLYPH) break;
        pos[n * 6] = x + x1 * size;
        pos[n * 6 + 1] = y1 * size;
        pos[n * 6 + 2] = 0;
        pos[n * 6 + 3] = x + x2 * size;
        pos[n * 6 + 4] = y2 * size;
        pos[n * 6 + 5] = 0;
        this.marginAt[n * 2] = at;
        this.marginAt[n * 2 + 1] = at;
        n++;
      }
      glyph.count = n - glyph.start;
      x += advance * size;
    }
    g.setDrawRange(0, n * 2);
    g.attributes.position.needsUpdate = true;
    this.marginCount = n;
  }

  /** Specimen pins: a leader off the cortex, a tick, and a lettered label. */
  _pinwork() {
    const max = MAX_PINS * (SEG_PER_GLYPH + 8);
    const pos = new Float32Array(max * 6);
    const col = new Float32Array(max * 6);
    const ink = new Float32Array(max * 2);
    this.pinOf = new Int32Array(max * 2);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));
    g.setDrawRange(0, 0);
    this.pinCount = 0;
    this.pinwork = new THREE.LineSegments(g, this._strokeMaterial());
    this.rig.add(this.pinwork);
  }

  /** Rebuild pin geometry. Runs at most once per committed decision. */
  _layoutPins() {
    const g = this.pinwork.geometry;
    const pos = g.attributes.position.array;
    const col = g.attributes.color.array;
    // Kept short: a lateral contact's normal points straight out of frame, so
    // a long leader takes the label off the plate entirely.
    const size = 0.070;
    const LEADER = 0.15;
    const TICK = 0.05;
    let n = 0;

    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    const worldUp = new THREE.Vector3(0, 1, 0);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();

    const push = (p1, p2, pinIndex, tint) => {
      if (n >= this.pinOf.length / 2) return;
      pos[n * 6] = p1.x; pos[n * 6 + 1] = p1.y; pos[n * 6 + 2] = p1.z;
      pos[n * 6 + 3] = p2.x; pos[n * 6 + 4] = p2.y; pos[n * 6 + 5] = p2.z;
      for (let v = 0; v < 2; v++) {
        const k = (n * 2 + v) * 3;
        col[k] = tint.r; col[k + 1] = tint.g; col[k + 2] = tint.b;
        this.pinOf[n * 2 + v] = pinIndex;
      }
      n++;
    };

    this.pins.forEach((pin, pi) => {
      const site = this.site[pin.site];
      const nrm = this.normal[pin.site];
      // Leader: straight out along the surface normal, then a short tick.
      a.copy(site);
      b.copy(site).addScaledVector(nrm, LEADER);
      c.copy(b).addScaledVector(nrm, TICK);
      push(a, b, pi, OXBLOOD);
      push(b, c, pi, OXBLOOD);

      right.crossVectors(worldUp, nrm);
      if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
      right.normalize();
      up.crossVectors(nrm, right).normalize();

      const { segments } = strokes(pin.label);
      const originX = 0.028;
      const originY = -0.03;
      const p1 = new THREE.Vector3();
      const p2 = new THREE.Vector3();
      for (const [x1, y1, x2, y2] of segments) {
        p1.copy(c).addScaledVector(right, (originX + x1 * size))
          .addScaledVector(up, (originY + y1 * size));
        p2.copy(c).addScaledVector(right, (originX + x2 * size))
          .addScaledVector(up, (originY + y2 * size));
        push(p1, p2, pi, INK);
      }
    });

    g.setDrawRange(0, n * 2);
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    this.pinCount = n;
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
    // Sets the nib: a soma lands around 6–16 px at reading distance, stipple
    // at 2–3 px so specks read as dots rather than merging into a wash.
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

  /** Motor imagery is contralateral: a right-hand class answers on the left. */
  static region(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('right')) return 'left_motor';
    if (l.includes('left')) return 'right_motor';
    return 'midline_motor';
  }

  flash(label) {
    this.flashSide = NeuralEnvironment.region(label);
    this.flashAmount = 1;
  }

  /**
   * Live posterior over classes, as a wash. Each class inks the hemisphere
   * that would produce it, so the two hypotheses visibly compete across the
   * plate instead of racing along a bar.
   */
  posterior(map) {
    if (!map) return;
    for (const k of Object.keys(this.wash)) this.wash[k] = 0;
    for (const [label, p] of Object.entries(map)) {
      const r = NeuralEnvironment.region(label);
      this.wash[r] = Math.max(this.wash[r], p);
    }
  }

  /**
   * A committed decision: flare the responding hemisphere and pin the site
   * that carried it, labelled with whatever the speller emitted.
   */
  commit(decision) {
    if (!decision) return;
    this.flash(decision.label);
    const region = this.flashSide;

    // Pin the most active contact in the responding region — the one the
    // decision actually rested on.
    let site = -1;
    let best = -Infinity;
    for (let i = 0; i < this.n; i++) {
      if (this.electrodes[i].region !== region) continue;
      if (this.vis[i] > best) { best = this.vis[i]; site = i; }
    }
    if (site < 0) return;

    const label = (decision.emitted && decision.emitted !== '—')
      ? decision.emitted
      : (decision.label || '?').charAt(0);
    this.pins.push({ site, label, age: 0 });
    if (this.pins.length > MAX_PINS) this.pins.shift();
    this._layoutPins();
  }

  /**
   * Sync the margin lettering to the spelled text. New characters are inked
   * stroke by stroke; the rest are left alone. Driven from the server's text
   * rather than from emissions, so the plate can never drift out of step with
   * what was actually decoded.
   */
  setText(str) {
    const next = Array.from(str || '');
    const same = next.length >= this.text.length
      && this.text.every((c, i) => c === next[i]);

    if (!same) {
      this.text = next.slice(-MAX_CHARS);
      this.glyphs = this.text.map(() => ({ progress: 1, start: 0, count: 0 }));
      this._layoutMargin();
      return;
    }
    for (let i = this.text.length; i < next.length; i++) {
      this.text.push(next[i]);
      this.glyphs.push({ progress: 0, start: 0, count: 0 });
      while (this.text.length > MAX_CHARS) { this.text.shift(); this.glyphs.shift(); }
    }
    this._layoutMargin();
  }

  reset() {
    this.power.fill(0.42);
    this.target.fill(0.42);
    this.flashAmount = 0;
    this.text = [];
    this.glyphs = [];
    this.pins = [];
    for (const k of Object.keys(this.wash)) this.wash[k] = 0;
    this._layoutMargin();
    this._layoutPins();
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
    this.rig.rotation.x = this.pitch + s * 0.35;
    this.rig.rotation.y = this.yaw;
    this.spatter.rotation.y = -this.yaw * 0.2;

    this.cam.position.set(
      this.pointer.x * 0.20,
      0.25 + s * 0.55 - this.pointer.y * 0.14,
      3.15 - s * 1.55,
    );
    this.cam.lookAt(0, s * 0.12, 0);

    const ease = 1 - Math.exp(-dt * 7);
    this.flashAmount = Math.max(0, this.flashAmount - dt * 1.1);
    // Colour only moves during a flash, so skip rewriting it the rest of the
    // time — it is by far the largest attribute on the plate.
    const recolour = this.flashAmount > 0 || this.flashWas > 0;
    this.flashWas = this.flashAmount;

    // -- somata -------------------------------------------------------------
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
      ssize[i] = 0.18 + v * 0.30 + lit * 0.28;
      sink[i] = 0.42 + v * 0.58;
    }

    // -- posterior wash -----------------------------------------------------
    for (const k of Object.keys(this.wash)) {
      this.washEased[k] += (this.wash[k] - this.washEased[k]) * (1 - Math.exp(-dt * 3.5));
    }

    // -- cortical hatching --------------------------------------------------
    const hink = this.hatch.geometry.attributes.aInk.array;
    const hcol = this.hatch.geometry.attributes.color.array;
    for (let k = 0; k < this.nHatch; k++) {
      const owner = this.hatchOwner[k];
      const region = this.electrodes[owner].region;
      const lit = this.flashAmount > 0 && region === this.flashSide
        ? this.flashAmount : 0;
      // The competing hypotheses saturate their own hemisphere. Subtracting a
      // half keeps an even 50/50 split neutral, so the wash reads as a lead
      // rather than as overall brightness.
      const bias = Math.max(0, (this.washEased[region] || 0) - 0.5) * 0.62;
      const w = this.hatchBase[k] * (0.55 + this.vis[owner] * 0.95 + bias) + lit * 0.30;
      hink[k * 2] = w;
      hink[k * 2 + 1] = w;
      if (recolour) {
        this._col.copy(INK).lerp(OXBLOOD, lit * 0.8);
        for (let v = 0; v < 2; v++) {
          const c = (k * 2 + v) * 3;
          hcol[c] = this._col.r; hcol[c + 1] = this._col.g; hcol[c + 2] = this._col.b;
        }
      }
    }

    // -- dendritic arbors ---------------------------------------------------
    const aink = this.arbors.geometry.attributes.aInk.array;
    const acol = this.arbors.geometry.attributes.color.array;
    for (let i = 0; i < this.n; i++) {
      const v = this.vis[i];
      const lit = this.flashAmount > 0
        && this.electrodes[i].region === this.flashSide ? this.flashAmount : 0;
      const base = 0.15 + v * 0.48;
      if (recolour) this._col.copy(INK).lerp(OXBLOOD, lit * 0.85);
      for (const vx of this.arbor[i]) {
        aink[vx.index] = base * vx.taper;
        if (recolour) {
          const c = vx.index * 3;
          acol[c] = this._col.r; acol[c + 1] = this._col.g; acol[c + 2] = this._col.b;
        }
      }
    }

    // -- stipple ------------------------------------------------------------
    const pink = this.stipple.geometry.attributes.aInk.array;
    const pcol = this.stipple.geometry.attributes.color.array;
    for (let k = 0; k < this.stippleOwner.length; k++) {
      const owner = this.stippleOwner[k];
      const lit = this.flashAmount > 0
        && this.electrodes[owner].region === this.flashSide ? this.flashAmount : 0;
      pink[k] = Math.max(0, (this.vis[owner] * 1.4 - 0.34) * this.stippleBias[k] + lit * 0.5);
      if (recolour) {
        this._col.copy(WASH).lerp(OXBLOOD, lit * 0.8);
        pcol[k * 3] = this._col.r;
        pcol[k * 3 + 1] = this._col.g;
        pcol[k * 3 + 2] = this._col.b;
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
      if (recolour) this._col.copy(INK).lerp(OXBLOOD, lit * 0.85);
      const base = 0.045 + v * 0.15;
      for (const vx of st.vertexIds) {
        link[vx.index] = base * vx.taper;
        if (recolour) {
          const c = vx.index * 3;
          lcol[c] = this._col.r; lcol[c + 1] = this._col.g; lcol[c + 2] = this._col.b;
        }
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
      const A = this.site[ia];
      const B = this.site[ib];
      const u = b.t;
      const j = k * 3;
      bpos[j] = A.x + (B.x - A.x) * u;
      bpos[j + 1] = A.y + (B.y - A.y) * u;
      bpos[j + 2] = A.z + (B.z - A.z) * u;
      const energy = (this.vis[ia] + this.vis[ib]) * 0.5;
      this._col.copy(INK).lerp(OXBLOOD, 0.35 + energy * 0.5);
      bcol[j] = this._col.r; bcol[j + 1] = this._col.g; bcol[j + 2] = this._col.b;
      bink[k] = Math.sin(u * Math.PI) * (0.35 + energy * 0.65);
    }

    // -- margin lettering ---------------------------------------------------
    // Roughly two characters a second of pen travel, so a letter is visibly
    // written rather than switched on.
    const mink = this.margin.geometry.attributes.aInk.array;
    for (let gi = 0; gi < this.glyphs.length; gi++) {
      const glyph = this.glyphs[gi];
      if (glyph.progress < 1) glyph.progress = Math.min(1, glyph.progress + dt * 2.2);
      for (let k = glyph.start; k < glyph.start + glyph.count; k++) {
        const at = this.marginAt[k * 2];
        const on = glyph.progress >= at ? 1
          : Math.max(0, 1 - (at - glyph.progress) * 14);
        mink[k * 2] = on;
        mink[k * 2 + 1] = on;
      }
    }
    this.margin.geometry.attributes.aInk.needsUpdate = true;

    // -- pins ---------------------------------------------------------------
    // Newest annotation is full strength; older ones fade back like earlier
    // marginalia, so the plate accumulates a history instead of a stack.
    if (this.pinCount) {
      const pkin = this.pinwork.geometry.attributes.aInk.array;
      for (const pin of this.pins) pin.age += dt;
      for (let k = 0; k < this.pinCount * 2; k++) {
        const pin = this.pins[this.pinOf[k]];
        const settle = pin ? Math.min(1, pin.age * 3.5) : 0;
        const fade = pin ? Math.max(0.34, 1 - pin.age * 0.045) : 0;
        pkin[k] = 0.8 * settle * fade;
      }
      this.pinwork.geometry.attributes.aInk.needsUpdate = true;
    }

    const dirty = [
      [this.somata, ['color', 'aSize', 'aInk']],
      [this.hatch, recolour ? ['color', 'aInk'] : ['aInk']],
      [this.arbors, recolour ? ['color', 'aInk'] : ['aInk']],
      [this.stipple, recolour ? ['color', 'aInk'] : ['aInk']],
      [this.processes, recolour ? ['color', 'aInk'] : ['aInk']],
      [this.beads, ['position', 'color', 'aInk']],
    ];
    for (const [obj, attrs] of dirty) {
      for (const a of attrs) obj.geometry.attributes[a].needsUpdate = true;
    }

    this.r.render(this.scene, this.cam);
  }
}

export { NeuralEnvironment as BrainScene };
