/**
 * A solid, lit, anatomically segmented brain you can turn and click.
 *
 * Modelled on the interaction of an anatomy atlas: real tissue shading, named
 * regions, and a region that lights when you select it — except here regions
 * also light on their own, driven by the decoder. Motor imagery is
 * contralateral, so a committed "right hand" flares the LEFT primary motor
 * cortex, and the live posterior warms whichever hemisphere is currently
 * winning.
 *
 * There is no brain mesh file to load. The surface is generated: a base
 * ellipsoid worked into shape by `brainSurface` (narrowed poles, temporal
 * bulge, flattened skull base, longitudinal fissure, Sylvian groove) and then
 * folded into gyri by two octaves of meandering sine. Cerebellum and brainstem
 * are separate meshes, and every vertex carries a region id assigned from its
 * position relative to the central and Sylvian sulci.
 *
 * Two things are worth knowing before changing any of this:
 *
 *   - Normals are analytic, not computed from the triangles. IcosahedronGeometry
 *     is non-indexed, so `computeVertexNormals` would give flat facets; instead
 *     each normal comes from the cross product of two finite differences of
 *     `brainSurface` itself, which is smooth regardless of tessellation.
 *   - Sulcal shading is an attribute, not a light. `aDepth` carries how deep in
 *     a fold each vertex sits and darkens it directly, which is what makes the
 *     convolutions read at a glance. No shadow pass would be affordable here.
 *
 * The palette stays with the page: warm tissue against cream paper, oxblood for
 * what the decoder is doing, verdigris for what the reader has selected.
 */

import * as THREE from './three.module.min.js';
import { strokes } from './inkfont.js';

const MAX_CHARS = 22;
const SEG_PER_GLYPH = 18;
const GLYPH_SIZE = 0.085;
const MAX_PINS = 9;
// The specimen spans about y -0.88 (brainstem) to +0.60 (vertex), so it is not
// centred on the origin and must be framed about its own middle.
const SPECIMEN_HALF = 0.80;
const SPECIMEN_CENTRE_Y = -0.13;

const INK = new THREE.Color('#241c14');
const SEPIA = new THREE.Color('#6d5333');
const OXBLOOD = new THREE.Color('#8f3320');
const VERDIGRIS = new THREE.Color('#3f6b57');

/** Region ids. Order is load-bearing: it indexes the shader uniform arrays. */
export const REGION = {
  FRONTAL: 0,
  MOTOR_LEFT: 1,
  MOTOR_RIGHT: 2,
  SOMATOSENSORY: 3,
  PARIETAL: 4,
  TEMPORAL: 5,
  OCCIPITAL: 6,
  CEREBELLUM: 7,
  BRAINSTEM: 8,
  CINGULATE: 9,
  INSULA: 10,
};
const N_REGIONS = 11;

export const REGION_NAME = [
  'Frontal lobe',
  'Primary motor cortex — left',
  'Primary motor cortex — right',
  'Somatosensory cortex',
  'Parietal lobe',
  'Temporal lobe',
  'Occipital lobe',
  'Cerebellum',
  'Brainstem',
  'Cingulate cortex',
  'Insula',
];

export const REGION_NOTE = [
  'Planning and initiation of movement.',
  'Precentral gyrus. Drives the right side of the body — imagining a right-hand movement suppresses the mu/beta rhythm here.',
  'Precentral gyrus. Drives the left side of the body — imagining a left-hand movement suppresses the mu/beta rhythm here.',
  'Postcentral gyrus. Receives touch and proprioception, immediately behind the central sulcus.',
  'Integrates sensation into a body and spatial map.',
  'Hearing and language comprehension.',
  'Vision.',
  'Coordination and timing of movement.',
  'Carries every signal between brain and body.',
  'Attention, error monitoring, and the emotional weighting of events.',
  'Interoception — the felt state of the body.',
];

/** Which anatomical region a montage region name should light. */
const SITE_TO_REGION = {
  left_motor: REGION.MOTOR_LEFT,
  right_motor: REGION.MOTOR_RIGHT,
  midline_motor: REGION.SOMATOSENSORY,
};

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

const _bs = { x: 0, y: 0, z: 0, fold: 0 };

/**
 * Decode static/cortex.bin — the real fsaverage pial surface, baked by
 * scripts/build_cortex.py. Positions are already in this viewer's frame and
 * scaled to sit just inside the unit scalp sphere; `depth` is FreeSurfer's
 * measured sulcal convexity and `region` the Desikan-Killiany parcellation.
 */
export function decodeCortex(buffer) {
  const dv = new DataView(buffer);
  const magic = String.fromCharCode(
    dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3),
  );
  if (magic !== 'CTX1') throw new Error('cortex.bin: bad magic ' + magic);

  const nV = dv.getUint32(4, true);
  const nT = dv.getUint32(8, true);
  let o = 12;
  const position = new Float32Array(buffer.slice(o, o + nV * 12));
  o += nV * 12;
  const depth = new Float32Array(buffer.slice(o, o + nV * 4));
  o += nV * 4;
  const region = new Uint8Array(buffer.slice(o, o + nV));
  o += nV + ((4 - (nV % 4)) % 4);          // indices are 4-byte aligned
  const index = new Uint32Array(buffer.slice(o, o + nT * 12));

  return { position, depth, region, index, nV, nT };
}

// ── shaders ───────────────────────────────────────────────────────────────

const TISSUE_VERT = `
  attribute float aRegion;
  attribute float aDepth;
  uniform float uHi[${N_REGIONS}];
  uniform vec3  uHiCol[${N_REGIONS}];
  varying vec3 vN;
  varying vec3 vP;
  varying vec3 vObj;
  varying float vSulc;
  varying float vHi;
  varying vec3 vHiCol;
  void main() {
    int r = int(aRegion + 0.5);
    vHi = uHi[r];
    vHiCol = uHiCol[r];
    vSulc = aDepth;
    vObj = position;
    vN = normalize(normalMatrix * normal);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vP = mv.xyz;
    gl_Position = projectionMatrix * mv;
  }`;

/**
 * Lit tissue.
 *
 * An engraved version of this was tried and abandoned: hatching at fixed
 * screen angles described nothing, hatching along level sets of convexity came
 * out as a contour map, and rotating each stroke to follow the surface
 * gradient collapsed into grey fur at any real stroke density. A folded
 * surface this fine simply does not survive being drawn in lines — so it is
 * lit instead, and the plate character stays in the paper and the type.
 *
 * Three lights, none of them physical: a warm key, a cool fill from below to
 * keep the underside from dying, and paper bounce along the silhouette so the
 * specimen sits on the page rather than floating over it. The one thing doing
 * the real work is `aDepth` — FreeSurfer's measured sulcal convexity, used
 * directly as occlusion. No shadow pass would be affordable here, and none is
 * needed: the folds are legible because the data already knows how deep they
 * are.
 */
const TISSUE_FRAG = `
  uniform vec3 uLit;
  uniform vec3 uMid;
  uniform vec3 uShade;
  uniform vec3 uPaper;
  uniform vec3 uVessel;
  varying vec3 vN;
  varying vec3 vP;
  varying vec3 vObj;
  varying float vSulc;
  varying float vHi;
  varying vec3 vHiCol;

  float h31(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
  }

  float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = h31(i + vec3(0.0, 0.0, 0.0));
    float n100 = h31(i + vec3(1.0, 0.0, 0.0));
    float n010 = h31(i + vec3(0.0, 1.0, 0.0));
    float n110 = h31(i + vec3(1.0, 1.0, 0.0));
    float n001 = h31(i + vec3(0.0, 0.0, 1.0));
    float n101 = h31(i + vec3(1.0, 0.0, 1.0));
    float n011 = h31(i + vec3(0.0, 1.0, 1.0));
    float n111 = h31(i + vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
  }

  void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(-vP);
    vec3 K = normalize(vec3(-0.48, 0.74, 0.60));   // key, high and to the left
    vec3 F = normalize(vec3(0.70, -0.30, 0.40));   // fill, low and opposite

    float depth = clamp(vSulc, 0.0, 1.0);

    // Wrapped key: tissue scatters, so the terminator is soft rather than a
    // hard edge, and the shadow side keeps some colour.
    float kw = dot(N, K) * 0.5 + 0.5;
    float kd = max(dot(N, K), 0.0);
    float fd = max(dot(N, F), 0.0);

    vec3 col = mix(uShade, uMid, pow(clamp(kw, 0.0, 1.0), 1.20));
    col = mix(col, uLit, kd * 0.80);
    col += uMid * fd * 0.20;

    // Tissue is not one colour. Broad mottling shifts the hue between the two
    // tones, fine grain breaks up the remaining flatness, and without either
    // the surface reads as moulded plastic however well it is lit.
    float mottle = vnoise(vObj * 7.0) * 0.62 + vnoise(vObj * 17.0) * 0.38;
    col *= 0.90 + mottle * 0.20;
    col = mix(col, col * vec3(1.03, 0.97, 0.96), mottle);
    col *= 0.985 + vnoise(vObj * 74.0) * 0.03;

    // Pial vessels: ridged noise, so it forms branching lines rather than
    // blobs. They run over the crowns and disappear into the sulci, which is
    // where the real ones are hidden.
    // Real pial vessels run *along* the sulci and thin out over the crowns, so
    // this is weighted toward the shoulder of a fold rather than its top.
    float ridge = 1.0 - abs(vnoise(vObj * 21.0) * 2.0 - 1.0);
    float shoulder = smoothstep(0.05, 0.42, depth) * (1.0 - smoothstep(0.62, 0.95, depth));
    float vessel = smoothstep(0.945, 0.999, ridge) * (0.25 + shoulder) * 0.40;
    col = mix(col, uVessel, vessel);

    // Sulcal occlusion, straight from the measured convexity. This is what
    // makes the convolutions read.
    col *= 1.0 - depth * 0.62;

    // A damp sheen on the crowns only — a fixed specimen is wet, not glossy.
    float spec = pow(max(dot(reflect(-K, N), V), 0.0), 34.0);
    col += vec3(1.0) * spec * 0.11 * (1.0 - depth);

    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    col = mix(col, uPaper * 0.92, rim * 0.34);

    col = mix(col, vHiCol, clamp(vHi, 0.0, 1.0) * 0.78);
    gl_FragColor = vec4(col, 1.0);
  }`;

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

export class NeuralEnvironment {
  constructor(canvas, geometry, cortex) {
    this.canvas = canvas;
    this.cortex = cortex;
    this.electrodes = geometry.electrodes;
    this.n = this.electrodes.length;
    this.power = new Float32Array(this.n).fill(0.42);
    this.target = new Float32Array(this.n).fill(0.42);
    this.vis = new Float32Array(this.n).fill(0.42);

    this.flashAmount = 0;
    this.flashRegion = -1;
    this.selected = -1;
    this.hovered = -1;
    this.wash = new Float32Array(N_REGIONS);
    this.washEased = new Float32Array(N_REGIONS);
    this.activity = new Float32Array(N_REGIONS);
    this.reportIn = 0;
    /** Set by the page to receive live per-structure activity. */
    this.onActivity = null;

    this.text = [];
    this.glyphs = [];
    this.pins = [];

    this.scroll = 0;
    this.scrollEased = 0;
    // Left lateral view: the one an atlas plate is drawn from.
    this.yaw = -1.30;
    this.pitch = 0.06;
    this.turned = false;
    this.drag = { active: false, x: 0, y: 0, moved: 0 };
    this.pointer = { x: 0, y: 0 };
    this.ndc = new THREE.Vector2();
    this.ray = new THREE.Raycaster();

    this.clock = new THREE.Clock();
    this._col = new THREE.Color();
    this.soft = isSoftwareGL();
    this.uNear = { value: 0.6 };
    this.uFar = { value: 6.5 };

    this._renderer();
    this._scene();
    this._tissue();
    this._cortexMesh(cortex);
    this._cerebellum();
    this._stem();
    this._anchors();
    this._callout();
    this._project();
    this._sites();
    this._margin();
    this._pinwork();
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
    this.cam = new THREE.PerspectiveCamera(44, 1, 0.1, 60);
    this.scene.add(this.cam);
    this.rig = new THREE.Group();
    // Turntable order, so yaw and pitch stay independent and _aimAt can solve
    // for them directly.
    this.rig.rotation.order = 'YXZ';
    this.rig.rotation.z = 0.03;
    this.scene.add(this.rig);
    this.dotTex = inkDot(0.74);
    this.pickable = [];
  }

  /** One material shared by every piece of tissue, so highlights stay in sync. */
  _tissue() {
    this.uHi = new Float32Array(N_REGIONS);
    this.uHiCol = [];
    for (let i = 0; i < N_REGIONS; i++) this.uHiCol.push(new THREE.Color(OXBLOOD));

    this.tissueMat = new THREE.ShaderMaterial({
      uniforms: {
        uHi: { value: this.uHi },
        uHiCol: { value: this.uHiCol },
        // Muted anatomical tones, kept in the paper's family so the specimen
        // belongs to the plate rather than sitting on top of it.
        // Cortex is far paler than it is usually drawn: a pale greyish-pink,
        // nearly beige in the light, cooling toward grey-violet in shadow
        // rather than warming toward brown.
        uLit: { value: new THREE.Color('#ddd4c9') },
        uMid: { value: new THREE.Color('#b2a49c') },
        uShade: { value: new THREE.Color('#585055') },
        uVessel: { value: new THREE.Color('#9c5a52') },
        uPaper: { value: new THREE.Color('#ece3cf') },
      },
      vertexShader: TISSUE_VERT,
      fragmentShader: TISSUE_FRAG,
    });
  }

  /**
   * Builds a mesh by pushing every vertex of a subdivided icosahedron out to a
   * surface, with analytic normals. `map(dx,dy,dz)` returns the surface point
   * and a fold depth; `assign` returns the region id.
   */
  _shell(detail, map, assign) {
    const geo = new THREE.IcosahedronGeometry(1, detail);
    const src = geo.attributes.position.array;
    const count = src.length / 3;

    const pos = new Float32Array(count * 3);
    const nrm = new Float32Array(count * 3);
    const reg = new Float32Array(count);
    const dep = new Float32Array(count);

    const EPS = 0.006;
    const t1 = new THREE.Vector3();
    const t2 = new THREE.Vector3();
    const d = new THREE.Vector3();
    const a = new THREE.Vector3();
    const b1 = new THREE.Vector3();
    const b2 = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const nv = new THREE.Vector3();

    for (let i = 0; i < count; i++) {
      d.set(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]).normalize();

      let s = map(d.x, d.y, d.z);
      a.set(s.x, s.y, s.z);
      const depth = s.fold;

      // Two tangents on the sphere, stepped and re-mapped: the cross product
      // of the resulting edges is a smooth normal that does not depend on how
      // the triangles happen to be laid out.
      t1.set(-d.y, d.x, 0);
      if (t1.lengthSq() < 1e-8) t1.set(1, 0, 0);
      t1.normalize();
      t2.crossVectors(d, t1).normalize();

      b1.copy(d).addScaledVector(t1, EPS).normalize();
      s = map(b1.x, b1.y, b1.z);
      e1.set(s.x - a.x, s.y - a.y, s.z - a.z);

      b2.copy(d).addScaledVector(t2, EPS).normalize();
      s = map(b2.x, b2.y, b2.z);
      e2.set(s.x - a.x, s.y - a.y, s.z - a.z);

      nv.crossVectors(e1, e2).normalize();
      if (nv.dot(a) < 0) nv.negate();
      if (!isFinite(nv.x)) nv.copy(a).normalize();

      pos[i * 3] = a.x; pos[i * 3 + 1] = a.y; pos[i * 3 + 2] = a.z;
      nrm[i * 3] = nv.x; nrm[i * 3 + 1] = nv.y; nrm[i * 3 + 2] = nv.z;
      reg[i] = assign(a.x, a.y, a.z);
      dep[i] = depth;
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setAttribute('aRegion', new THREE.BufferAttribute(reg, 1));
    g.setAttribute('aDepth', new THREE.BufferAttribute(dep, 1));
    geo.dispose();

    const mesh = new THREE.Mesh(g, this.tissueMat);
    this.rig.add(mesh);
    this.pickable.push(mesh);
    return mesh;
  }

  /**
   * The real cortex. Indexed geometry, so `computeVertexNormals` produces
   * genuinely smooth shading across shared vertices — the generated shells
   * below are non-indexed and have to compute normals analytically instead.
   */
  _cortexMesh(cortex) {
    const g = new THREE.BufferGeometry();
    g.setIndex(new THREE.BufferAttribute(cortex.index, 1));
    g.setAttribute('position', new THREE.BufferAttribute(cortex.position, 3));
    g.setAttribute('aDepth', new THREE.BufferAttribute(cortex.depth, 1));
    g.setAttribute('aRegion', new THREE.BufferAttribute(
      Float32Array.from(cortex.region), 1,
    ));
    g.computeVertexNormals();

    this.cerebrum = new THREE.Mesh(g, this.tissueMat);
    this.rig.add(this.cerebrum);
    this.pickable.push(this.cerebrum);

    // Kept for projecting contacts onto the surface and reading their region.
    this.cortexPos = cortex.position;
    this.cortexRegion = cortex.region;
    this.cortexN = cortex.nV;
  }

  /** Folia are far finer and more regular than cortical gyri. */
  _cerebellum() {
    // Placed against the real cortex: tucked below the occipital lobe and
    // behind the temporal lobes. fsaverage's pial surface is cortex only, so
    // the cerebellum and stem stay generated.
    const C = { x: 0, y: -0.42, z: 0.74 };
    const R = { x: 0.36, y: 0.19, z: 0.24 };
    // Below 5 the facets show: it is a small smooth blob next to real cortex,
    // so any faceting on it is the first thing the eye finds.
    const detail = this.soft ? 5 : 6;
    this.cerebellumMesh = this._shell(
      detail,
      (dx, dy, dz) => {
        // Tight horizontal ridges, plus a shallow vermis groove at the midline.
        // Folia frequency has to stay inside what this tessellation can carry;
        // finer than about 26 and it aliases into visible icosahedral facets.
        const folia = Math.sin(dy * 26) * 0.5 + Math.sin(dy * 41 + 1.3) * 0.2;
        const vermis = Math.exp(-(dx * dx) / 0.006);
        const k = 1 + folia * 0.022 - vermis * 0.045;
        _bs.x = C.x + dx * R.x * k;
        _bs.y = C.y + dy * R.y * k;
        _bs.z = C.z + dz * R.z * k;
        // Occlusion only, so just the grooves: lighter than the cortex, which
        // already casts it into shadow.
        _bs.fold = Math.min(1, Math.max(0, -folia) * 0.55 + vermis * 0.45);
        return _bs;
      },
      () => REGION.CEREBELLUM,
    );
  }

  _stem() {
    const top = { x: 0, y: -0.20, z: 0.16 };
    const bot = { x: 0, y: -0.78, z: 0.30 };
    const detail = this.soft ? 4 : 5;
    this.stemMesh = this._shell(
      detail,
      (dx, dy, dz) => {
        // A tapering trunk: `dy` selects the height, the other two the ring.
        const t = (1 - dy) * 0.5;                       // 0 at top, 1 at bottom
        const rad = 0.115 * (1 - 0.42 * t);
        const ring = Math.sqrt(Math.max(1e-4, dx * dx + dz * dz)) || 1;
        const bulge = 1 + 0.10 * Math.exp(-((t - 0.28) ** 2) / 0.02);
        _bs.x = top.x + (bot.x - top.x) * t + (dx / ring) * rad * bulge;
        _bs.y = top.y + (bot.y - top.y) * t;
        _bs.z = top.z + (bot.z - top.z) * t + (dz / ring) * rad * bulge * 0.85;
        _bs.fold = 0.12;
        return _bs;
      },
      () => REGION.BRAINSTEM,
    );
  }

  /**
   * A point on the surface to hang each structure's callout from, and to aim
   * the specimen at when it is selected. The centroid of a folded region sits
   * *inside* the brain, so it is used only as a direction: the anchor is the
   * vertex of that region lying furthest along it.
   */
  _anchors() {
    const pos = this.cortexPos;
    const reg = this.cortexRegion;
    const sum = [];
    const count = new Float32Array(N_REGIONS);
    for (let r = 0; r < N_REGIONS; r++) sum.push(new THREE.Vector3());

    for (let i = 0; i < this.cortexN; i++) {
      const k = i * 3;
      const r = reg[i];
      sum[r].x += pos[k]; sum[r].y += pos[k + 1]; sum[r].z += pos[k + 2];
      count[r] += 1;
    }

    this.anchor = [];
    for (let r = 0; r < N_REGIONS; r++) {
      if (count[r] === 0) { this.anchor.push(null); continue; }
      const dir = sum[r].multiplyScalar(1 / count[r]).normalize();
      let best = -1;
      let bd = -Infinity;
      for (let i = 0; i < this.cortexN; i++) {
        if (reg[i] !== r) continue;
        const k = i * 3;
        const len = Math.hypot(pos[k], pos[k + 1], pos[k + 2]) || 1;
        const d = (pos[k] * dir.x + pos[k + 1] * dir.y + pos[k + 2] * dir.z) / len;
        if (d > bd) { bd = d; best = i; }
      }
      const k = best * 3;
      this.anchor.push(new THREE.Vector3(pos[k], pos[k + 1], pos[k + 2]));
    }

    // These two are generated, so they have no cortical vertices to average.
    this.anchor[REGION.CEREBELLUM] = new THREE.Vector3(0.30, -0.46, 0.82);
    this.anchor[REGION.BRAINSTEM] = new THREE.Vector3(0.12, -0.62, 0.28);
  }

  /**
   * Leader and tick for the selected structure, grown from nothing. The
   * geometry is rewritten every frame from the anchor and the growth
   * parameter, which is cheap at four vertices.
   */
  _callout() {
    const pos = new Float32Array(4 * 3);
    const col = new Float32Array(4 * 3);
    const ink = new Float32Array(4);
    for (let i = 0; i < 4; i++) {
      col[i * 3] = OXBLOOD.r; col[i * 3 + 1] = OXBLOOD.g; col[i * 3 + 2] = OXBLOOD.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));

    this.lead = this._lines([], [], []);
    this.lead.geometry.dispose();
    this.lead.geometry = g;
    // A callout is drawn over the figure, not buried in it.
    this.lead.material.depthTest = false;
    this.lead.renderOrder = 20;
    this.rig.add(this.lead);

    this.leadT = 0;
    this.leadTip = new THREE.Vector3();
  }

  /**
   * Drop each scalp contact onto the real cortex beneath it, by finding the
   * surface vertex whose direction from centre best matches the electrode's.
   * Both are in the same MNE-derived frame, so this is a genuine projection
   * rather than a fitted approximation.
   */
  _project() {
    this.site = [];
    this.normal = [];
    this.siteRegion = new Int32Array(this.n);
    const pos = this.cortexPos;
    const v = new THREE.Vector3();

    for (const e of this.electrodes) {
      const len = Math.hypot(e.x, e.y, e.z) || 1;
      const ex = e.x / len;
      const ey = e.y / len;
      const ez = e.z / len;

      let best = 0;
      let bestDot = -Infinity;
      for (let i = 0; i < this.cortexN; i++) {
        const k = i * 3;
        const r = Math.hypot(pos[k], pos[k + 1], pos[k + 2]) || 1;
        const d = (pos[k] * ex + pos[k + 1] * ey + pos[k + 2] * ez) / r;
        if (d > bestDot) { bestDot = d; best = i; }
      }
      const k = best * 3;
      v.set(pos[k], pos[k + 1], pos[k + 2]);
      this.site.push(v.clone().multiplyScalar(1.035));
      this.normal.push(v.clone().normalize());
      // Which anatomical structure this contact actually sits over. This is
      // what lets the uploaded recording light real brain parts.
      this.siteRegion[this.site.length - 1] = this.cortexRegion[best];
    }

    // Contacts per region, so activity can be averaged rather than summed —
    // a region with eight electrodes must not read as busier than one with two.
    this.regionCount = new Float32Array(N_REGIONS);
    for (let i = 0; i < this.n; i++) this.regionCount[this.siteRegion[i]] += 1;
  }

  /** Electrode markers, sized and darkened by measured band power. */
  _sites() {
    const pos = new Float32Array(this.n * 3);
    const col = new Float32Array(this.n * 3);
    const size = new Float32Array(this.n);
    const ink = new Float32Array(this.n).fill(1);
    for (let i = 0; i < this.n; i++) {
      const s = this.site[i];
      pos[i * 3] = s.x; pos[i * 3 + 1] = s.y; pos[i * 3 + 2] = s.z;
      col[i * 3] = INK.r; col[i * 3 + 1] = INK.g; col[i * 3 + 2] = INK.b;
      size[i] = 0.3;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));
    this.sites = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: {
        uNear: this.uNear, uFar: this.uFar,
        uMap: { value: this.dotTex }, uScale: { value: 90 },
      },
      vertexShader: DOT_VERT, fragmentShader: DOT_FRAG,
      transparent: true, depthWrite: false, vertexColors: true,
    }));
    this.rig.add(this.sites);
  }

  _lines(pos, col, ink) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.Float32BufferAttribute(ink, 1));
    return new THREE.LineSegments(g, new THREE.ShaderMaterial({
      uniforms: { uNear: this.uNear, uFar: this.uFar },
      vertexShader: STROKE_VERT, fragmentShader: STROKE_FRAG,
      transparent: true, depthWrite: false, vertexColors: true,
    }));
  }

  /** The ruled margin the decoded text is written onto, held in camera space. */
  _margin() {
    const max = MAX_CHARS * SEG_PER_GLYPH;
    const pos = new Float32Array(max * 6);
    const col = new Float32Array(max * 6);
    const ink = new Float32Array(max * 2);
    this.marginAt = new Float32Array(max * 2);
    for (let i = 0; i < max * 2; i++) {
      col[i * 3] = INK.r; col[i * 3 + 1] = INK.g; col[i * 3 + 2] = INK.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 1));
    g.setDrawRange(0, 0);
    this.marginCount = 0;
    this.margin = new THREE.LineSegments(g, new THREE.ShaderMaterial({
      uniforms: { uNear: this.uNear, uFar: this.uFar },
      vertexShader: STROKE_VERT, fragmentShader: STROKE_FRAG,
      transparent: true, depthWrite: false, depthTest: false, vertexColors: true,
    }));
    this.margin.renderOrder = 10;
    this.margin.position.set(0, -0.58, -2.0);
    this.cam.add(this.margin);

    const rp = [];
    const rc = [];
    const ri = [];
    const half = (MAX_CHARS * 0.86 * GLYPH_SIZE) / 2;
    for (let s = 0; s < 60; s++) {
      rp.push(-half + (s / 60) * half * 2, -0.055, 0,
        -half + ((s + 1) / 60) * half * 2, -0.055, 0);
      for (let v = 0; v < 2; v++) {
        rc.push(SEPIA.r, SEPIA.g, SEPIA.b);
        ri.push(0.16 + 0.1 * Math.sin(s * 0.7));
      }
    }
    this.rule = this._lines(rp, rc, ri);
    this.rule.material.depthTest = false;
    this.rule.renderOrder = 10;
    this.rule.position.copy(this.margin.position);
    this.cam.add(this.rule);
  }

  _layoutMargin() {
    const g = this.margin.geometry;
    const pos = g.attributes.position.array;
    let n = 0;
    let width = 0;
    for (const ch of this.text) width += strokes(ch).advance * GLYPH_SIZE;
    let x = -width / 2;

    for (let gi = 0; gi < this.text.length; gi++) {
      const { segments, advance } = strokes(this.text[gi]);
      const glyph = this.glyphs[gi];
      glyph.start = n;
      for (const [x1, y1, x2, y2, at] of segments) {
        if (n >= MAX_CHARS * SEG_PER_GLYPH) break;
        pos[n * 6] = x + x1 * GLYPH_SIZE;
        pos[n * 6 + 1] = y1 * GLYPH_SIZE;
        pos[n * 6 + 2] = 0;
        pos[n * 6 + 3] = x + x2 * GLYPH_SIZE;
        pos[n * 6 + 4] = y2 * GLYPH_SIZE;
        pos[n * 6 + 5] = 0;
        this.marginAt[n * 2] = at;
        this.marginAt[n * 2 + 1] = at;
        n++;
      }
      glyph.count = n - glyph.start;
      x += advance * GLYPH_SIZE;
    }
    g.setDrawRange(0, n * 2);
    g.attributes.position.needsUpdate = true;
    this.marginCount = n;
  }

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
    this.pinwork = this._lines([], [], []);
    this.pinwork.geometry.dispose();
    this.pinwork.geometry = g;
    this.rig.add(this.pinwork);
  }

  _layoutPins() {
    const g = this.pinwork.geometry;
    const pos = g.attributes.position.array;
    const col = g.attributes.color.array;
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
    const p1 = new THREE.Vector3();
    const p2 = new THREE.Vector3();

    const push = (q1, q2, pinIndex, tint) => {
      if (n >= this.pinOf.length / 2) return;
      pos[n * 6] = q1.x; pos[n * 6 + 1] = q1.y; pos[n * 6 + 2] = q1.z;
      pos[n * 6 + 3] = q2.x; pos[n * 6 + 4] = q2.y; pos[n * 6 + 5] = q2.z;
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
      a.copy(site);
      b.copy(site).addScaledVector(nrm, LEADER);
      c.copy(b).addScaledVector(nrm, TICK);
      push(a, b, pi, OXBLOOD);
      push(b, c, pi, OXBLOOD);

      right.crossVectors(worldUp, nrm);
      if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
      right.normalize();
      up.crossVectors(nrm, right).normalize();

      for (const [x1, y1, x2, y2] of strokes(pin.label).segments) {
        p1.copy(c).addScaledVector(right, 0.028 + x1 * size)
          .addScaledVector(up, -0.03 + y1 * size);
        p2.copy(c).addScaledVector(right, 0.028 + x2 * size)
          .addScaledVector(up, -0.03 + y2 * size);
        push(p1, p2, pi, INK);
      }
    });

    g.setDrawRange(0, n * 2);
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    this.pinCount = n;
  }

  // -- interaction ----------------------------------------------------------

  _input() {
    const move = (e) => {
      this.pointer.x = (e.clientX / innerWidth) * 2 - 1;
      this.pointer.y = (e.clientY / innerHeight) * 2 - 1;
      this.ndc.set(this.pointer.x, -this.pointer.y);
      if (!this.drag.active) return;
      const dx = e.clientX - this.drag.x;
      const dy = e.clientY - this.drag.y;
      this.drag.moved += Math.abs(dx) + Math.abs(dy);
      if (this.drag.moved > 6) this.turned = true;
      this.yaw += dx * 0.006;
      this.pitch = Math.max(-1.1, Math.min(1.1, this.pitch + dy * 0.005));
      this.drag.x = e.clientX; this.drag.y = e.clientY;
    };
    addEventListener('pointermove', move, { passive: true });

    addEventListener('pointerdown', (e) => {
      if (e.target.closest('a, button, select, input, label, table')) return;
      // Without this the browser starts a text selection and drags a blue
      // highlight across the marginalia while the specimen turns.
      e.preventDefault();
      this.drag = { active: true, x: e.clientX, y: e.clientY, moved: 0 };
      document.body.classList.add('grabbing');
    });

    const stop = () => {
      // A drag that barely moved is a click: select whatever is under it.
      if (this.drag.active && this.drag.moved < 6) this._pick(true);
      this.drag.active = false;
      document.body.classList.remove('grabbing');
    };
    addEventListener('pointerup', stop);
    addEventListener('pointercancel', stop);
  }

  /** Raycast the tissue and report the region under the cursor. */
  _pick(select) {
    this.ray.setFromCamera(this.ndc, this.cam);
    const hits = this.ray.intersectObjects(this.pickable, false);
    let region = -1;
    if (hits.length) {
      const hit = hits[0];
      const attr = hit.object.geometry.attributes.aRegion;
      if (attr && hit.face) region = Math.round(attr.getX(hit.face.a));
    }
    if (select) {
      this.select(this.selected === region ? -1 : region);
    } else {
      this.hovered = region;
    }
    return region;
  }

  /** The element the specimen should be centred in and fitted to. */
  setStage(el) {
    this.stageEl = el || null;
  }

  /**
   * Camera framing for the current stage box. The specimen sits at the world
   * origin, so putting it somewhere other than the middle of the screen means
   * moving the camera the opposite way: one pixel of offset is one
   * `worldPerPixel` of camera shift.
   */
  _frame() {
    const vw = innerWidth;
    const vh = innerHeight;
    let cx = vw / 2;
    let cy = vh / 2;
    let boxH = vh * 0.72;
    let boxW = vw * 0.55;

    if (this.stageEl) {
      // Frame to the part of the stage actually on screen. The canvas is
      // fixed to the viewport, so on a stacked layout the stage can run below
      // the fold — centring on the whole box then crops the specimen.
      const r = this.stageEl.getBoundingClientRect();
      const l = Math.max(0, r.left);
      const t = Math.min(vw, r.right);
      const top = Math.max(0, r.top);
      const bot = Math.min(vh, r.bottom);
      if (t - l > 8 && bot - top > 8) {
        cx = (l + t) / 2;
        cy = (top + bot) / 2;
        boxW = t - l;
        boxH = bot - top;
      }
    }

    const half = Math.tan((this.cam.fov * Math.PI) / 360);
    // Fit by whichever axis binds first, so a tall narrow column does not
    // push the specimen out of its sides.
    const needV = SPECIMEN_HALF / (boxH / vh);
    const needH = SPECIMEN_HALF / ((boxW / vw) * this.cam.aspect);
    const dist = (Math.max(needV, needH) * 1.14) / half;
    const wpp = (2 * dist * half) / vh;

    return {
      dist,
      x: -(cx - vw / 2) * wpp,
      y: (cy - vh / 2) * wpp,
      cx, cy, boxW, boxH, half, vw, vh,
    };
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
    this.sites.material.uniforms.uScale.value = Math.max(64, h * 0.055);
  }

  // -- public ---------------------------------------------------------------

  /** Names in region-id order, for building a key. */
  regionNames() {
    return REGION_NAME.slice(0, N_REGIONS);
  }

  /** Whether any contact sits over this structure, so it can be measured. */
  hasContacts(id) {
    return this.regionCount[id] > 0;
  }

  /** The element to ride the callout tip. */
  setCalloutEl(el) {
    this.calloutEl = el || null;
  }

  /**
   * Aim the specimen so a structure faces the reader.
   *
   * Elevation goes to pitch and azimuth to yaw, in that order, because the rig
   * applies X before Y. Solving pitch as atan2(dy, dz) instead is wrong in a
   * way that only shows at the poles: a frontal structure needs nearly 180
   * degrees of *pitch*, the clamp below flattens that to almost nothing, yaw
   * comes out near zero, and clicking the frontal lobe turns the brain to show
   * its back.
   *
   * Pitch is clamped because aiming a low structure dead-on tips the specimen
   * onto its base and shows the generated cerebellum and stem end-on, a view no
   * plate is drawn from. Yaw carries the rest, and always can: it is a full
   * turntable.
   */
  _aimAt(id) {
    const a = this.anchor[id];
    if (!a) return;
    const d = a.clone().normalize();

    const pitch = Math.max(-0.42, Math.min(0.42,
      Math.asin(Math.max(-1, Math.min(1, d.y)))));
    // Where the anchor sits after that pitch, so yaw solves against the real
    // remaining offset rather than the original direction.
    const uz = d.y * Math.sin(pitch) + d.z * Math.cos(pitch);
    const yaw = Math.atan2(-d.x, uz);

    this.targetPitch = pitch;
    // Take the short way round rather than unwinding several turns.
    let t = yaw;
    while (t - this.yaw > Math.PI) t -= Math.PI * 2;
    while (t - this.yaw < -Math.PI) t += Math.PI * 2;
    this.targetYaw = t;
    this.aiming = true;
    this.turned = true;
  }

  /** Select a structure from the interface, as clicking it would. */
  select(id) {
    const next = (id === this.selected || id == null || id < 0) ? -1 : id;
    this.selected = next;
    if (next >= 0) this._aimAt(next);
    dispatchEvent(new CustomEvent('brain:select', {
      detail: next < 0 ? null : {
        id: next,
        name: REGION_NAME[next],
        note: REGION_NOTE[next],
      },
    }));
  }

  static region(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('right')) return 'left_motor';
    if (l.includes('left')) return 'right_motor';
    return 'midline_motor';
  }

  update(bandPower) {
    if (!bandPower || bandPower.length !== this.n) return;
    for (let i = 0; i < this.n; i++) this.target[i] = bandPower[i];
  }

  flash(label) {
    this.flashSide = NeuralEnvironment.region(label);
    this.flashRegion = SITE_TO_REGION[this.flashSide];
    this.flashAmount = 1;
  }

  /** Live posterior, warming the hemisphere that would produce each class. */
  posterior(map) {
    if (!map) return;
    this.wash.fill(0);
    for (const [label, p] of Object.entries(map)) {
      const r = SITE_TO_REGION[NeuralEnvironment.region(label)];
      // An even split should read as neutral, not as overall brightness.
      this.wash[r] = Math.max(this.wash[r], Math.max(0, p - 0.5) * 2);
    }
  }

  commit(decision) {
    if (!decision) return;
    this.flash(decision.label);
    let site = -1;
    let best = -Infinity;
    for (let i = 0; i < this.n; i++) {
      if (this.electrodes[i].region !== this.flashSide) continue;
      if (this.vis[i] > best) { best = this.vis[i]; site = i; }
    }
    if (site < 0) return;
    const label = (decision.emitted && decision.emitted !== '—')
      ? decision.emitted : (decision.label || '?').charAt(0);
    this.pins.push({ site, label, age: 0 });
    if (this.pins.length > MAX_PINS) this.pins.shift();
    this._layoutPins();
  }

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
    this.wash.fill(0);
    this.text = [];
    this.glyphs = [];
    this.pins = [];
    this._layoutMargin();
    this._layoutPins();
  }

  // -- loop -----------------------------------------------------------------

  _loop() {
    requestAnimationFrame(() => this._loop());
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    if (!this.drag.active) this._pick(false);

    // Turning toward a selected structure. Dragging cancels it, so the reader
    // is never fighting the animation for control.
    if (this.aiming && !this.drag.active) {
      const k = 1 - Math.exp(-dt * 5.5);
      this.yaw += (this.targetYaw - this.yaw) * k;
      this.pitch += (this.targetPitch - this.pitch) * k;
      if (Math.abs(this.targetYaw - this.yaw) < 0.003
        && Math.abs(this.targetPitch - this.pitch) < 0.003) this.aiming = false;
    } else if (this.drag.active) {
      this.aiming = false;
    }

    // Until it is turned by hand the specimen sways about a lateral view
    // rather than spinning freely: a plate should always be recognisable as
    // one, and a free spin catches it at arbitrary, unreadable angles.
    const sway = this.turned ? 0 : 0.30 * Math.sin(t * 0.13);
    this.rig.rotation.x = this.pitch;
    this.rig.rotation.y = this.yaw + sway;

    // The specimen is centred on the stage and stays put: it is a mounted
    // figure, not something the page scrolls past.
    const f = this._frame();
    const ty = f.y + SPECIMEN_CENTRE_Y;
    this.cam.position.set(f.x + this.pointer.x * 0.05, ty - this.pointer.y * 0.04, f.dist);
    this.cam.lookAt(f.x, ty, 0);

    // Keep the written line under the stage rather than under the viewport.
    const wpp2 = (2 * 2.0 * f.half) / f.vh;
    this.margin.position.set(
      (f.cx - f.vw / 2) * wpp2,
      -((f.cy + f.boxH / 2 - 30) - f.vh / 2) * wpp2,
      -2.0,
    );
    this.rule.position.copy(this.margin.position);

    const ease = 1 - Math.exp(-dt * 7);
    this.flashAmount = Math.max(0, this.flashAmount - dt * 1.0);

    // -- contacts -----------------------------------------------------------
    const scol = this.sites.geometry.attributes.color.array;
    const ssize = this.sites.geometry.attributes.aSize.array;
    const sink = this.sites.geometry.attributes.aInk.array;
    for (let i = 0; i < this.n; i++) {
      this.power[i] += (this.target[i] - this.power[i]) * ease;
      const e = this.electrodes[i];
      const v = Math.min(1, Math.max(0,
        this.power[i] + 0.08 * Math.sin(t * 0.8 + e.y * 3.1 + e.x * 2.2)));
      this.vis[i] = v;
      const lit = this.flashAmount > 0 && e.region === this.flashSide
        ? this.flashAmount : 0;
      this._col.copy(INK).lerp(OXBLOOD, lit * 0.9);
      const k = i * 3;
      scol[k] = this._col.r; scol[k + 1] = this._col.g; scol[k + 2] = this._col.b;
      ssize[i] = 0.16 + v * 0.22 + lit * 0.24;
      sink[i] = 0.40 + v * 0.55;
    }
    for (const a of ['color', 'aSize', 'aInk']) {
      this.sites.geometry.attributes[a].needsUpdate = true;
    }

    // -- live activity per structure ----------------------------------------
    // Straight from the uploaded recording: every contact's measured mu/beta
    // power is attributed to the structure it sits over, and averaged.
    this.activity.fill(0);
    for (let i = 0; i < this.n; i++) this.activity[this.siteRegion[i]] += this.vis[i];
    for (let r = 0; r < N_REGIONS; r++) {
      if (this.regionCount[r] > 0) this.activity[r] /= this.regionCount[r];
    }

    // -- region highlights --------------------------------------------------
    for (let r = 0; r < N_REGIONS; r++) {
      this.washEased[r] += (this.wash[r] - this.washEased[r]) * (1 - Math.exp(-dt * 3.5));
      // Resting power sits near 0.42, so subtract it: a structure should only
      // engrave harder when it is actually above its own baseline.
      const live = Math.max(0, this.activity[r] - 0.44) * 1.5;
      let hi = Math.max(live * 0.42, this.washEased[r] * 0.34);
      if (r === this.flashRegion) hi = Math.max(hi, this.flashAmount);
      if (r === this.hovered && r !== this.selected) hi = Math.max(hi, 0.18);
      if (r === this.selected) hi = Math.max(hi, 0.62 + 0.08 * Math.sin(t * 2.4));
      this.uHi[r] = hi;
      // Oxblood is what the recording is doing; verdigris is what you selected.
      this.uHiCol[r].copy(r === this.selected ? VERDIGRIS : OXBLOOD);
    }

    // -- callout ------------------------------------------------------------
    // The leader grows out of the structure and the label rides its tip, the
    // way a plate numbers what it is pointing at.
    const want = this.selected >= 0 && this.anchor[this.selected] ? 1 : 0;
    this.leadT += (want - this.leadT) * (1 - Math.exp(-dt * 6));
    const lp = this.lead.geometry.attributes.position.array;
    const li = this.lead.geometry.attributes.aInk.array;

    if (this.leadT > 0.01 && this.selected >= 0) {
      const a = this.anchor[this.selected];
      const n = a.clone().normalize();
      const grow = this.leadT;
      // Long enough that the tip clears the silhouette — a label sitting on
      // the tissue is unreadable however it is styled.
      const mid = a.clone().addScaledVector(n, 0.62 * grow);
      const tip = mid.clone().addScaledVector(n, 0.18 * grow);

      lp[0] = a.x; lp[1] = a.y; lp[2] = a.z;
      lp[3] = mid.x; lp[4] = mid.y; lp[5] = mid.z;
      lp[6] = mid.x; lp[7] = mid.y; lp[8] = mid.z;
      lp[9] = tip.x; lp[10] = tip.y; lp[11] = tip.z;
      for (let i = 0; i < 4; i++) li[i] = grow;
      this.leadTip.copy(tip).applyMatrix4(this.rig.matrixWorld);
    } else {
      for (let i = 0; i < 4; i++) li[i] = 0;
    }
    this.lead.geometry.attributes.position.needsUpdate = true;
    this.lead.geometry.attributes.aInk.needsUpdate = true;

    if (this.calloutEl) {
      if (this.leadT > 0.02) {
        this.rig.updateMatrixWorld();
        const p = this.leadTip.clone().project(this.cam);
        // A fixed-length leader can point anywhere, including straight off the
        // page, so the label is held inside the stage even when its tip is not.
        const f = this._frame();
        const padX = 18;
        const padY = 14;
        const x = Math.max(f.cx - f.boxW / 2 + padX, Math.min(
          f.cx + f.boxW / 2 - padX, (p.x * 0.5 + 0.5) * innerWidth));
        const y = Math.max(f.cy - f.boxH / 2 + padY, Math.min(
          f.cy + f.boxH / 2 - padY, (-p.y * 0.5 + 0.5) * innerHeight));
        // Read back toward the figure when the tip is near the right edge.
        const flip = x > f.cx + f.boxW / 2 - 250;
        this.calloutEl.style.transform =
          `translate(${Math.round(x)}px, ${Math.round(y)}px)`
          + (flip ? ' translateX(-100%)' : '');
        this.calloutEl.dataset.flip = flip ? '1' : '0';
        this.calloutEl.style.opacity = Math.max(0, (this.leadT - 0.35) / 0.65).toFixed(2);
        this.calloutEl.hidden = false;
      } else if (!this.calloutEl.hidden) {
        this.calloutEl.hidden = true;
      }
    }

    // Report to the interface at a readable rate, not every frame.
    this.reportIn -= dt;
    if (this.reportIn <= 0 && this.onActivity) {
      this.reportIn = 0.12;
      this.onActivity(this.activity, this.selected);
    }

    // -- margin lettering ---------------------------------------------------
    const mink = this.margin.geometry.attributes.aInk.array;
    for (const glyph of this.glyphs) {
      if (glyph.progress < 1) glyph.progress = Math.min(1, glyph.progress + dt * 2.2);
      for (let k = glyph.start; k < glyph.start + glyph.count; k++) {
        const at = this.marginAt[k * 2];
        const on = glyph.progress >= at ? 1 : Math.max(0, 1 - (at - glyph.progress) * 14);
        mink[k * 2] = on;
        mink[k * 2 + 1] = on;
      }
    }
    this.margin.geometry.attributes.aInk.needsUpdate = true;

    // -- pins ---------------------------------------------------------------
    if (this.pinCount) {
      const pk = this.pinwork.geometry.attributes.aInk.array;
      for (const pin of this.pins) pin.age += dt;
      for (let k = 0; k < this.pinCount * 2; k++) {
        const pin = this.pins[this.pinOf[k]];
        const settle = pin ? Math.min(1, pin.age * 3.5) : 0;
        const fade = pin ? Math.max(0.34, 1 - pin.age * 0.045) : 0;
        pk[k] = 0.85 * settle * fade;
      }
      this.pinwork.geometry.attributes.aInk.needsUpdate = true;
    }

    this.r.render(this.scene, this.cam);
  }
}

export { NeuralEnvironment as BrainScene };
