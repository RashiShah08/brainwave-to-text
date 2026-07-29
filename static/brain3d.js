/**
 * Page-wide neural environment.
 *
 * The canvas is not a panel on the page — it *is* the page. A fixed,
 * full-viewport scene sits behind all content, and scrolling flies the camera
 * through it rather than scrolling past it.
 *
 * What is rendered is real: 64 contacts at their true montage coordinates,
 * brightness driven by measured mu/beta power, and signal pulses that travel
 * the connection paths between them. When a decision commits, the hemisphere
 * that should respond flares — motor imagery is contralateral, so right-hand
 * imagery lights the left cortex.
 *
 * Kept to a handful of draw calls: instanced contacts, one Points cloud for
 * their glow, one for travelling pulses, one for ambient dust. An earlier
 * version built ~200 separate objects and lost the WebGL context outright.
 */

import * as THREE from './three.module.min.js';

/**
 * A full-viewport multisampled buffer is more than a software rasteriser can
 * hold — it drops the context outright. Probe on a throwaway canvas (the real
 * one only gets one context) and scale the scene down when there is no GPU.
 */
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

const C = {
  cold: new THREE.Color('#16255e'),
  mid: new THREE.Color('#2563eb'),
  warm: new THREE.Color('#22d3ee'),
  hot: new THREE.Color('#5eead4'),
  peak: new THREE.Color('#ecfeff'),
  left: new THREE.Color('#fb7185'),
  right: new THREE.Color('#c4b5fd'),
  axon: new THREE.Color('#1d4ed8'),
};

function softDisc(inner, mid) {
  const s = 128;
  const cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const ctx = cv.getContext('2d');
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0.0, `rgba(255,255,255,${inner})`);
  g.addColorStop(0.28, `rgba(255,255,255,${mid})`);
  g.addColorStop(0.6, 'rgba(255,255,255,0.08)');
  g.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const t = new THREE.CanvasTexture(cv);
  t.needsUpdate = true;
  return t;
}

const POINT_VERT = `
  uniform float uScale;
  attribute float aSize;
  varying vec3 vColor;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * uScale / max(0.15, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const POINT_FRAG = `
  uniform sampler2D uMap;
  varying vec3 vColor;
  void main() {
    vec4 t = texture2D(uMap, gl_PointCoord);
    if (t.a < 0.01) discard;
    gl_FragColor = vec4(vColor, 1.0) * t;
  }`;

export class NeuralEnvironment {
  constructor(canvas, geometry) {
    this.canvas = canvas;
    this.electrodes = geometry.electrodes;
    this.n = this.electrodes.length;
    this.power = new Float32Array(this.n).fill(0.42);
    this.target = new Float32Array(this.n).fill(0.42);
    this.vis = new Float32Array(this.n).fill(0.42);   // displayed level, per frame
    this.flashAmount = 0;
    this.flashSide = null;
    this.flashColor = C.warm.clone();

    this.scroll = 0;          // 0..1 through the document
    this.scrollEased = 0;
    this.yaw = -0.55;
    this.pitch = 0.10;
    this.drag = { active: false, x: 0, y: 0 };
    this.pointer = { x: 0, y: 0 };

    this.clock = new THREE.Clock();
    this._tmp = new THREE.Object3D();
    this._col = new THREE.Color();
    this.soft = isSoftwareGL();

    this._renderer();
    this._scene();
    this._cortex();
    this._axons();
    this._contacts();
    this._glow();
    this._pulses();
    this._field();
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
    this.r.setPixelRatio(this.soft ? 1 : Math.min(devicePixelRatio, 1.75));
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
    this.cam = new THREE.PerspectiveCamera(46, 1, 0.1, 260);
    this.rig = new THREE.Group();
    this.scene.add(this.rig);
    this.glowTex = softDisc(1.0, 0.4);
    this.pulseTex = softDisc(1.0, 0.55);
  }

  /** Wireframe scalp, dark interior, and the interhemispheric midline. */
  _cortex() {
    // Deliberately faint: a strong wireframe sphere reads as a globe and
    // fights the brain. The connection mesh is what should define the form.
    this.shell = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(1.26, 3)),
      new THREE.LineBasicMaterial({
        color: 0x1b3a63, transparent: true, opacity: 0.07,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.shell.scale.set(1, 1.07, 1.14);
    this.rig.add(this.shell);

    const interior = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.0, 4),
      new THREE.MeshBasicMaterial({
        color: 0x050d1a, transparent: true, opacity: 0.9, side: THREE.BackSide,
      }),
    );
    interior.scale.set(1, 1.07, 1.14);
    this.rig.add(interior);

    const arc = [];
    for (let i = 0; i <= 110; i++) {
      const a = (i / 110) * Math.PI;
      arc.push(new THREE.Vector3(0, Math.cos(a) * 1.14, -Math.sin(a) * 1.21));
    }
    this.midline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(arc),
      new THREE.LineBasicMaterial({
        color: 0x7dd3fc, transparent: true, opacity: 0.26,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.rig.add(this.midline);
  }

  /**
   * Connections between contacts — the paths pulses travel, and the thing that
   * actually makes this read as a network rather than a dot cloud. Short edges
   * form the local mesh; a sparse set of long edges reads as association
   * fibres and stops the mesh looking like a lattice.
   */
  _axons() {
    this.nearPairs = [];
    this.paths = [];
    const near = [];
    const far = [];
    for (let i = 0; i < this.n; i++) {
      const a = this.electrodes[i];
      for (let j = i + 1; j < this.n; j++) {
        const b = this.electrodes[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (d < 0.52) {
          near.push(a.x, a.y, a.z, b.x, b.y, b.z);
          this.nearPairs.push([i, j]);
          this.paths.push([i, j]);
        } else if (d < 1.15 && (i * 31 + j * 17) % 11 === 0) {
          far.push(a.x, a.y, a.z, b.x, b.y, b.z);
          this.paths.push([i, j]);
        }
      }
    }

    // Per-vertex colour, refreshed each frame from the power at each end, so
    // an active region lights its own connections instead of the whole mesh
    // sitting at one flat brightness.
    const gn = new THREE.BufferGeometry();
    gn.setAttribute('position', new THREE.Float32BufferAttribute(near, 3));
    gn.setAttribute('color', new THREE.Float32BufferAttribute(
      new Float32Array(near.length), 3,
    ));
    this.links = new THREE.LineSegments(gn, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.rig.add(this.links);

    const gf = new THREE.BufferGeometry();
    gf.setAttribute('position', new THREE.Float32BufferAttribute(far, 3));
    this.tracts = new THREE.LineSegments(gf, new THREE.LineBasicMaterial({
      color: 0x38bdf8, transparent: true, opacity: 0.18,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.rig.add(this.tracts);
  }

  _contacts() {
    this.dots = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.030, 14, 10),
      new THREE.MeshBasicMaterial({ toneMapped: false }),
      this.n,
    );
    this.dots.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.dots.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(this.n * 3), 3,
    );
    for (let i = 0; i < this.n; i++) {
      const e = this.electrodes[i];
      this._tmp.position.set(e.x, e.y, e.z);
      this._tmp.scale.setScalar(1);
      this._tmp.updateMatrix();
      this.dots.setMatrixAt(i, this._tmp.matrix);
      this.dots.setColorAt(i, C.mid);
    }
    this.rig.add(this.dots);
  }

  _glow() {
    const pos = new Float32Array(this.n * 3);
    const col = new Float32Array(this.n * 3);
    const size = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const e = this.electrodes[i];
      pos[i * 3] = e.x; pos[i * 3 + 1] = e.y; pos[i * 3 + 2] = e.z;
      col[i * 3] = C.mid.r; col[i * 3 + 1] = C.mid.g; col[i * 3 + 2] = C.mid.b;
      size[i] = 0.45;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.Float32BufferAttribute(size, 1));
    this.glow = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uMap: { value: this.glowTex }, uScale: { value: 340 } },
      vertexShader: POINT_VERT, fragmentShader: POINT_FRAG,
      transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, vertexColors: true,
    }));
    this.rig.add(this.glow);
  }

  /** Signal pulses travelling along axon paths — what makes it read as a net. */
  _pulses() {
    this.nPulse = Math.min(this.soft ? 70 : 150, this.paths.length);
    this.pulse = [];
    const pos = new Float32Array(this.nPulse * 3);
    const col = new Float32Array(this.nPulse * 3);
    const size = new Float32Array(this.nPulse);
    for (let k = 0; k < this.nPulse; k++) {
      this.pulse.push({
        path: (Math.random() * this.paths.length) | 0,
        t: Math.random(),
        speed: 0.22 + Math.random() * 0.5,
      });
      size[k] = 0.24 + Math.random() * 0.18;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.Float32BufferAttribute(size, 1));
    this.pulses = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uMap: { value: this.pulseTex }, uScale: { value: 300 } },
      vertexShader: POINT_VERT, fragmentShader: POINT_FRAG,
      transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, vertexColors: true,
    }));
    this.rig.add(this.pulses);
  }

  /** Deep ambient field so flying through the space reads as depth. */
  _field() {
    const n = this.soft ? 420 : 1100;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const r = 3 + Math.pow(Math.random(), 0.6) * 26;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
      pos[i * 3 + 1] = r * Math.cos(ph) * 0.7;
      pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      c.copy(C.mid).lerp(C.warm, Math.random()).multiplyScalar(0.30 + Math.random() * 0.5);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
      size[i] = 0.05 + Math.random() * 0.13;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.Float32BufferAttribute(size, 1));
    this.field = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uMap: { value: this.glowTex }, uScale: { value: 300 } },
      vertexShader: POINT_VERT, fragmentShader: POINT_FRAG,
      transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, vertexColors: true,
    }));
    this.scene.add(this.field);
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

    // Only drag when the gesture starts on the background, so page controls
    // keep working normally.
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
    const s = Math.max(240, h * 0.62);
    this.glow.material.uniforms.uScale.value = s;
    this.pulses.material.uniforms.uScale.value = s;
    this.field.material.uniforms.uScale.value = s;
  }

  // -- public ---------------------------------------------------------------

  update(bandPower) {
    if (!bandPower || bandPower.length !== this.n) return;
    for (let i = 0; i < this.n; i++) this.target[i] = bandPower[i];
  }

  flash(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('right')) { this.flashSide = 'left_motor'; this.flashColor = C.left.clone(); }
    else if (l.includes('left')) { this.flashSide = 'right_motor'; this.flashColor = C.right.clone(); }
    else { this.flashSide = 'midline_motor'; this.flashColor = C.peak.clone(); }
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

    // Scroll flies the camera: it starts outside the head and descends into
    // the network as the page goes down.
    this.scrollEased += (this.scroll - this.scrollEased) * (1 - Math.exp(-dt * 4));
    const s = this.scrollEased;
    const dist = 2.85 - s * 1.35;
    const height = 0.25 + s * 0.55;

    if (!this.drag.active) this.yaw += dt * 0.055;
    // Sits right of the hero copy at the top, then centres as you descend.
    this.rig.position.x = 0.55 * (1 - s);
    this.rig.rotation.set(this.pitch + s * 0.35, this.yaw, 0);
    this.field.rotation.y = -this.yaw * 0.25;

    this.cam.position.set(
      this.pointer.x * 0.22,
      height - this.pointer.y * 0.16,
      dist,
    );
    this.cam.lookAt(0, s * 0.12, 0);

    const ease = 1 - Math.exp(-dt * 7);
    this.flashAmount = Math.max(0, this.flashAmount - dt * 1.4);

    const gcol = this.glow.geometry.attributes.color.array;
    const gsize = this.glow.geometry.attributes.aSize.array;

    for (let i = 0; i < this.n; i++) {
      this.power[i] += (this.target[i] - this.power[i]) * ease;
      const e0 = this.electrodes[i];
      // A slow travelling shimmer so the montage is never a flat field of
      // identical dots, at rest or mid-stream.
      const v = Math.min(1, Math.max(0,
        this.power[i] + 0.13 * Math.sin(t * 0.85 + e0.y * 3.1 + e0.x * 2.2)));
      this.vis[i] = v;

      if (v < 0.38) this._col.copy(C.cold).lerp(C.mid, v / 0.38);
      else if (v < 0.68) this._col.copy(C.mid).lerp(C.warm, (v - 0.38) / 0.30);
      else this._col.copy(C.warm).lerp(C.hot, (v - 0.68) / 0.32);
      if (v > 0.90) this._col.lerp(C.peak, (v - 0.90) / 0.10);

      let boost = 0;
      if (this.flashAmount > 0 && this.electrodes[i].region === this.flashSide) {
        boost = this.flashAmount;
        this._col.lerp(this.flashColor, 0.85 * boost);
      }

      const breathe = 1 + 0.10 * Math.sin(t * 2.1 + i * 0.5);
      const e = this.electrodes[i];
      this._tmp.position.set(e.x, e.y, e.z);
      this._tmp.scale.setScalar((0.8 + v * 0.75 + boost * 1.7) * breathe);
      this._tmp.updateMatrix();
      this.dots.setMatrixAt(i, this._tmp.matrix);
      this.dots.setColorAt(i, this._col);

      const k = i * 3;
      const lift = 0.55 + v * 1.9 + boost * 2.6;
      gcol[k] = this._col.r * lift;
      gcol[k + 1] = this._col.g * lift;
      gcol[k + 2] = this._col.b * lift;
      gsize[i] = (0.40 + v * 0.68 + boost * 1.25) * breathe;
    }

    // Light each edge from the activity at its own two ends.
    const lcol = this.links.geometry.attributes.color.array;
    for (let k = 0; k < this.nearPairs.length; k++) {
      const [ia, ib] = this.nearPairs[k];
      const j = k * 6;
      for (let end = 0; end < 2; end++) {
        const idx = end === 0 ? ia : ib;
        const lit = 0.30 + this.vis[idx] * 1.05;
        this._col.copy(C.axon).lerp(C.warm, this.vis[idx]).multiplyScalar(lit);
        lcol[j + end * 3] = this._col.r;
        lcol[j + end * 3 + 1] = this._col.g;
        lcol[j + end * 3 + 2] = this._col.b;
      }
    }
    this.links.geometry.attributes.color.needsUpdate = true;

    // Advance pulses along their axon paths.
    const ppos = this.pulses.geometry.attributes.position.array;
    const pcol = this.pulses.geometry.attributes.color.array;
    for (let k = 0; k < this.nPulse; k++) {
      const p = this.pulse[k];
      p.t += dt * p.speed;
      if (p.t > 1) { p.t = 0; p.path = (Math.random() * this.paths.length) | 0; }
      const [ia, ib] = this.paths[p.path];
      const a = this.electrodes[ia];
      const b = this.electrodes[ib];
      const u = p.t;
      const j = k * 3;
      ppos[j] = a.x + (b.x - a.x) * u;
      ppos[j + 1] = a.y + (b.y - a.y) * u;
      ppos[j + 2] = a.z + (b.z - a.z) * u;
      const energy = (this.power[ia] + this.power[ib]) * 0.5;
      const fade = Math.sin(u * Math.PI);
      this._col.copy(C.warm).lerp(C.peak, energy).multiplyScalar(fade * (1.1 + energy * 1.4));
      pcol[j] = this._col.r; pcol[j + 1] = this._col.g; pcol[j + 2] = this._col.b;
    }

    this.dots.instanceMatrix.needsUpdate = true;
    if (this.dots.instanceColor) this.dots.instanceColor.needsUpdate = true;
    this.glow.geometry.attributes.color.needsUpdate = true;
    this.glow.geometry.attributes.aSize.needsUpdate = true;
    this.pulses.geometry.attributes.position.needsUpdate = true;
    this.pulses.geometry.attributes.color.needsUpdate = true;

    this.tracts.material.opacity = 0.11 + 0.10 * (0.5 + 0.5 * Math.sin(t * 0.7 + 1.4));
    this.shell.material.opacity = 0.05 + 0.04 * (0.5 + 0.5 * Math.sin(t * 0.6));
    this.midline.material.opacity = 0.20 + 0.32 * this.flashAmount;

    this.r.render(this.scene, this.cam);
  }
}

/** Backwards-compatible alias — the page used to mount a boxed scene. */
export { NeuralEnvironment as BrainScene };
