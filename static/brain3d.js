/**
 * 3D cortical visualiser.
 *
 * Every electrode sits at its true montage coordinate and its brightness is
 * driven by real mu/beta power streamed from the decoder. When the left
 * sensorimotor strip lights up, that is genuinely C3 and its neighbours.
 *
 * Built for a handful of draw calls: one InstancedMesh for the 64 contacts and
 * one Points cloud for their glow, rather than a mesh-plus-sprites per
 * electrode. The first version created ~200 objects and lost the WebGL context
 * outright on software renderers, which drew nothing at all.
 *
 * Deliberately unlit — appearance depends only on measured power, never on
 * where a light happens to sit.
 */

import * as THREE from './three.module.min.js';

const C = {
  cold: new THREE.Color('#1b2a6b'),
  mid: new THREE.Color('#2563eb'),
  warm: new THREE.Color('#22d3ee'),
  hot: new THREE.Color('#5eead4'),
  peak: new THREE.Color('#f0fdfa'),
  left: new THREE.Color('#fb7185'),
  right: new THREE.Color('#c4b5fd'),
};

function glowTexture() {
  const s = 128;
  const cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const ctx = cv.getContext('2d');
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0.00, 'rgba(255,255,255,1)');
  g.addColorStop(0.25, 'rgba(255,255,255,0.45)');
  g.addColorStop(0.55, 'rgba(255,255,255,0.12)');
  g.addColorStop(1.00, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const t = new THREE.CanvasTexture(cv);
  t.needsUpdate = true;
  return t;
}

export class BrainScene {
  constructor(container, geometry) {
    this.el = container;
    this.electrodes = geometry.electrodes;
    this.n = this.electrodes.length;
    this.power = new Float32Array(this.n).fill(0.45);
    this.target = new Float32Array(this.n).fill(0.45);
    this.flashAmount = 0;
    this.flashSide = null;
    this.flashColor = C.warm.clone();
    this.autoRotate = true;
    this.spin = { yaw: -0.6, pitch: 0.12 };
    this.zoom = 3.3;
    this.clock = new THREE.Clock();
    this._tmp = new THREE.Object3D();
    this._col = new THREE.Color();

    this._renderer();
    this._scene();
    this._shell();
    this._links();
    this._contacts();
    this._glow();
    this._dust();
    this._input();

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(container);
    this._resize();
    this._loop();
  }

  _renderer() {
    this.r = new THREE.WebGLRenderer({
      antialias: true, alpha: true, powerPreference: 'high-performance',
    });
    this.r.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    this.r.setClearColor(0x000000, 0);
    this.el.appendChild(this.r.domElement);

    // Losing the context silently renders an empty frame, which is exactly the
    // failure that shipped last time. Surface it instead.
    this.r.domElement.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.el.dataset.glLost = '1';
    });
  }

  _scene() {
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    this.rig = new THREE.Group();
    this.scene.add(this.rig);
    this.glowTex = glowTexture();
  }

  _shell() {
    // A dense wireframe reads as a scalp without hiding the contacts inside.
    const shell = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(1.20, 3)),
      new THREE.LineBasicMaterial({
        color: 0x1a4d7a, transparent: true, opacity: 0.16,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    shell.scale.set(1, 1.07, 1.14);
    this.rig.add(shell);
    this.shell = shell;

    // Dark interior so front contacts read brighter than the ones behind.
    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.0, 4),
      new THREE.MeshBasicMaterial({
        color: 0x061426, transparent: true, opacity: 0.88, side: THREE.BackSide,
      }),
    );
    core.scale.set(1, 1.07, 1.14);
    this.rig.add(core);

    // The interhemispheric fissure: a bright midline arc that makes the
    // left/right split legible, which is the whole point of a contralateral
    // highlight.
    const curve = [];
    for (let i = 0; i <= 96; i++) {
      const a = (i / 96) * Math.PI;
      curve.push(new THREE.Vector3(0, Math.cos(a) * 1.13, -Math.sin(a) * 1.20));
    }
    const midline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(curve),
      new THREE.LineBasicMaterial({
        color: 0x7dd3fc, transparent: true, opacity: 0.30,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.rig.add(midline);
    this.midline = midline;
  }

  _links() {
    const pts = [];
    for (let i = 0; i < this.n; i++) {
      const a = this.electrodes[i];
      for (let j = i + 1; j < this.n; j++) {
        const b = this.electrodes[j];
        if (Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) < 0.46) {
          pts.push(a.x, a.y, a.z, b.x, b.y, b.z);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    this.links = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      color: 0x22d3ee, transparent: true, opacity: 0.13,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.rig.add(this.links);
  }

  /** All 64 contacts in a single instanced draw call. */
  _contacts() {
    this.dots = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.032, 14, 10),
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

  /** One additive Points cloud carrying the halo for every contact. */
  _glow() {
    const pos = new Float32Array(this.n * 3);
    const col = new Float32Array(this.n * 3);
    const size = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const e = this.electrodes[i];
      pos[i * 3] = e.x; pos[i * 3 + 1] = e.y; pos[i * 3 + 2] = e.z;
      col[i * 3] = C.mid.r; col[i * 3 + 1] = C.mid.g; col[i * 3 + 2] = C.mid.b;
      size[i] = 0.4;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.Float32BufferAttribute(size, 1));

    this.glow = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uMap: { value: this.glowTex }, uScale: { value: 340 } },
      vertexShader: `
        uniform float uScale;
        attribute float aSize;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * uScale / -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform sampler2D uMap;
        varying vec3 vColor;
        void main() {
          vec4 t = texture2D(uMap, gl_PointCoord);
          gl_FragColor = vec4(vColor, 1.0) * t;
        }`,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      vertexColors: true,
    }));
    this.glow.material.uniforms.uScale.value = 340;
    this.rig.add(this.glow);
  }

  _dust() {
    const n = 420;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const r = 2.4 + Math.random() * 6.5;
      const t = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(t);
      pos[i * 3 + 1] = r * Math.cos(ph) * 0.55;
      pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(t);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.dust = new THREE.Points(g, new THREE.PointsMaterial({
      color: 0x2f7fb5, size: 0.035, transparent: true, opacity: 0.7,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.scene.add(this.dust);
  }

  _input() {
    const el = this.r.domElement;
    let drag = false;
    let last = { x: 0, y: 0 };
    el.style.touchAction = 'none';
    el.style.cursor = 'grab';

    el.addEventListener('pointerdown', (e) => {
      drag = true; last = { x: e.clientX, y: e.clientY };
      this.autoRotate = false; el.style.cursor = 'grabbing';
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      this.spin.yaw += (e.clientX - last.x) * 0.006;
      this.spin.pitch = Math.max(-1.05, Math.min(1.05,
        this.spin.pitch + (e.clientY - last.y) * 0.006));
      last = { x: e.clientX, y: e.clientY };
    });
    const stop = () => { drag = false; el.style.cursor = 'grab'; };
    el.addEventListener('pointerup', stop);
    el.addEventListener('pointercancel', stop);
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom = Math.max(2.2, Math.min(6.0, this.zoom + e.deltaY * 0.002));
    }, { passive: false });
  }

  _resize() {
    const w = this.el.clientWidth || 900;
    const h = this.el.clientHeight || 500;
    this.r.setSize(w, h, false);
    this.cam.aspect = w / h;
    this.cam.updateProjectionMatrix();
    this.glow.material.uniforms.uScale.value = Math.max(220, h * 0.62);
  }

  // -- public ---------------------------------------------------------------

  update(bandPower) {
    if (!bandPower || bandPower.length !== this.n) return;
    for (let i = 0; i < this.n; i++) this.target[i] = bandPower[i];
  }

  /**
   * Motor imagery is contralateral: right-hand imagery flares the LEFT
   * hemisphere. The visual asserts the same physiology the ERD plots measure.
   */
  flash(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('right')) { this.flashSide = 'left_motor'; this.flashColor = C.left.clone(); }
    else if (l.includes('left')) { this.flashSide = 'right_motor'; this.flashColor = C.right.clone(); }
    else { this.flashSide = 'midline_motor'; this.flashColor = C.peak.clone(); }
    this.flashAmount = 1;
  }

  reset() {
    this.power.fill(0.45);
    this.target.fill(0.45);
    this.flashAmount = 0;
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.r.dispose();
    this.r.domElement.remove();
  }

  // -- loop -----------------------------------------------------------------

  _loop() {
    this._raf = requestAnimationFrame(() => this._loop());
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    if (this.autoRotate) this.spin.yaw += dt * 0.2;
    this.rig.rotation.set(this.spin.pitch, this.spin.yaw, 0);
    this.dust.rotation.y = -this.spin.yaw * 0.18;

    // Always look at the centre. Omitting this is what rendered an empty frame.
    this.cam.position.set(0, 0.3, this.zoom);
    this.cam.lookAt(0, 0, 0);

    const ease = 1 - Math.exp(-dt * 7);
    this.flashAmount = Math.max(0, this.flashAmount - dt * 1.4);

    const gcol = this.glow.geometry.attributes.color.array;
    const gsize = this.glow.geometry.attributes.aSize.array;

    for (let i = 0; i < this.n; i++) {
      this.power[i] += (this.target[i] - this.power[i]) * ease;
      const v = this.power[i];

      if (v < 0.38) this._col.copy(C.cold).lerp(C.mid, v / 0.38);
      else if (v < 0.68) this._col.copy(C.mid).lerp(C.warm, (v - 0.38) / 0.30);
      else this._col.copy(C.warm).lerp(C.hot, (v - 0.68) / 0.32);
      if (v > 0.90) this._col.lerp(C.peak, (v - 0.90) / 0.10);

      let boost = 0;
      if (this.flashAmount > 0 && this.electrodes[i].region === this.flashSide) {
        boost = this.flashAmount;
        this._col.lerp(this.flashColor, 0.85 * boost);
      }

      const breathe = 1 + 0.10 * Math.sin(t * 2.2 + i * 0.5);
      const e = this.electrodes[i];

      this._tmp.position.set(e.x, e.y, e.z);
      this._tmp.scale.setScalar((0.80 + v * 0.75 + boost * 1.7) * breathe);
      this._tmp.updateMatrix();
      this.dots.setMatrixAt(i, this._tmp.matrix);
      this.dots.setColorAt(i, this._col);

      const k = i * 3;
      const lift = 0.55 + v * 1.85 + boost * 2.6;
      gcol[k] = this._col.r * lift;
      gcol[k + 1] = this._col.g * lift;
      gcol[k + 2] = this._col.b * lift;
      gsize[i] = (0.42 + v * 0.70 + boost * 1.25) * breathe;
    }

    this.dots.instanceMatrix.needsUpdate = true;
    if (this.dots.instanceColor) this.dots.instanceColor.needsUpdate = true;
    this.glow.geometry.attributes.color.needsUpdate = true;
    this.glow.geometry.attributes.aSize.needsUpdate = true;

    this.links.material.opacity = 0.14 + 0.10 * (0.5 + 0.5 * Math.sin(t * 1.1));
    this.shell.material.opacity = 0.12 + 0.08 * (0.5 + 0.5 * Math.sin(t * 0.65));
    this.midline.material.opacity = 0.22 + 0.30 * this.flashAmount;

    this.r.render(this.scene, this.cam);
  }
}
