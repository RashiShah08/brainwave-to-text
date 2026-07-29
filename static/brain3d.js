/**
 * 3D cortical visualiser.
 *
 * Every electrode sits at its true montage coordinate, and its glow is driven
 * by real mu/beta power streamed from the decoder. Nothing here is decorative
 * geometry: when the left sensorimotor strip lights up, that is genuinely C3
 * and its neighbours desynchronising.
 *
 * Uses core three.js only — the glow is additive sprites rather than a
 * post-processing bloom pass, which keeps the page to a single vendored file.
 */

import * as THREE from './three.module.min.js';

const PALETTE = {
  idle: new THREE.Color(0x2b3a55),
  low: new THREE.Color(0x1e3a8a),
  mid: new THREE.Color(0x22d3ee),
  high: new THREE.Color(0x5eead4),
  hot: new THREE.Color(0xfbbf24),
  left: new THREE.Color(0xf472b6),
  right: new THREE.Color(0x818cf8),
};

/** Radial-gradient sprite used for every glow. Generated, not loaded. */
function glowTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0.0, 'rgba(255,255,255,1)');
  g.addColorStop(0.2, 'rgba(255,255,255,0.65)');
  g.addColorStop(0.5, 'rgba(255,255,255,0.18)');
  g.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

export class BrainScene {
  constructor(container, geometry) {
    this.container = container;
    this.electrodes = geometry.electrodes;
    this.n = this.electrodes.length;
    this.power = new Float32Array(this.n).fill(0.5);
    this.target = new Float32Array(this.n).fill(0.5);
    this.autoRotate = true;
    this.decisionFlash = 0;
    this.flashColor = PALETTE.high.clone();
    this.clock = new THREE.Clock();

    this._initRenderer();
    this._initScene();
    this._buildHead();
    this._buildElectrodes();
    this._buildConnections();
    this._buildStarfield();
    this._bindInput();

    window.addEventListener('resize', () => this._resize());
    this._resize();
    this._animate();
  }

  // ---------------------------------------------------------------- setup --

  _initRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.container.appendChild(this.renderer.domElement);
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x070b14, 0.16);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this.camera.position.set(0, 0.9, 4.2);

    this.rig = new THREE.Group();
    this.scene.add(this.rig);

    this.scene.add(new THREE.AmbientLight(0x2a3550, 1.4));
    const key = new THREE.PointLight(0x5eead4, 40, 20);
    key.position.set(4, 5, 5);
    this.scene.add(key);
    const rim = new THREE.PointLight(0x7dd3fc, 25, 20);
    rim.position.set(-5, -2, -4);
    this.scene.add(rim);

    this.glowTex = glowTexture();
  }

  /** A translucent shell suggesting the scalp, with a wireframe over it. */
  _buildHead() {
    const shell = new THREE.Mesh(
      new THREE.SphereGeometry(1.0, 64, 48),
      new THREE.MeshPhysicalMaterial({
        color: 0x16233d,
        transparent: true,
        opacity: 0.22,
        roughness: 0.35,
        metalness: 0.1,
        transmission: 0.6,
        side: THREE.DoubleSide,
      }),
    );
    shell.scale.set(1.0, 1.05, 1.12); // slightly ovoid, like a head
    this.rig.add(shell);
    this.shell = shell;

    const wire = new THREE.Mesh(
      new THREE.SphereGeometry(1.005, 28, 20),
      new THREE.MeshBasicMaterial({
        color: 0x2dd4bf, wireframe: true, transparent: true, opacity: 0.07,
      }),
    );
    wire.scale.copy(shell.scale);
    this.rig.add(wire);
    this.wire = wire;
  }

  /** One sphere plus one additive glow sprite per electrode. */
  _buildElectrodes() {
    this.nodes = [];
    this.glows = [];
    const sphere = new THREE.SphereGeometry(0.028, 16, 12);

    this.electrodes.forEach((e) => {
      // Push slightly outward so contacts sit on the shell, not inside it.
      const p = new THREE.Vector3(e.x, e.y, e.z).multiplyScalar(1.04);

      const mat = new THREE.MeshStandardMaterial({
        color: PALETTE.idle.clone(),
        emissive: PALETTE.idle.clone(),
        emissiveIntensity: 0.6,
        roughness: 0.4,
      });
      const node = new THREE.Mesh(sphere, mat);
      node.position.copy(p);
      node.userData = e;
      this.rig.add(node);
      this.nodes.push(node);

      const glow = new THREE.Sprite(new THREE.SpriteMaterial({
        map: this.glowTex,
        color: PALETTE.mid.clone(),
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        opacity: 0.35,
      }));
      glow.position.copy(p);
      glow.scale.setScalar(0.22);
      this.rig.add(glow);
      this.glows.push(glow);
    });
  }

  /** Faint lines between nearby electrodes — reads as a sensor net. */
  _buildConnections() {
    const points = [];
    const limit = 0.42;
    for (let i = 0; i < this.n; i++) {
      const a = this.electrodes[i];
      for (let j = i + 1; j < this.n; j++) {
        const b = this.electrodes[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (d < limit) {
          points.push(a.x * 1.04, a.y * 1.04, a.z * 1.04,
                      b.x * 1.04, b.y * 1.04, b.z * 1.04);
        }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    this.links = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      color: 0x2dd4bf, transparent: true, opacity: 0.12,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.rig.add(this.links);
  }

  _buildStarfield() {
    const count = 900;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 8 + Math.random() * 14;
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(p) * Math.cos(t);
      pos[i * 3 + 1] = r * Math.cos(p);
      pos[i * 3 + 2] = r * Math.sin(p) * Math.sin(t);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.stars = new THREE.Points(geo, new THREE.PointsMaterial({
      color: 0x64748b, size: 0.05, transparent: true, opacity: 0.5,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.scene.add(this.stars);
  }

  _bindInput() {
    const el = this.renderer.domElement;
    let dragging = false;
    let last = { x: 0, y: 0 };
    this.spin = { x: 0.12, y: 0 };

    const down = (x, y) => { dragging = true; last = { x, y }; this.autoRotate = false; };
    const move = (x, y) => {
      if (!dragging) return;
      this.spin.x += (x - last.x) * 0.005;
      this.spin.y = Math.max(-1.2, Math.min(1.2, this.spin.y + (y - last.y) * 0.005));
      last = { x, y };
    };
    const up = () => { dragging = false; };

    el.addEventListener('pointerdown', (e) => down(e.clientX, e.clientY));
    window.addEventListener('pointermove', (e) => move(e.clientX, e.clientY));
    window.addEventListener('pointerup', up);
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.position.z = Math.max(2.4, Math.min(7, this.camera.position.z + e.deltaY * 0.002));
    }, { passive: false });
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight || 460;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // --------------------------------------------------------------- public --

  /** Feed one window's normalised per-electrode power (array of 0..1). */
  update(bandPower) {
    if (!bandPower || bandPower.length !== this.n) return;
    for (let i = 0; i < this.n; i++) this.target[i] = bandPower[i];
  }

  /**
   * Flash the hemisphere that should respond to a decoded class.
   * Motor imagery is contralateral, so right-hand imagery highlights the LEFT
   * hemisphere — the visual says the same thing the physiology does.
   */
  flash(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('right')) { this.flashSide = 'left_motor'; this.flashColor = PALETTE.left.clone(); }
    else if (l.includes('left')) { this.flashSide = 'right_motor'; this.flashColor = PALETTE.right.clone(); }
    else { this.flashSide = 'midline_motor'; this.flashColor = PALETTE.hot.clone(); }
    this.decisionFlash = 1.0;
  }

  reset() {
    this.power.fill(0.5);
    this.target.fill(0.5);
    this.decisionFlash = 0;
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }

  // ---------------------------------------------------------------- loop ---

  _animate() {
    this._raf = requestAnimationFrame(() => this._animate());
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    if (this.autoRotate) this.spin.x += dt * 0.18;
    this.rig.rotation.y = this.spin.x;
    this.rig.rotation.x = this.spin.y;
    this.stars.rotation.y = -this.spin.x * 0.15;

    // Ease measured power toward its target so the scene breathes rather than
    // snapping between windows.
    const ease = 1 - Math.exp(-dt * 6);
    this.decisionFlash = Math.max(0, this.decisionFlash - dt * 1.6);

    const c = new THREE.Color();
    for (let i = 0; i < this.n; i++) {
      this.power[i] += (this.target[i] - this.power[i]) * ease;
      const v = this.power[i];

      // Blue -> cyan -> mint as band power rises.
      if (v < 0.5) c.copy(PALETTE.low).lerp(PALETTE.mid, v * 2);
      else c.copy(PALETTE.mid).lerp(PALETTE.high, (v - 0.5) * 2);

      const node = this.nodes[i];
      const glow = this.glows[i];
      const region = node.userData.region;

      let boost = 0;
      if (this.decisionFlash > 0 && region === this.flashSide) {
        boost = this.decisionFlash;
        c.lerp(this.flashColor, 0.75 * boost);
      }

      node.material.color.copy(c);
      node.material.emissive.copy(c);
      node.material.emissiveIntensity = 0.5 + v * 1.9 + boost * 2.2;

      const pulse = 1 + 0.06 * Math.sin(t * 3 + i * 0.4);
      node.scale.setScalar((0.85 + v * 0.9 + boost * 1.1) * pulse);

      glow.material.color.copy(c);
      glow.material.opacity = 0.14 + v * 0.5 + boost * 0.5;
      glow.scale.setScalar((0.16 + v * 0.34 + boost * 0.4) * pulse);
    }

    const breathe = 1 + 0.012 * Math.sin(t * 1.3);
    this.shell.scale.set(1.0 * breathe, 1.05 * breathe, 1.12 * breathe);
    this.links.material.opacity = 0.08 + 0.10 * (0.5 + 0.5 * Math.sin(t * 0.9));

    this.renderer.render(this.scene, this.camera);
  }
}
