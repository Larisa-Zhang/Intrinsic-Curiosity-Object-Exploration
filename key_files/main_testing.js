// main_test.js — parallel testing mode for the parallel-training architecture
// Mirrors main.js (training) but with testing-specific differences:
//   • Loads from /public/testing_models/ (separate model set)
//   • Lazy GLB loading (eager: false) — avoids bundling test models into the chunk
//   • Deterministic initial rotation per model (hash-based, reproducible across runs)
//   • STEPS_PER_MODEL = 90 (longer evaluation window per model)
//   • Posts directly to http://127.0.0.1:5000 (matches server_test.py, no Vite proxy)
//
// Launch:
//   ACTIVE_EXP=exp_1 PORT=5001 python server_test.py
//   VITE_PORT=5173 npx vite          (open at http://localhost:5173/?seed=42)

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
if (window.__AUTO_RUN_STARTED__) throw new Error("Auto-run already started (HMR duplicate).");
window.__AUTO_RUN_STARTED__ = true;

// =========================================
// Seeded RNG (mulberry32)
// =========================================
function mulberry32(seed) {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const params    = new URLSearchParams(window.location.search);
const seedParam = Number.parseInt(params.get("seed") ?? "42", 10);
const FRONT_SEED = Number.isFinite(seedParam) ? seedParam : 42;
const rng = mulberry32(FRONT_SEED);

console.log("🌱 Front-end SEED =", FRONT_SEED);

// =========================================
// Backend URL — direct (no Vite proxy)
// =========================================
const BACKEND_PORT = Number.parseInt(params.get("backend_port") ?? "5000", 10);
const BACKEND = `http://127.0.0.1:${BACKEND_PORT}`;
console.log(`🔌 Backend: ${BACKEND}`);

// =========================================
// Angle helpers
// =========================================
function toDegNorm(rad) {
  const deg = THREE.MathUtils.radToDeg(rad);
  return (deg % 360 + 360) % 360;
}
function wrap180(deg) {
  return ((deg + 180) % 360 + 360) % 360 - 180;
}
function getWorldYPRDeg(obj) {
  const q = new THREE.Quaternion();
  obj.getWorldQuaternion(q);
  const e = new THREE.Euler(0, 0, 0, 'YXZ');
  e.setFromQuaternion(q, 'YXZ');
  return { yaw: toDegNorm(e.y), pitch: toDegNorm(e.x), roll: toDegNorm(e.z) };
}

// =========================================
// Screenshot helper
// =========================================
function cropCenterImage(renderer, cropSize = 350) {
  const src  = renderer.domElement;
  const left = (src.width  - cropSize) / 2;
  const top  = (src.height - cropSize) / 2;
  const tmp  = document.createElement("canvas");
  tmp.width  = cropSize;
  tmp.height = cropSize;
  tmp.getContext("2d").drawImage(src, left, top, cropSize, cropSize, 0, 0, cropSize, cropSize);
  return tmp.toDataURL("image/png", 0.4);
}

// =========================================
// Model discovery — LAZY (eager: false)
// Testing models live in /public/testing_models/
// =========================================
const modules = import.meta.glob('/public/test_models/*.glb', {
  query: '?url',
  import: 'default',
  eager: false,   // <-- lazy: don't bundle all test models upfront
});

const modelPaths = Object.keys(modules);
const modelList  = modelPaths.map(p => p.split('/').pop());

async function getUrlByName(name) {
  const path = modelPaths.find(p => p.endsWith('/' + name));
  if (!path) return null;
  return await modules[path]();   // dynamic import → returns URL string
}

// Shuffle with seeded RNG for reproducible ordering
for (let i = modelList.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [modelList[i], modelList[j]] = [modelList[j], modelList[i]];
}
console.log(`✅ Discovered ${modelList.length} testing models (shuffled):`, modelList);

// =========================================
// State
// =========================================
const STEPS_PER_MODEL = 90;   // longer evaluation window than training (50)
const loader = new GLTFLoader();

let model             = null;
let currentModelName  = '';
let stepCount         = 0;
let processedModels   = [];
let currentModelIndex = 0;
let sessionId         = `${Date.now()}-${Math.floor(rng() * 1e6)}-seed${FRONT_SEED}`;
let isLoadingModel    = false;
let isProcessing      = false;
let lastActionTime    = null;

// =========================================
// Scene setup
// =========================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcfcfcf);

const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshPhongMaterial({ color: 0xcbcbcb, specular: 0x474747 })
);
plane.rotation.x = -Math.PI / 2;
plane.position.y = -20;
plane.receiveShadow = true;
scene.add(plane);
scene.add(new THREE.HemisphereLight(0xffffff, 0xe0e0e0, 0.8));
addShadowedLight(-10, 2, 3, 0xffffff, 2.8);
addShadowedLight(  2, 1, 2, 0xffffff, 1.5);

function addShadowedLight(x, y, z, color, intensity) {
  const light = new THREE.DirectionalLight(color, intensity);
  light.position.set(x, y, z);
  light.castShadow = true;
  light.shadow.mapSize.set(2048, 2048);
  const d = 10;
  Object.assign(light.shadow.camera, { left: -d, right: d, top: d, bottom: -d, near: 0.1, far: 50 });
  light.shadow.bias = -0.0005;
  scene.add(light);
  return light;
}

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(window.innerWidth, window.innerHeight, false);
document.body.appendChild(renderer.domElement);
renderer.outputColorSpace    = THREE.SRGBColorSpace;
renderer.toneMapping         = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled   = true;
renderer.shadowMap.type      = THREE.PCFSoftShadowMap;
renderer.setAnimationLoop(() => renderer.render(scene, camera));

const pmrem = new THREE.PMREMGenerator(renderer);
pmrem.compileEquirectangularShader();
new EXRLoader()
  .setDataType(THREE.FloatType)
  .setPath('/hdrs/')
  .load('table_mountain_1_puresky_4k.exr', (exrTex) => {
    const envMap = pmrem.fromEquirectangular(exrTex).texture;
    scene.environment = envMap;
    scene.background  = envMap;
    exrTex.dispose();
  }, undefined, (err) => console.error('❌ EXR load failed:', err));

// =========================================
// Deterministic initial rotation (hash-based)
// Same function as the original testing main.js —
// ensures the same model always starts at the same angle,
// making test runs comparable across different sessions.
// =========================================
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function getInitialRotation(name) {
  const hashX = hashString('test'+name);
  const rotationsX = Math.floor(hashX % (360/5)*5);
  const angleX = THREE.MathUtils.degToRad(rotationsX);

  const hashY = hashString('test/' + name);
  const rotationsY = Math.floor(hashY % (360/5)*5);
  const angleY = THREE.MathUtils.degToRad(rotationsY);
  return { angleX, angleY };
}

// =========================================
// Memory management
// =========================================
function disposeModel(root) {
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    obj.geometry?.dispose();
    const mat = obj.material;
    (Array.isArray(mat) ? mat : [mat]).forEach(disposeMaterial);
  });
}
function disposeMaterial(mat) {
  if (!mat) return;
  for (const key in mat) {
    const val = mat[key];
    if (val?.isTexture) val.dispose();
  }
  mat.dispose?.();
}

// =========================================
// Frame helpers
// =========================================
async function waitFrames(n = 5) {
  for (let i = 0; i < n; i++) await new Promise(requestAnimationFrame);
}
function nextFrame() {
  return new Promise(requestAnimationFrame);
}

// =========================================
// Network helpers
// =========================================
async function sendInitRow(name, init) {
  try {
    const res = await fetch('/api/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId,
        frontend_seed: FRONT_SEED,
        modelName: name,
        actionId: -1,
        initialAngles: { yaw: init.yaw, pitch: init.pitch },
        s_t_img: '', s_t1_img: '',
        imgData1: 'data:image/png;base64,',
        imgData2: 'data:image/png;base64,',
      })
    });
    if (!res.ok) throw new Error('Upload failed');
  } catch (e) {
    console.warn('Init row POST failed (non-fatal):', e);
  }
}

function saveProgressLog() {
  fetch(`${BACKEND}/save-progress`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ processedModels }),
  })
  .then(res => { if (!res.ok) throw new Error('Failed'); console.log('✅ Progress saved'); })
  .catch(err => console.error('❌ Error saving progress:', err));
}

// =========================================
// UI
// =========================================
function generateFilename(groupId, suffix) { return `${groupId}_${suffix}.png`; }

function updateStepCountdownUI() {
  const el = document.getElementById('countdown-timer');
  if (el) el.textContent = `[TEST] ${currentModelName} | Step: ${stepCount}/${STEPS_PER_MODEL}`;
}

function logUI(message) {
  console.log(message);
  let el = document.getElementById('log');
  if (!el) {
    el = document.createElement('div');
    el.id = 'log';
    el.style.cssText = `
      position: fixed; top: 10px; left: 10px;
      background: rgba(0,0,0,0.7); color: #00ffcc;
      padding: 15px; font-family: monospace; font-size: 12px;
      max-width: 400px; z-index: 1000; border: 1px solid #00ffcc;`;
    document.body.appendChild(el);
  }
  el.textContent = message;
}

// =========================================
// Camera axes helper
// =========================================
function getCameraRelativeAxes() {
  const direction  = new THREE.Vector3();
  camera.getWorldDirection(direction);
  const worldUp    = new THREE.Vector3(0, 1, 0);
  const cameraRight = new THREE.Vector3().crossVectors(direction, worldUp).normalize();
  const cameraUp    = new THREE.Vector3().crossVectors(cameraRight, direction).normalize();
  return { cameraRight, cameraUp };
}

// =========================================
// Model loading
// =========================================
async function loadModel(name) {
  isLoadingModel = true;   // guard: block actions until model is ready
  console.log(`⏳ Loading test model: ${name} ...`);

  scene.children
    .filter(obj => obj.userData?.isModel)
    .forEach(obj => { disposeModel(obj); scene.remove(obj); });
  model = null;
  isProcessing = false;
  renderer.renderLists?.dispose?.();

  currentModelName = name;
  const url = await getUrlByName(name);
  if (!url) {
    console.error('❌ URL not found for model:', name);
    isLoadingModel = false;
    return;
  }

  loader.load(url, (gltf) => {
    model = gltf.scene;
    model.userData.isModel = true;
    model.scale.set(0.5, 0.5, 0.5);
    model.position.set(0, 0, -2.5);

    // Deterministic rotation — same starting angle every test run for this model
    const { angleX, angleY } = getInitialRotation(name);
    model.rotation.set(angleX, angleY, 0);

    const init = getWorldYPRDeg(model);
    model.userData.initialAngles = { yaw: init.yaw, pitch: init.pitch };
    sendInitRow(name, init);

    scene.add(model);
    console.log(`🚀 Test model ${name} loaded, starting evaluation...`);

    (async () => {
      await waitFrames(8);
      isLoadingModel = false;
      const firstAction = Math.floor(rng() * 4);
      recordStepAndAct(firstAction);
    })();
  }, undefined, (err) => {
    console.error('❌ Model load failed:', err);
    isLoadingModel = false;
  });
}

function loadNextModel() {
  if (currentModelIndex >= modelList.length) {
    console.log('✓ All test models evaluated! Restarting...');
    currentModelIndex = 0;
    processedModels   = [];
    saveProgressLog();
  }

  const name = modelList[currentModelIndex];
  currentModelIndex++;
  stepCount = 0;
  sessionId = `${Date.now()}-${Math.floor(rng() * 1e6)}-seed${FRONT_SEED}`;

  logUI(`[TEST] Loading: ${name} | 0/${STEPS_PER_MODEL} steps`);
  loadModel(name);
}

// =========================================
// Core action loop
// =========================================
function applyAction(actionId) {
  const { cameraRight, cameraUp } = getCameraRelativeAxes();
  const step = THREE.MathUtils.degToRad(5);
  switch (actionId) {
    case 0: model.rotateOnWorldAxis(cameraRight, -step); break;  // Up
    case 1: model.rotateOnWorldAxis(cameraRight,  step); break;  // Down
    case 2: model.rotateOnWorldAxis(cameraUp,    -step); break;  // Left
    case 3: model.rotateOnWorldAxis(cameraUp,     step); break;  // Right
  }
}

async function postStep(actionId, s_t_img, s_t1_img, imgData1, imgData2, after, delta) {
  const res = await fetch('/api/record', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      frontend_seed: FRONT_SEED,
      modelName: currentModelName,
      actionId,
      s_t_img, s_t1_img, imgData1, imgData2,
      sessionId,
      afterAngles: { yaw: after.yaw,  pitch: after.pitch  },
      deltaAngles: { yaw: delta.yaw,  pitch: delta.pitch  },
    })
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

async function onStepComplete(data) {
  stepCount++;
  updateStepCountdownUI();

  if (stepCount >= STEPS_PER_MODEL) {
    processedModels.push(currentModelName);
    console.log('✓ Evaluated:', currentModelName);
    saveProgressLog();
    logUI(`✓ [TEST] ${currentModelName} done! Loading next...`);
    await new Promise(resolve => setTimeout(resolve, 1000));
    loadNextModel();
    return;
  }

  if (typeof data.next_action === 'number') {
    setTimeout(() => simulateAction(data.next_action), 150);
  }
}

// Entry-point action (also called on keydown)
async function recordStepAndAct(actionId) {
  if (!model || isProcessing || isLoadingModel) return;
  isProcessing   = true;
  lastActionTime = Date.now();

  const before    = getWorldYPRDeg(model);
  const timestamp = Date.now();
  const rand      = Math.floor(rng() * 1e6);
  const groupId   = `${timestamp}-${rand}`;
  const s_t_img   = `${groupId}_before.png`;
  const imgData1  = cropCenterImage(renderer, 350);

  applyAction(actionId);
  await nextFrame();

  const after  = getWorldYPRDeg(model);
  const delta  = { yaw: wrap180(after.yaw - before.yaw), pitch: wrap180(after.pitch - before.pitch) };
  const s_t1_img = generateFilename(groupId, 'after');
  const imgData2  = cropCenterImage(renderer, 350);

  try {
    const data = await postStep(actionId, s_t_img, s_t1_img, imgData1, imgData2, after, delta);
    console.log(`✅ Recorded: ${currentModelName}, action=${actionId}`);
    updateStepCountdownUI();
    await onStepComplete(data);
  } catch (e) {
    console.error('❌ Recording failed:', e);
  } finally {
    isProcessing = false;
  }
}

// Continuation action (chained from server's next_action)
async function simulateAction(actionId) {
  if (!model || isProcessing || isLoadingModel) return;
  if (stepCount >= STEPS_PER_MODEL) return;
  isProcessing   = true;
  lastActionTime = Date.now();

  const before   = getWorldYPRDeg(model);
  const timestamp = Date.now();
  const rand     = Math.floor(rng() * 1e6);
  const groupId  = `${timestamp}-${rand}`;
  const s_t_img  = `${groupId}_before.png`;
  const imgData1 = cropCenterImage(renderer, 350);

  applyAction(actionId);
  await nextFrame();

  const after  = getWorldYPRDeg(model);
  const delta  = { yaw: wrap180(after.yaw - before.yaw), pitch: wrap180(after.pitch - before.pitch) };
  const s_t1_img = generateFilename(groupId, 'after');
  const imgData2  = cropCenterImage(renderer, 350);

  try {
    const data = await postStep(actionId, s_t_img, s_t1_img, imgData1, imgData2, after, delta);
    await onStepComplete(data);
  } catch (e) {
    console.error('❌ Recording failed:', e);
  } finally {
    isProcessing = false;
  }
}

// =========================================
// Init
// =========================================
window.addEventListener('DOMContentLoaded', () => {
  logUI('[TEST] Auto-Evaluation Started');
  loadNextModel();
});

// Manual override via arrow keys
document.addEventListener('keydown', (event) => {
  switch (event.key) {
    case 'ArrowUp':    recordStepAndAct(0); break;
    case 'ArrowDown':  recordStepAndAct(1); break;
    case 'ArrowLeft':  recordStepAndAct(2); break;
    case 'ArrowRight': recordStepAndAct(3); break;
  }
});

// Watchdog — recover if chain stalls for >30s
setInterval(() => {
  if (lastActionTime === null) return;
  if (isLoadingModel) return;
  if (Date.now() - lastActionTime > 30000) {
    console.warn("⏰ Stalled >30s, triggering recovery action");
    simulateAction(Math.floor(rng() * 4));
  }
}, 5000);