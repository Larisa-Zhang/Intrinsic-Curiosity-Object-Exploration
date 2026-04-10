import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
if (window.__AUTO_RUN_STARTED__) throw new Error("Auto-run already started (HMR duplicate).");
window.__AUTO_RUN_STARTED__ = true;

// ===== Seeded RNG for reproducibility =====
function mulberry32(seed) {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Seed from URL: ?seed=42
const params = new URLSearchParams(window.location.search);
const seedParam = Number.parseInt(params.get("seed") ?? "42", 10);
const FRONT_SEED = Number.isFinite(seedParam) ? seedParam : 42;
const rng = mulberry32(FRONT_SEED);

console.log("🌱 Front-end SEED =", FRONT_SEED);

// === Heatmap helpers ===
function toDegNorm(rad) {
  const deg = THREE.MathUtils.radToDeg(rad);
  return (deg % 360 + 360) % 360; // 0..360
}
function wrap180(deg) { // 350 -> -10, 181 -> -179, etc.
  return ((deg + 180) % 360 + 360) % 360 - 180;
}
// World-frame yaw/pitch (Euler order YXZ)
function getWorldYPRDeg(obj) {
  const q = new THREE.Quaternion();
  obj.getWorldQuaternion(q);
  const e = new THREE.Euler(0, 0, 0, 'YXZ'); // yaw=Y, pitch=X, roll=Z
  e.setFromQuaternion(q, 'YXZ');
  return { yaw: toDegNorm(e.y), pitch: toDegNorm(e.x), roll: toDegNorm(e.z) };
}

function cropCenterImage(renderer, cropSize = 350) {
  const src = renderer.domElement;
  const w = src.width;
  const h = src.height;

  // Compute center crop coordinates
  const left = (w - cropSize) / 2;
  const top = (h - cropSize) / 2;

  // Prepare temporary canvas
  const tmp = document.createElement("canvas");
  tmp.width = cropSize;
  tmp.height = cropSize;
  const ctx = tmp.getContext("2d");

  // Copy the centered region
  ctx.drawImage(src, left, top, cropSize, cropSize, 0, 0, cropSize, cropSize);

  // Return cropped image as base64
  return tmp.toDataURL("image/png", 0.4);
}

// 🔄 Auto-discover all GLB models under /public/models
const modules = import.meta.glob('/public/models/*.glb', {
  query: '?url',
  import: 'default',
  eager: false
});

const modelPaths = Object.keys(modules);
const modelList = modelPaths.map(p => p.split('/').pop());

async function getUrlByName(name) {
  const path = modelPaths.find(p => p.endsWith('/' + name));
  if (!path) return null;
  return await modules[path](); // returns the url string
}

// ✅ Shuffle modelList in-place (Fisher–Yates)
for (let i = modelList.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [modelList[i], modelList[j]] = [modelList[j], modelList[i]];
}

console.log(`✅ Discovered ${modelList.length} models (shuffled):`, modelList);


// ===== AUTO-COLLECTION STATE =====
const STEPS_PER_MODEL = 90;
const loader = new GLTFLoader();

let model = null;
let currentModelName = '';
let stepCount = 0;
let processedModels = [];
let currentModelIndex = 0;
let sessionId = `${Date.now()}-${Math.floor(rng() * 1e6)}-seed${FRONT_SEED}`;
let isLoadingModel = false;

let lastActionTime = null;
let isProcessing = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcfcfcf);

// 添加 STL 风格地面
const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshPhongMaterial({ color: 0xcbcbcb, specular: 0x474747 })
);
plane.rotation.x = -Math.PI / 2;
plane.position.y = -20;
plane.receiveShadow = true;
scene.add(plane);
scene.add(new THREE.HemisphereLight(0xffffff, 0xe0e0e0, 0.8));

// 主阳光：来自摄像头左后上方
addShadowedLight(-10, 2, 3, 0xffffff, 2.8);

// 补光：来自右后方
addShadowedLight(2, 1, 2, 0xffffff, 1.5);

function addShadowedLight(x, y, z, color, intensity) {
  const directionalLight = new THREE.DirectionalLight(color, intensity);
  directionalLight.position.set(x, y, z);

  // ✅ enable shadow casting
  directionalLight.castShadow = true;

  // ✅ bigger shadow map = sharper, cleaner shadows
  directionalLight.shadow.mapSize.set(2048, 2048);

  // ✅ widen the shadow camera so shadows don't get cut off
  const d = 10;
  directionalLight.shadow.camera.left   = -d;
  directionalLight.shadow.camera.right  =  d;
  directionalLight.shadow.camera.top    =  d;
  directionalLight.shadow.camera.bottom = -d;
  directionalLight.shadow.camera.near   = 0.1;
  directionalLight.shadow.camera.far    = 50;

  // ✅ reduce shadow acne
  directionalLight.shadow.bias = -0.0005;

  scene.add(directionalLight);
  return directionalLight;
}

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(window.innerWidth, window.innerHeight, false);
document.body.appendChild(renderer.domElement);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0; 
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);
});

const pmrem = new THREE.PMREMGenerator(renderer);
pmrem.compileEquirectangularShader();

new EXRLoader()
  .setDataType(THREE.FloatType)
  .setPath('/hdrs/')
  .load('table_mountain_1_puresky_4k.exr', (exrTex) => {
    const envMap = pmrem.fromEquirectangular(exrTex).texture;
    scene.environment = envMap;
    scene.background = envMap;
    exrTex.dispose();
  }, undefined, (err) => {
    console.error('❌ exr 加载失败:', err);
  });

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  }
  return hash;
}

async function sendInitRow(name, init) {
  try {
    const res = await fetch('http://127.0.0.1:5000/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId,
        frontend_seed: FRONT_SEED,
        modelName: name,
        actionId: -1,
        initialAngles: { yaw: init.yaw, pitch: init.pitch },
        s_t_img: '',
        s_t1_img: '',
        imgData1: 'data:image/png;base64,',
        imgData2: 'data:image/png;base64,'
      })
    });
    
    if (!res.ok) throw new Error('Upload failed');

  } catch (e) {
    console.warn('Init row POST failed (non-fatal):', e);
  }
}

function disposeModel(root) {
  root.traverse((obj) => {
    if (obj.isMesh) {
      if (obj.geometry) obj.geometry.dispose();

      const mat = obj.material;
      if (Array.isArray(mat)) {
        mat.forEach(disposeMaterial);
      } else if (mat) {
        disposeMaterial(mat);
      }
    }
  });
}

function disposeMaterial(mat) {
  // dispose textures on the material
  for (const key in mat) {
    const val = mat[key];
    if (val && val.isTexture) val.dispose();
  }
  mat.dispose?.();
}

async function waitFrames(n = 5) {
  for (let i = 0; i < n; i++) {
    await new Promise(requestAnimationFrame);
  }
}

function nextFrame() {
  return new Promise(requestAnimationFrame);
}

async function loadModel(name) {
  isLoadingModel = true;
  console.log(`⏳ Loading model: ${name} ...`);
  // ✅ Remove + dispose old model(s) properly
  scene.children
    .filter(obj => obj.userData?.isModel)
    .forEach(obj => {
      disposeModel(obj);
      scene.remove(obj);
    });
  model = null;
  isProcessing = false;
  renderer.renderLists?.dispose?.(); // helps three.js clear internal caches

  currentModelName = name;
  const url = await getUrlByName(name);
  if (!url) {
    console.error('❌ URL not found for model:', name);
    isLoadingModel = false;
    return;
  }
  
  loader.load(
    url,
    (gltf) => {
      model = gltf.scene;
      model.userData.isModel = true;
      model.scale.set(0.5, 0.5, 0.5);
      model.position.set(0, 0, -2.5);
      const degree_rotate = 60;
      // Deterministic initial rotation
      const hashX = hashString('test' + name);
      const rotationsX = Math.floor(hashX % (360 / degree_rotate) * degree_rotate);
      const angleRadX = THREE.MathUtils.degToRad(rotationsX);

      const hashY = hashString('test/' + name);
      const rotationsY = Math.floor(hashY % (360 / degree_rotate) * degree_rotate);
      const angleRadY = THREE.MathUtils.degToRad(rotationsY);

      model.rotation.set(angleRadX, angleRadY, 0);

      // Cache initial Y/P and send init row
      const init = getWorldYPRDeg(model);
      model.userData.initialAngles = { yaw: init.yaw, pitch: init.pitch };
      sendInitRow(name, init);

      scene.add(model);

      // ✅ Auto-start first action after model is fully in scene
      console.log(`🚀 Model ${name} loaded, starting first action automatically...`);
      (async () => {
        // ✅ wait a few real render frames (better than fixed milliseconds)
        await waitFrames(8);
        isLoadingModel = false;
        const firstAction = Math.floor(rng() * 4);
        recordStepAndAct(firstAction);
      })();
    },
    undefined,
    (err) => {
      console.error('❌ 模型加载失败:', err);
      isLoadingModel = false; 
    }
  );
}

function loadNextModel() {
  // If we've gone through all models, restart
  if (currentModelIndex >= modelList.length) {
    console.log('✓ All models processed! Restarting...');
    currentModelIndex = 0;
    processedModels = [];
    saveProgressLog();
  }

  const name = modelList[currentModelIndex];
  currentModelIndex++;
  stepCount = 0;
  sessionId = `${Date.now()}-${Math.floor(rng() * 1e6)}-seed${FRONT_SEED}`;

  logUI(`Loading: ${name} | 0/${STEPS_PER_MODEL} steps`);
  loadModel(name);
}

function init() {
  logUI('Auto-Collection Started');
  loadNextModel();
}

// Call init when page loads
window.addEventListener('DOMContentLoaded', () => {
  init();
});

function generateFilename(groupId, suffix) {
  return `${groupId}_${suffix}.png`;
}

function updateStepCountdownUI() {
  const el = document.getElementById('countdown-timer');
  if (el) {
    el.textContent = `${currentModelName} | Step: ${stepCount}/${STEPS_PER_MODEL}`;
  }
}

function logUI(message) {
  console.log(message);
  let logElement = document.getElementById('log');
  if (!logElement) {
    logElement = document.createElement('div');
    logElement.id = 'log';
    logElement.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: #00ff00;
      padding: 15px;
      font-family: monospace;
      font-size: 12px;
      max-width: 400px;
      z-index: 1000;
      border: 1px solid #00ff00;
    `;
    document.body.appendChild(logElement);
  }
  logElement.textContent = message;
}

function saveProgressLog() {
  fetch('http://127.0.0.1:5000/save-progress', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ processedModels }),
  })
  .then(res => {
    if (!res.ok) throw new Error('Failed to save progress');
    console.log('✅ Progress logged to Flask');
  })
  .catch(err => console.error('❌ Error saving progress:', err));
}

function getCameraRelativeAxes() {
  const direction = new THREE.Vector3();
  camera.getWorldDirection(direction);
  const worldUp = new THREE.Vector3(0, 1, 0);
  const cameraRight = new THREE.Vector3().crossVectors(direction, worldUp).normalize();
  const cameraUp = new THREE.Vector3().crossVectors(cameraRight, direction).normalize();
  return { cameraRight, cameraUp };
}

async function recordStepAndAct(actionId) {
  if (!model || isProcessing || isLoadingModel) return;
  isProcessing = true;
  
  // --- Before rotation ---
  const before = getWorldYPRDeg(model);
  lastActionTime = Date.now();
  
  // take BEFORE screenshot
  const modelName = currentModelName;
  const timestamp = Date.now();
  const rand = Math.floor(rng() * 1e6);
  const groupId = `${timestamp}-${rand}`;
  const s_t_img = `${groupId}_before.png`;
  const imgData1 = cropCenterImage(renderer, 350);
  
  // rotate by a small step
  const { cameraRight, cameraUp } = getCameraRelativeAxes();
  const step = THREE.MathUtils.degToRad(5);
  switch (actionId) {
    case 0: model.rotateOnWorldAxis(cameraRight, -step); break; // Up
    case 1: model.rotateOnWorldAxis(cameraRight, step); break;  // Down
    case 2: model.rotateOnWorldAxis(cameraUp, -step); break;    // Left
    case 3: model.rotateOnWorldAxis(cameraUp, step); break;     // Right
  }

  await nextFrame();

  // --- After rotation ---
  const after = getWorldYPRDeg(model);
  const delta = {
    yaw:   wrap180(after.yaw   - before.yaw),
    pitch: wrap180(after.pitch - before.pitch),
  };

  // take AFTER screenshot
  const s_t1_img = generateFilename(groupId, 'after');
  const imgData2 = cropCenterImage(renderer, 350);

  try {
    const res = await fetch('http://127.0.0.1:5000/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frontend_seed: FRONT_SEED,
        modelName: currentModelName,
        actionId,
        s_t_img, s_t1_img, imgData1, imgData2,
        sessionId,
        afterAngles: { yaw: after.yaw, pitch: after.pitch },
        deltaAngles: { yaw: delta.yaw, pitch: delta.pitch }
      })
    });
    
    updateStepCountdownUI();

    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();
    console.log(`✅ Recorded: ${modelName}, ${s_t_img}, ${actionId}, ${s_t1_img}`);
    console.log(data.next_action);
    
    stepCount++;
    updateStepCountdownUI();
    
    if (stepCount >= STEPS_PER_MODEL) {
      processedModels.push(currentModelName);
      console.log('✓ Completed:', currentModelName);
      saveProgressLog();
      logUI(`✓ ${currentModelName} done! Loading next...`);
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      loadNextModel();
    } else {
      // Get next action from AI
      if (data.next_action !== undefined && typeof data.next_action === 'number') {
        setTimeout(() => {
          simulateAction(data.next_action);
        }, 150);
      }
    }
  } catch (e) {
    console.error('❌ Recording failed:', e);
  } finally {
    isProcessing = false;
  }
}

async function simulateAction(actionId) {
  if (!model || isProcessing || isLoadingModel) return;
  if (stepCount >= STEPS_PER_MODEL) return;
  lastActionTime = Date.now();
  isProcessing = true;

  // --- Before rotation ---
  const before = getWorldYPRDeg(model);
  const timestamp = Date.now();
  const rand = Math.floor(rng() * 1e6);
  const groupId = `${timestamp}-${rand}`;
  const s_t_img = `${groupId}_before.png`;
  const imgData1 = cropCenterImage(renderer, 350);

  // Rotate by action
  const { cameraRight, cameraUp } = getCameraRelativeAxes();
  const step = THREE.MathUtils.degToRad(5);
  switch (actionId) {
    case 0: model.rotateOnWorldAxis(cameraRight, -step); break;
    case 1: model.rotateOnWorldAxis(cameraRight, step); break;
    case 2: model.rotateOnWorldAxis(cameraUp, -step); break;
    case 3: model.rotateOnWorldAxis(cameraUp, step); break;
  }

  await nextFrame();

  // --- After rotation ---
  const after = getWorldYPRDeg(model);
  const delta = {
    yaw: wrap180(after.yaw - before.yaw),
    pitch: wrap180(after.pitch - before.pitch),
  };

  const s_t1_img = generateFilename(groupId, 'after');
  const imgData2 = cropCenterImage(renderer, 350);

  try {
    const res = await fetch('http://127.0.0.1:5000/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frontend_seed: FRONT_SEED,
        modelName: currentModelName,
        actionId,
        s_t_img,
        s_t1_img,
        imgData1,
        imgData2,
        sessionId,
        afterAngles: { yaw: after.yaw, pitch: after.pitch },
        deltaAngles: { yaw: delta.yaw, pitch: delta.pitch }
      })
    });

    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();

    stepCount++;
    updateStepCountdownUI();

    // Check if done
    if (stepCount >= STEPS_PER_MODEL) {
      processedModels.push(currentModelName);
      console.log('✓ Completed:', currentModelName);
      saveProgressLog();
      logUI(`✓ ${currentModelName} done! Loading next...`);
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      loadNextModel();
    } else {
      // Chain next action
      if (data.next_action !== undefined && typeof data.next_action === 'number') {
        setTimeout(() => {
          simulateAction(data.next_action);
        }, 150);
      }
    }

  } catch (e) {
    console.error('❌ Recording failed:', e);
  } finally {
    isProcessing = false;
  }
}

let firstKeyPressed = false;
document.addEventListener('keydown', (event) => {
  if (!firstKeyPressed) {
    firstKeyPressed = true;
    console.log('Starting auto-exploration...');
  }
  
  switch (event.key) {
    case 'ArrowUp': recordStepAndAct(0); break;
    case 'ArrowDown': recordStepAndAct(1); break;
    case 'ArrowLeft': recordStepAndAct(2); break;
    case 'ArrowRight': recordStepAndAct(3); break;
  }
});

// Auto-trigger if inactive for 30 seconds
setInterval(() => {
  const now = Date.now();
  if (lastActionTime === null) return;
  if (isLoadingModel) return;   // ✅ ADD HERE
  if (now - lastActionTime > 30000) {
    console.warn("⏰ 超过30秒无操作，自动触发一次 simulateAction");
    const randomAction = Math.floor(rng() * 4);
    simulateAction(randomAction);
  }
}, 5000);