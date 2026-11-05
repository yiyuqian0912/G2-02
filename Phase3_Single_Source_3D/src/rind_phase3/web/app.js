import * as THREE from 'three';
import { OrbitControls } from '/assets/vendor/OrbitControls.js';
import { responseVolume } from '/assets/volume.js';

const $ = id => document.getElementById(id);
const FAMILY = {
  barriers: ['屏障', 'Barriers', '▱'], convex_polyhedra: ['凸多面体', 'Convex polyhedra', '◇'],
  corridors: ['走廊', 'Corridors', '⋔'], multiscale_convex: ['多尺度组合', 'Multi-scale convex', '◫'],
  occluder_clusters: ['遮挡簇', 'Occluder clusters', '▧'], prisms: ['棱柱', 'Prisms', '⬡'],
  rotated_boxes: ['旋转方盒', 'Rotated boxes', '▰'], thin_slabs: ['薄板', 'Thin slabs', '▱'],
  wedges: ['楔形体', 'Wedges', '◭'],
};
const AXES = ['x', 'y', 'z'];
const UV = { x: [1, 2], y: [0, 2], z: [0, 1] };
const state = {
  catalog: null, current: null, scene: null, window: null, scope: 'global', focus: 'global',
  sceneVersion: 0, windowVersion: 0, shown: 55, filtered: [],
  axes: Object.fromEntries(AXES.map(axis => [axis, {
    index: 512, image: null, pixels: null, loadedScene: null, loadedIndex: null,
    inFlight: false, timer: null, mesh: null, counts: null,
  }])),
};
let renderer, world, camera, controls, obstacles, selectedBox, allBoxes, frame = 0, toastTimer;
let sceneController, volumeMesh, sourceGroup;
let volumeVersion = 0, volumeBusy = false, volumePending = false;

function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6500);
}
async function json(path, params = {}, signal) {
  const response = await fetch(path + '?' + new URLSearchParams(params), { signal });
  const value = await response.json();
  if (!response.ok) throw Error(value.message || value.error || `HTTP ${response.status}`);
  return value;
}
const percent = ratio => ratio == null ? '—' : (ratio * 100).toFixed(ratio > .999 || ratio < .001 ? 4 : 2);
const number = value => Number(value).toLocaleString('en-US');
const shortNumber = value => value >= 1e6 ? (value / 1e6).toFixed(2) + 'M' : number(value);

function requestRender() { if (renderer && !frame) frame = requestAnimationFrame(render); }
function render() {
  frame = 0;
  const changed = controls.update();
  renderer.render(world, camera);
  if (changed || controls.autoRotate) requestRender();
}
function dispose(object) {
  if (!object) return;
  object.traverse(node => {
    node.geometry?.dispose();
    node.userData.volumeTexture?.dispose();
    const materials = node.material ? (Array.isArray(node.material) ? node.material : [node.material]) : [];
    materials.forEach(material => { material.map?.dispose(); material.dispose(); });
  });
  object.removeFromParent();
}
function axisLabel(text, color, position) {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 96;
  const ctx = canvas.getContext('2d'); ctx.fillStyle = color; ctx.font = '42px monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 48, 48);
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
  sprite.position.set(...position); sprite.scale.setScalar(.072); world.add(sprite);
}
function initThree() {
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.localClippingEnabled = true;
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('aria-label', '三维场景，左键旋转，滚轮缩放，右键平移');
    $('viewport').append(renderer.domElement);
    world = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(43, 1, .002, 25);
    camera.up.set(0, 0, 1); camera.position.set(1.8, -1.7, 1.45);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(.5, .5, .5); controls.enableDamping = true; controls.dampingFactor = .09;
    controls.minDistance = .025; controls.maxDistance = 7; controls.autoRotateSpeed = .7;
    controls.addEventListener('change', requestRender);
    world.add(new THREE.HemisphereLight(0xc6dbf2, 0x303a45, 2.0));
    const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(2, -1, 3); world.add(light);
    const grid = new THREE.GridHelper(1, 10, 0x52677d, 0x2a4057);
    grid.rotation.x = Math.PI / 2; grid.position.set(.5, .5, -.002);
    grid.material.transparent = true; grid.material.opacity = .43; world.add(grid);
    const bounds = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0x4d657f, transparent: true, opacity: .43 }));
    bounds.position.set(.5, .5, .5); world.add(bounds);
    world.add(new THREE.AxesHelper(1.04));
    axisLabel('X', '#ed9096', [1.09, 0, 0]); axisLabel('Y', '#7dceb2', [0, 1.09, 0]);
    axisLabel('Z', '#9ebbf6', [0, 0, 1.09]);
    for (const axis of AXES) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Array(12).fill(0), 3));
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
      geometry.setIndex([0, 1, 2, 0, 2, 3]);
      const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true,
        opacity: .9, depthWrite: false, toneMapped: false });
      const mesh = new THREE.Mesh(geometry, material); mesh.visible = false; mesh.renderOrder = 1;
      state.axes[axis].mesh = mesh; world.add(mesh);
    }
    new ResizeObserver(() => {
      const box = $('viewport').getBoundingClientRect();
      camera.aspect = box.width / box.height; camera.updateProjectionMatrix();
      renderer.setSize(box.width, box.height); requestRender();
    }).observe($('viewport'));
    controls.update(); requestRender();
  } catch (error) {
    console.error(error); $('webgl-error').hidden = false;
    renderer?.dispose(); renderer = null;
  }
}
function windowBox(window, color, opacity = 1) {
  const width = window.side / 1024;
  const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(width, width, width));
  const box = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, opacity,
    transparent: true, depthTest: opacity < 1 }));
  box.position.set(...window.origin.map(v => (v + window.side / 2) / 1024));
  box.renderOrder = opacity === 1 ? 5 : 0;
  return box;
}
function buildGeometry() {
  if (!renderer) return;
  dispose(obstacles); dispose(allBoxes); dispose(selectedBox); selectedBox = null;
  obstacles = new THREE.Group(); allBoxes = new THREE.Group();
  const alpha = Number($('obstacle-opacity').value) / 100;
  state.scene.components.forEach(component => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(component.triangles, 3));
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x9aafc1,
      roughness: .68, metalness: .07, flatShading: true, transparent: true,
      opacity: alpha, side: THREE.DoubleSide, depthWrite: alpha >= .99 }));
    obstacles.add(mesh);
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(component.edges, 3));
    const edges = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({ color: 0xa8c5e1,
      opacity: .40, transparent: true, depthWrite: false }));
    edges.userData.outline = true; edges.visible = $('show-edges').checked; obstacles.add(edges);
  });
  state.scene.windows.forEach(window => allBoxes.add(windowBox(window, 0xd4ae70, .095)));
  allBoxes.visible = $('all-windows').checked; world.add(obstacles, allBoxes);
  applyClipping(); requestRender();
}

function buildSource() {
  dispose(sourceGroup); sourceGroup = null;
  const source = state.scene?.source_position;
  const available = state.scene?.source_status === 'verified_original' &&
    Array.isArray(source) && source.length === 3 && source.every(Number.isFinite);
  $('show-source').disabled = !available; $('locate-source').disabled = !available || !renderer;
  $('source-coordinates').hidden = !available;
  $('source-badge').classList.toggle('available', available);
  $('source-badge').textContent = available ? '原始记录已匹配' : '未提供';
  $('source-summary').textContent = available ? '本地真实源 · 已匹配' : '公开数据隐藏';
  $('source-legend').textContent = available ? '真实驱动源' : '真实源未提供';
  const messages = {
    not_configured: '公开包隐藏源位置。配置本地原始生产记录后可查看；不推测源坐标。',
    record_missing: '本地没有此场景的原始源记录；不显示源标记。',
    identity_rejected: '原始记录与当前场景未通过身份或文件校验；不显示源标记。',
  };
  $('source-status').textContent = available ? '单位世界 XYZ。源只在本地查看器显示，不写入公开数据或模型输入。' :
    messages[state.scene?.source_status] || '没有可用的真实源记录。';
  if (!available) return;
  AXES.forEach((axis, i) => {
    $('source-' + axis).textContent = source[i].toFixed(7);
    $('source-' + axis).title = source[i].toPrecision(17);
  });
  if (!renderer) return;
  sourceGroup = new THREE.Group();
  const point = new THREE.Vector3(...source);
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(.009, 18, 12),
    new THREE.MeshBasicMaterial({color:0xffdf81, depthTest:false, depthWrite:false}));
  sphere.position.copy(point); sphere.renderOrder = 21; sourceGroup.add(sphere);
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const glow = ctx.createRadialGradient(64,64,3,64,64,64);
  glow.addColorStop(0,'#fff5beef'); glow.addColorStop(.2,'#ffcf6baa'); glow.addColorStop(1,'#ffb23b00');
  ctx.fillStyle = glow; ctx.fillRect(0,0,128,128);
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false,depthWrite:false,sizeAttenuation:false}));
  halo.position.copy(point); halo.scale.setScalar(.04); halo.renderOrder = 20; sourceGroup.add(halo);
  const label = document.createElement('canvas'); label.width = 256; label.height = 64;
  const lc = label.getContext('2d'); lc.fillStyle = '#ffdd8a'; lc.font = '500 28px sans-serif';
  lc.textAlign = 'center'; lc.fillText('真实驱动源',128,42);
  const labelTexture = new THREE.CanvasTexture(label); labelTexture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map:labelTexture,depthTest:false,depthWrite:false,sizeAttenuation:false}));
  sprite.position.copy(point).add(new THREE.Vector3(0,0,.055)); sprite.scale.set(.13,.0325,1);
  sprite.renderOrder = 22; sourceGroup.add(sprite);
  const geometry = new THREE.BufferGeometry().setFromPoints([point, new THREE.Vector3(source[0],source[1],0)]);
  const guide = new THREE.Line(geometry,new THREE.LineDashedMaterial({color:0xffd77b,dashSize:.008,
    gapSize:.007,opacity:.5,transparent:true,depthTest:false,depthWrite:false}));
  guide.computeLineDistances(); guide.renderOrder = 20; sourceGroup.add(guide);
  sourceGroup.visible = $('show-source').checked; world.add(sourceGroup); requestRender();
}

function updateVolume() {
  const modeNames = ['蓝色阴影体积','金色受光体积','受光 + 阴影','明暗边界'];
  $('response-summary').textContent = $('show-volume').checked ?
    `${modeNames[Number($('volume-mode').value)]} · ${$('volume-quality').value}³ 预览` : '三维体积已隐藏';
  if (!volumeMesh) return;
  const u = volumeMesh.material.uniforms;
  u.uMode.value = Number($('volume-mode').value);
  u.uDensity.value = Number($('volume-density').value) / 10;
  const crop = $('volume-crop').checked && state.window;
  u.uCropMin.value.set(...(crop ? state.window.origin.map(v => v / 1024) : [0,0,0]));
  u.uCropMax.value.set(...(crop ? state.window.origin.map(v => (v + state.window.side) / 1024) : [1,1,1]));
  volumeMesh.visible = $('show-volume').checked; requestRender();
}

async function loadVolume() {
  const version = ++volumeVersion; dispose(volumeMesh); volumeMesh = null;
  updateVolume(); requestRender();
  // Keep one HTTP request in flight. A change replaces the pending request;
  // aborting fetch would not cancel the server's ongoing packed-bit scan.
  if (volumeBusy) { volumePending = true; return; }
  if (!renderer || !state.scene || !$('show-volume').checked) {
    $('volume-status').textContent = renderer ? '体积已隐藏；精确切片仍可使用。' : '三维体积需要 WebGL 2；精确切片仍可使用。';
    requestRender(); return;
  }
  const sid = state.scene.id, resolution = Number($('volume-quality').value);
  $('volume-status').textContent = `正在读取 ${resolution}³ 体积预览…`;
  volumeBusy = true;
  try {
    const response = await fetch('/api/volume?' + new URLSearchParams({scene:sid,resolution}));
    if (!response.ok) { const value = await response.json(); throw Error(value.error || value.message); }
    if (Number(response.headers.get('X-Volume-Resolution')) !== resolution) throw Error('预览尺寸不一致');
    const labels = new Uint8Array(await response.arrayBuffer());
    if (labels.length !== resolution ** 3 || labels.some(value => value > 2)) throw Error('无效体积数据');
    if (version !== volumeVersion || state.scene?.id !== sid) return;
    volumeMesh = responseVolume(labels,resolution); world.add(volumeMesh); updateVolume();
    $('volume-status').textContent = `✓ ${resolution}³ 全局预览 · 原始场 1024³ · 抽样步长 ${1024/resolution} 体素`;
  } catch (error) {
    if (error.name !== 'AbortError' && version === volumeVersion) {
      $('volume-status').textContent = '体积加载失败：' + error.message;
      toast('体积加载失败：' + error.message);
    }
  } finally {
    volumeBusy = false;
    if (volumePending) { volumePending = false; loadVolume(); }
  }
}
function applyClipping() {
  if (!renderer || !obstacles) return;
  let planes = [];
  if ($('clip-window').checked && state.window) {
    const lo = state.window.origin.map(v => v / 1024);
    const hi = state.window.origin.map(v => (v + state.window.side) / 1024);
    planes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), -lo[0]),
      new THREE.Plane(new THREE.Vector3(-1, 0, 0), hi[0]),
      new THREE.Plane(new THREE.Vector3(0, 1, 0), -lo[1]),
      new THREE.Plane(new THREE.Vector3(0, -1, 0), hi[1]),
      new THREE.Plane(new THREE.Vector3(0, 0, 1), -lo[2]),
      new THREE.Plane(new THREE.Vector3(0, 0, -1), hi[2])];
  }
  obstacles.children.forEach(object => { object.material.clippingPlanes = planes; object.material.needsUpdate = true; });
  requestRender();
}
function cameraTarget() {
  if (state.focus === 'source' && state.scene?.source_position) return {
    center:new THREE.Vector3(...state.scene.source_position), distance:.6,
  };
  if (state.focus === 'window' && state.window) return {
    center: new THREE.Vector3(...state.window.origin.map(v => (v + state.window.side / 2) / 1024)),
    distance: state.window.side / 1024 * 2.8,
  };
  return { center: new THREE.Vector3(.5, .5, .5), distance: 2.65 };
}
function setCamera(mode) {
  if (!renderer) return;
  const { center, distance } = cameraTarget();
  let direction = camera.position.clone().sub(controls.target).normalize();
  if (mode === 'reset') direction.set(1.3, -1.6, 1.0).normalize();
  if (mode === 'xy') direction.set(0, -.00001, 1).normalize();
  if (mode === 'xz') direction.set(0, -1, 0);
  if (mode === 'yz') direction.set(1, 0, 0);
  camera.position.copy(center).addScaledVector(direction, distance); controls.target.copy(center);
  controls.update(); requestRender();
}
function setFocus(focus) {
  state.focus = focus; $('global-view').classList.toggle('active', focus === 'global');
  $('focus-view').classList.toggle('active', focus === 'window'); setCamera();
}

function renderLibrary(reset = true) {
  if (!state.catalog) return;
  if (reset) state.shown = 55;
  const search = $('scene-search').value.trim().toLowerCase();
  state.filtered = state.catalog.scenes.filter(scene => (!search || scene.id.includes(search) ||
    String(scene.ordinal).includes(search) || (FAMILY[scene.family] || []).join(' ').toLowerCase().includes(search)) &&
    (!$('family-filter').value || scene.family === $('family-filter').value) &&
    (!$('split-filter').value || scene.split === $('split-filter').value));
  $('filter-count').textContent = `${state.filtered.length} / ${state.catalog.scene_count}`;
  const fragment = document.createDocumentFragment();
  const visibleRows = state.filtered.slice(0, state.shown);
  const activeRow = state.filtered.find(scene => scene.id === state.current);
  if (activeRow && !visibleRows.includes(activeRow)) visibleRows.unshift(activeRow);
  visibleRows.forEach(scene => {
    const family = FAMILY[scene.family] || [scene.family, scene.family, '◇'];
    const button = document.createElement('button'); button.className = 'scene-item';
    button.dataset.scene = scene.id; button.classList.toggle('active', scene.id === state.current);
    button.setAttribute('role', 'listitem'); button.setAttribute('aria-label', `场景 ${scene.ordinal} ${family[0]} ${scene.id}`);
    const icon = document.createElement('span'); icon.className = 'scene-icon'; icon.textContent = family[2];
    const info = document.createElement('div'); info.className = 'scene-item-info';
    const title = document.createElement('div'); title.className = 'scene-item-title';
    const name = document.createElement('span'); name.textContent = `${String(scene.ordinal).padStart(4, '0')}  ${family[0]}`;
    const split = document.createElement('span'); split.className = 'mini-split'; split.textContent = scene.split;
    title.append(name, split); const id = document.createElement('div'); id.className = 'scene-item-id'; id.textContent = scene.id.slice(0, 12);
    info.append(title, id); button.append(icon, info); fragment.append(button);
  });
  $('scene-list').replaceChildren(fragment); $('load-more').hidden = state.shown >= state.filtered.length;
  if (!state.filtered.length) { const label = document.createElement('p'); label.textContent = '没有匹配的场景'; label.className = 'muted'; $('scene-list').append(label); }
}
function updateURL() {
  if (!state.scene || !state.window) return;
  const params = new URLSearchParams({ scene: state.scene.id, window: state.window.id });
  AXES.forEach(axis => params.set(axis, state.axes[axis].index));
  history.replaceState(null, '', '?' + params);
}
async function loadScene(id, desiredWindow) {
  const version = ++state.sceneVersion; sceneController?.abort(); sceneController = new AbortController();
  state.current = id; state.windowVersion++; state.window = null; state.scene = null;
  volumeVersion++; dispose(volumeMesh); volumeMesh = null;
  dispose(sourceGroup); sourceGroup = null;
  $('source-coordinates').hidden = true; $('source-badge').textContent = '读取中';
  $('show-source').disabled = true; $('locate-source').disabled = true;
  $('volume-status').textContent = '准备读取全局响应…';
  const loadingControls = ['window-select', 'window-size', 'previous-window', 'next-window', 'focus-view', 'center-slices'];
  loadingControls.forEach(key => { $(key).disabled = true; });
  document.body.classList.add('scene-loading'); $('viewport-loading').hidden = false;
  document.body.classList.remove('library-open'); renderLibrary(false);
  AXES.forEach(axis => { state.axes[axis].mesh && (state.axes[axis].mesh.visible = false); });
  try {
    const value = await json('/api/scene', { scene: id }, sceneController.signal);
    if (version !== state.sceneVersion) return;
    state.scene = value; const family = FAMILY[value.family] || [value.family, value.family];
    const row = state.catalog.scenes.find(r => r.id === id);
    $('scene-number').textContent = `RIND3D / SCENE ${String(row.ordinal).padStart(4, '0')}`;
    $('scene-title').textContent = `${family[0]} / ${family[1]}`;
    $('scene-subtitle').textContent = value.id.slice(0, 20) + ' · 1024 × 1024 × 1024';
    $('active-split').textContent = value.split;
    $('geometry-summary').textContent = `${value.components.length} 个几何组件 · ${value.components.reduce((n, c) => n + c.display_faces, 0)} 个显示面`;
    $('obstacle-count').textContent = value.components.length + ' 个组件';
    const count = value.global_counts; $('global-lit').textContent = percent(count.lit / (count.lit + count.shadow)) + '%';
    $('global-solid').textContent = (count.solid / 1024 ** 3 * 100).toFixed(2) + '%';
    $('global-components').textContent = value.components.length;
    buildGeometry(); buildSource(); $('window-size').value = ''; fillWindowOptions();
    const chosen = value.windows.find(w => w.id === desiredWindow) || value.windows[0];
    selectWindow(chosen.id); setFocus('global'); setCamera('reset'); loadVolume();
  } catch (error) {
    if (error.name !== 'AbortError' && version === state.sceneVersion) { toast('场景加载失败：' + error.message); console.error(error); }
  } finally {
    if (version === state.sceneVersion) {
      $('viewport-loading').hidden = true; document.body.classList.remove('scene-loading');
      loadingControls.forEach(key => { $(key).disabled = !state.scene; });
    }
  }
}
function filteredWindows() { return state.scene?.windows.filter(w => !$('window-size').value || w.side === Number($('window-size').value)) || []; }
function fillWindowOptions() {
  $('window-select').replaceChildren(...filteredWindows().map(w => {
    const option = document.createElement('option'); option.value = w.id;
    option.textContent = `窗口 ${String(w.id).padStart(3, '0')} · ${w.side}³`; return option;
  }));
}
function windowStep(delta) {
  const list = filteredWindows(); if (!list.length || !state.window) return;
  const index = list.findIndex(w => w.id === state.window.id);
  selectWindow(list[(index + delta + list.length) % list.length].id);
}
function selectWindow(id) {
  if (!state.scene) return;
  const selected = state.scene.windows.find(w => w.id === id); if (!selected) return;
  const version = ++state.windowVersion; const sid = state.scene.id; state.window = selected;
  $('window-select').value = id; $('window-counter').textContent = `${id + 1} / ${state.scene.windows.length}`;
  $('window-origin').textContent = selected.origin.join(', '); $('window-side').textContent = `${selected.side} × ${selected.side} × ${selected.side}`;
  $('lit-ratio').textContent = '—'; $('lit-count').textContent = '—'; $('shadow-count').textContent = '—';
  $('window-validity').textContent = '正在读取精确体素计数…'; $('window-validity').className = '';
  if (renderer) { dispose(selectedBox); selectedBox = windowBox(selected, 0xffb457); world.add(selectedBox); applyClipping(); }
  updateVolume();
  if ($('follow-window').checked) centerSlices(); else AXES.forEach(axis => { drawSlice(axis); queuePlane(axis); });
  if (state.focus === 'window') setCamera();
  updateURL(); requestRender();
  json('/api/window', { scene: sid, id }).then(counts => {
    if (version !== state.windowVersion || sid !== state.scene.id) return;
    $('lit-ratio').innerHTML = `${percent(counts.lit_ratio)}<small>%</small>`;
    $('balance-lit').style.width = (counts.lit_ratio * 100) + '%';
    $('lit-count').textContent = shortNumber(counts.lit); $('shadow-count').textContent = shortNumber(counts.shadow);
    const extreme = counts.lit_ratio > .99 || counts.lit_ratio < .01;
    $('window-validity').classList.toggle('extreme', extreme);
    $('window-validity').textContent = !counts.informative ? '此窗口未通过信息性检查。' : extreme ?
      `极端明暗比例 · 仍含 ${number(Math.min(counts.lit, counts.shadow))} 个少数类自由体素。` :
      `✓ 同时包含受光与阴影自由体素 · 实体 ${number(counts.solid)}。`;
  }).catch(error => { if (version === state.windowVersion) toast('窗口计数读取失败：' + error.message); });
}
function centerSlices() {
  if (!state.window) return;
  AXES.forEach((axis, i) => setAxis(axis, state.window.origin[i] + Math.floor(state.window.side / 2), true));
}
function setAxis(axis, value, immediate = false) {
  const index = Math.max(0, Math.min(1023, Math.round(Number(value)))); if (!Number.isFinite(index)) return;
  const a = state.axes[axis]; a.index = index;
  $('range-' + axis).value = index; $('number-' + axis).value = index; $('slice-label-' + axis).textContent = index;
  if (a.loadedIndex !== index || a.loadedScene !== state.current) {
    if (a.mesh) a.mesh.visible = false; $('busy-' + axis).hidden = false;
  }
  drawSlice(axis); requestRender(); clearTimeout(a.timer);
  if (immediate) queuePlane(axis); else a.timer = setTimeout(() => queuePlane(axis), 160);
  updateURL();
}
function updatePlaneMesh(axis) {
  const a = state.axes[axis]; if (!renderer || !a.image) return;
  const p = (a.loadedIndex + .5) / 1024;
  const points = axis === 'z' ? [0, 0, p, 1, 0, p, 1, 1, p, 0, 1, p] : axis === 'y' ?
    [0, p, 0, 1, p, 0, 1, p, 1, 0, p, 1] : [p, 0, 0, p, 1, 0, p, 1, 1, p, 0, 1];
  const attribute = a.mesh.geometry.getAttribute('position'); attribute.array.set(points); attribute.needsUpdate = true;
  a.mesh.geometry.computeBoundingSphere();
  a.mesh.material.map?.dispose();
  const texture = new THREE.Texture(a.image); texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = texture.magFilter = THREE.NearestFilter; texture.generateMipmaps = false; texture.needsUpdate = true;
  a.mesh.material.map = texture; a.mesh.material.needsUpdate = true;
  a.mesh.visible = $('show-' + axis).checked; requestRender();
}
async function queuePlane(axis) {
  const a = state.axes[axis]; if (!state.scene || state.scene.id !== state.current || a.inFlight) return;
  if (a.loadedScene === state.current && a.loadedIndex === a.index) { drawSlice(axis); if (a.mesh) a.mesh.visible = $('show-' + axis).checked; return; }
  const sid = state.current; const index = a.index; a.inFlight = true; $('busy-' + axis).hidden = false;
  try {
    const response = await fetch('/api/plane?' + new URLSearchParams({ scene: sid, axis, index }));
    if (!response.ok) { const value = await response.json(); throw Error(value.error || value.message); }
    if (Number(response.headers.get('X-Index')) !== index) throw Error('切片索引不一致');
    const blob = await response.blob(); const objectURL = URL.createObjectURL(blob); const image = new Image();
    try { image.src = objectURL; await image.decode(); } finally { URL.revokeObjectURL(objectURL); }
    if (image.width !== 1024 || image.height !== 1024) throw Error('切片尺寸不一致');
    if (state.current !== sid || a.index !== index) return;
    const scratch = document.createElement('canvas'); scratch.width = scratch.height = 1024;
    const context = scratch.getContext('2d', { willReadFrequently: true }); context.drawImage(image, 0, 0);
    a.image = image; a.pixels = context.getImageData(0, 0, 1024, 1024).data;
    a.loadedScene = sid; a.loadedIndex = index;
    a.counts = ['X-Lit', 'X-Shadow', 'X-Solid'].map(key => Number(response.headers.get(key)));
    updatePlaneMesh(axis); drawSlice(axis);
  } catch (error) { if (state.current === sid && a.index === index) toast('切片读取失败：' + error.message); }
  finally {
    a.inFlight = false;
    if (state.current === sid && a.index === index) $('busy-' + axis).hidden = true;
    else queuePlane(axis);
  }
}
function sliceRegion(axis) {
  if (state.scope !== 'window' || !state.window) return { u: 0, v: 0, size: 1024, outside: false };
  const [u, v] = UV[axis]; const index = state.axes[axis].index; const i = AXES.indexOf(axis); const w = state.window;
  return { u: w.origin[u], v: w.origin[v], size: w.side, outside: index < w.origin[i] || index >= w.origin[i] + w.side };
}
function drawSlice(axis) {
  const a = state.axes[axis]; const canvas = $('slice-' + axis); const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 1024, 1024); const region = sliceRegion(axis);
  $('empty-' + axis).hidden = !region.outside;
  if (region.outside) { $('slice-info-' + axis).textContent = '切面不穿过当前窗口'; return; }
  if (!a.image || a.loadedScene !== state.current || a.loadedIndex !== a.index) {
    $('slice-info-' + axis).textContent = '正在读取存储的响应与实体位…'; return;
  }
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(a.image, region.u, 1024 - region.v - region.size, region.size, region.size, 0, 0, 1024, 1024);
  const [ui, vi] = UV[axis]; const w = state.window; const ai = AXES.indexOf(axis);
  if (state.scope === 'global' && w && a.index >= w.origin[ai] && a.index < w.origin[ai] + w.side) {
    ctx.strokeStyle = '#ffab52'; ctx.lineWidth = 5;
    ctx.strokeRect(w.origin[ui], 1024 - w.origin[vi] - w.side, w.side, w.side);
  }
  const u = (state.axes[AXES[ui]].index - region.u + .5) / region.size * 1024;
  const v = 1024 - (state.axes[AXES[vi]].index - region.v + .5) / region.size * 1024;
  ctx.save(); ctx.strokeStyle = '#ffffff72'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 8]);
  ctx.beginPath(); ctx.moveTo(u, 0); ctx.lineTo(u, 1024); ctx.moveTo(0, v); ctx.lineTo(1024, v); ctx.stroke(); ctx.restore();
  const count = a.counts; $('slice-info-' + axis).textContent = state.scope === 'window' ?
    `${region.size} × ${region.size} · 原始体素，最近邻显示` :
    `受光 ${percent(count[0] / (count[0] + count[1]))}% · 1024 × 1024`;
}
function slicePointer(axis, event, pick = false) {
  const a = state.axes[axis]; const region = sliceRegion(axis);
  if (!a.pixels || region.outside || a.loadedScene !== state.current || a.loadedIndex !== a.index) return;
  const box = $('slice-' + axis).getBoundingClientRect();
  const u = region.u + Math.min(region.size - 1, Math.max(0, Math.floor((event.clientX - box.left) / box.width * region.size)));
  const v = region.v + region.size - 1 - Math.min(region.size - 1, Math.max(0, Math.floor((event.clientY - box.top) / box.height * region.size)));
  const [ui, vi] = UV[axis]; const point = AXES.map(key => state.axes[key].index); point[ui] = u; point[vi] = v;
  const offset = ((1023 - v) * 1024 + u) * 4;
  const label = a.pixels[offset] === 246 ? '受光 FREE' : a.pixels[offset] === 23 ? '阴影 FREE' : '实体障碍物';
  $('slice-info-' + axis).textContent = `XYZ (${point.join(', ')}) · ${label}`;
  if (pick) { setAxis(AXES[ui], u, true); setAxis(AXES[vi], v, true); AXES.forEach(drawSlice); }
}
function setScope(scope) {
  state.scope = scope; $('scope-global').classList.toggle('active', scope === 'global');
  $('scope-window').classList.toggle('active', scope === 'window'); AXES.forEach(drawSlice);
}

function bindEvents() {
  $('show-volume').onchange = () => {
    if (volumeMesh) { updateVolume(); $('volume-status').textContent = $('show-volume').checked ?
      `✓ ${$('volume-quality').value}³ 全局预览 · 原始场 1024³` : '体积已隐藏；精确切片仍可使用。'; }
    else loadVolume();
  };
  $('volume-quality').onchange = loadVolume;
  $('volume-mode').onchange = updateVolume; $('volume-crop').onchange = updateVolume;
  $('volume-density').oninput = () => {
    $('volume-density-label').value = (Number($('volume-density').value)/10).toFixed(1); updateVolume();
  };
  $('show-source').onchange = () => { if (sourceGroup) sourceGroup.visible = $('show-source').checked; requestRender(); };
  $('locate-source').onclick = () => {
    if (!renderer || !state.scene?.source_position) return;
    setFocus('source');
  };
  $('scene-search').addEventListener('input', () => renderLibrary());
  ['family-filter', 'split-filter'].forEach(id => $(id).addEventListener('change', () => renderLibrary()));
  $('load-more').onclick = () => { state.shown += 80; renderLibrary(false); };
  $('scene-list').onclick = event => { const button = event.target.closest('[data-scene]'); if (button) loadScene(button.dataset.scene); };
  $('library-toggle').onclick = () => document.body.classList.toggle('library-open');
  $('window-select').onchange = () => selectWindow(Number($('window-select').value));
  $('window-size').onchange = () => { fillWindowOptions(); const list = filteredWindows(); if (list.length && !list.some(w => w.id === state.window?.id)) selectWindow(list[0].id); else if (state.window) $('window-select').value = state.window.id; };
  $('previous-window').onclick = () => windowStep(-1); $('next-window').onclick = () => windowStep(1);
  $('global-view').onclick = () => setFocus('global'); $('focus-view').onclick = () => setFocus('window');
  ['xy', 'xz', 'yz'].forEach(view => $('view-' + view).onclick = () => setCamera(view));
  $('reset-view').onclick = () => setCamera('reset');
  $('scope-global').onclick = () => setScope('global'); $('scope-window').onclick = () => setScope('window');
  $('center-slices').onclick = centerSlices; $('follow-window').onchange = () => { if ($('follow-window').checked) centerSlices(); };
  AXES.forEach(axis => {
    $('range-' + axis).addEventListener('input', event => setAxis(axis, event.target.value));
    $('range-' + axis).addEventListener('change', () => queuePlane(axis));
    $('number-' + axis).addEventListener('input', event => { if (event.target.value !== '') setAxis(axis, event.target.value); });
    $('number-' + axis).addEventListener('change', event => setAxis(axis, event.target.value, true));
    $('show-' + axis).onchange = () => { const a = state.axes[axis]; if (a.mesh) a.mesh.visible = $('show-' + axis).checked && a.loadedScene === state.current && a.loadedIndex === a.index; requestRender(); };
    $('slice-' + axis).addEventListener('pointermove', event => slicePointer(axis, event));
    $('slice-' + axis).addEventListener('pointerleave', () => drawSlice(axis));
    $('slice-' + axis).addEventListener('click', event => slicePointer(axis, event, true));
  });
  document.querySelectorAll('[data-open]').forEach(button => button.onclick = () => {
    if (!state.scene) return; const axis = button.dataset.open;
    window.open('/api/plane?' + new URLSearchParams({ scene: state.current, axis, index: state.axes[axis].index }), '_blank', 'noopener');
  });
  $('obstacle-opacity').oninput = () => {
    const alpha = Number($('obstacle-opacity').value) / 100; $('obstacle-opacity-label').value = Math.round(alpha * 100) + '%';
    obstacles?.children.filter(o => !o.userData.outline).forEach(o => { o.material.opacity = alpha; o.material.depthWrite = alpha >= .99; }); requestRender();
  };
  $('plane-opacity').oninput = () => {
    const alpha = Number($('plane-opacity').value) / 100; $('plane-opacity-label').value = Math.round(alpha * 100) + '%';
    AXES.forEach(axis => { if (state.axes[axis].mesh) state.axes[axis].mesh.material.opacity = alpha; }); requestRender();
  };
  $('show-edges').onchange = () => { obstacles?.children.filter(o => o.userData.outline).forEach(o => o.visible = $('show-edges').checked); requestRender(); };
  $('all-windows').onchange = () => { if (allBoxes) allBoxes.visible = $('all-windows').checked; requestRender(); };
  $('clip-window').onchange = applyClipping;
  $('auto-rotate').onchange = () => { if (controls) controls.autoRotate = $('auto-rotate').checked; requestRender(); };
  $('fullscreen').onclick = async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.querySelector('.viewport-panel').requestFullscreen(); }
    catch { toast('当前浏览器不允许全屏，可使用浏览器窗口最大化。'); }
  };
  $('screenshot').onclick = () => {
    if (!renderer || !state.scene) return toast('三维渲染尚未就绪');
    renderer.render(world, camera); renderer.domElement.toBlob(blob => {
      const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url;
      anchor.download = `RIND3D-${state.current.slice(0, 12)}-window-${state.window.id}.png`; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 3000);
    });
  };
  $('help-button').onclick = () => $('help-dialog').showModal(); $('close-help').onclick = () => $('help-dialog').close();
  $('help-dialog').addEventListener('click', event => { if (event.target === $('help-dialog')) { const b = event.target.getBoundingClientRect(); if (event.clientX < b.left || event.clientX > b.right || event.clientY < b.top || event.clientY > b.bottom) $('help-dialog').close(); } });
  document.addEventListener('keydown', event => {
    if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName) || $('help-dialog').open || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); windowStep(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); windowStep(1); }
    if (event.key.toLowerCase() === 'g') setFocus('global');
    if (event.key.toLowerCase() === 'f') setFocus('window');
    if (event.key.toLowerCase() === 'r') setCamera('reset');
    if (event.key === '?') $('help-dialog').showModal();
  });
}

async function boot() {
  bindEvents(); initThree();
  try {
    state.catalog = await json('/api/catalog');
    $('scene-total').textContent = number(state.catalog.scene_count); $('window-total').textContent = number(state.catalog.window_count);
    [...new Set(state.catalog.scenes.map(s => s.family))].sort().forEach(family => {
      const option = document.createElement('option'); option.value = family; option.textContent = (FAMILY[family] || [family])[0]; $('family-filter').append(option);
    });
    renderLibrary(); const params = new URLSearchParams(location.search); const queryScene = params.get('scene');
    const chosen = state.catalog.scenes.find(s => s.id === queryScene) || state.catalog.scenes.find(s => s.id.startsWith('cd9f325f1b')) || state.catalog.scenes[0];
    if (!chosen) throw Error('数据集中没有可浏览的 READY 场景');
    await loadScene(chosen.id, params.has('window') ? Number(params.get('window')) : chosen.id.startsWith('cd9f325f1b') ? 198 : 0);
    AXES.forEach(axis => { if (params.has(axis)) setAxis(axis, Number(params.get(axis)), true); });
  } catch (error) { $('viewport-loading').hidden = true; toast('数据集连接失败：' + error.message); console.error(error); }
}
boot();
