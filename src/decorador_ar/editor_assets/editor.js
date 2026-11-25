const $ = id => document.getElementById(id);
const stage = $('stage'), draft = $('draft'), canvas = $('outline');
const room = $('room'), surface = $('surface'), piece = $('piece'), support = $('support');
const status = $('status');
const controls = document.querySelector('aside');
let photo = $('photo');
let config, state, point, corners = null, busy = false, queued = false, sending = false;
let revision = 0;

async function request(action, data) {
  const response = await fetch(action, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)});
  if (!response.ok) throw new Error((await response.json()).error);
  return action === 'preview' ? response.blob() : response.json();
}

function settings() {
  return {room: room.value, surface: surface.value, piece: piece.value,
    support: Number(support.value), point, scale: Number($('scale').value),
    angle: Number($('angle').value), vertical: $('vertical').checked, lighting: $('lighting').checked};
}

function options(select, entries, previous = select.value) {
  select.replaceChildren(...entries.map(([label, value]) => new Option(label, value)));
  if (entries.some(([, value]) => String(value) === previous)) select.value = previous;
}

function supportInfo() { return surface.value === 'raised' ? state?.supports[support.value] : null; }
function firstPoint() {
  return surface.value === 'raised' ? supportInfo()?.center ?? null : state?.anchor;
}

function instructions() {
  if (corners) return `Marca las cuatro esquinas del tablero · ${corners.length}/4`;
  if (surface.value === 'raised' && !supportInfo()) return 'Detecta un apoyo o coloca primero un mueble con tablero.';
  return 'Mueve la pieza con el cursor. Haz clic para colocarla.';
}

function hideDraft() {
  revision += 1;
  queued = false;
  draft.hidden = true;
}

function drawOutline() {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const outline = corners ?? supportInfo()?.outline;
  if (!outline?.length) return;
  ctx.strokeStyle = ctx.fillStyle = '#42e1ff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  outline.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  if (!corners) ctx.closePath();
  ctx.stroke();
  if (corners) for (const [x, y] of corners) {
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
  }
}

async function loadComposition() {
  const path = `photo?room=${encodeURIComponent(room.value)}&lit=${+$('lighting').checked}`;
  // The path stays the same after adding a piece, so use the bytes of a fresh response.
  const response = await fetch(path, {cache: 'no-store'});
  if (!response.ok) throw new Error('No se pudo actualizar la foto.');
  const url = URL.createObjectURL(await response.blob());
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
  image.id = photo.id;
  image.alt = photo.alt;
  image.draggable = false;
  photo.replaceWith(image);
  if (photo.src.startsWith('blob:')) URL.revokeObjectURL(photo.src);
  photo = image;
}

async function refresh(next, {reset = false, reload = false} = {}) {
  if (reload) await loadComposition();
  state = next;
  [canvas.width, canvas.height] = state.size;
  stage.style.aspectRatio = state.size.join(' / ');
  const active = surface.value === 'raised';
  $('raised').hidden = $('manual').hidden = !active;
  $('vertical-row').hidden = surface.value !== 'floor' || config.models.includes(piece.value);
  $('detect').disabled = state.scanned;
  options(support, state.supports.length
    ? state.supports.map(({label}, i) => [label, i]) : [['Sin apoyo', -1]]);
  options($('items'), state.items.length ? state.items.map((label, i) => [label, i]) : [['Ninguna', -1]]);
  $('remove').disabled = $('save').disabled = !state.items.length;
  if (reset) point = firstPoint();
  drawOutline();
}

async function renderDraft() {
  if (sending || busy || !queued || !point || corners) return;
  sending = true;
  queued = false;
  const id = revision;
  let url;
  try {
    const image = await request('preview', settings());
    if (id !== revision) return;
    url = URL.createObjectURL(image);
    const loaded = new Image();
    loaded.src = url;
    await loaded.decode();
    if (id !== revision) return;
    if (draft.src.startsWith('blob:')) URL.revokeObjectURL(draft.src);
    draft.src = url;
    url = undefined;
    draft.hidden = false;
    status.textContent = instructions();
  } catch (error) {
    if (id === revision) {
      draft.hidden = true;
      status.textContent = error.message;
    }
  } finally {
    if (url) URL.revokeObjectURL(url);
    sending = false;
    if (queued) requestAnimationFrame(renderDraft);
  }
}

function preview() {
  if (!point || busy || corners) return;
  if (surface.value === 'raised' && !supportInfo()) return;
  queued = true;
  if (!sending) requestAnimationFrame(renderDraft);
}

async function change(action = 'state', extra = {}, {reset = false, reload = false} = {}) {
  if (busy) return;
  busy = true;
  controls.inert = true;
  hideDraft();
  try {
    await refresh(await request(action, {...settings(), ...extra}), {reset, reload});
    status.textContent = instructions();
  } catch (error) {
    status.textContent = error.message;
    drawOutline();
    return false;
  } finally {
    busy = false;
    controls.inert = false;
  }
  return true;
}

function coords(event) {
  const rect = stage.getBoundingClientRect();
  return [(event.clientX - rect.left) * state.size[0] / rect.width,
    (event.clientY - rect.top) * state.size[1] / rect.height];
}

async function commit(target) {
  if (busy || !target) return;
  point = target;
  if (corners) {
    corners.push(point);
    drawOutline();
    if (corners.length < 4) { status.textContent = instructions(); return; }
    const marked = corners;
    corners = null;
    $('mark').textContent = 'Marcar tablero';
    if (await change('mark', {corners: marked, height: Number($('height').value)})) {
      support.selectedIndex = support.options.length - 1;
      point = firstPoint();
      drawOutline();
      preview();
    }
    return;
  }
  if (await change('add', {}, {reload: true})) status.textContent = 'Pieza colocada.';
}

stage.addEventListener('pointermove', event => {
  if (state && !busy) { point = coords(event); preview(); }
});
stage.addEventListener('pointerleave', () => {
  hideDraft();
  if (state && !busy) status.textContent = instructions();
});
stage.addEventListener('click', event => {
  if (state) void commit(coords(event));
});
stage.addEventListener('keydown', event => {
  if (!state || busy) return;
  const moves = {ArrowUp: [0, -8], ArrowDown: [0, 8], ArrowLeft: [-8, 0], ArrowRight: [8, 0]};
  if (event.key === 'Enter') { event.preventDefault(); void commit(point ?? firstPoint()); }
  if (event.key in moves && (point ?? firstPoint())) {
    event.preventDefault();
    point = (point ?? firstPoint()).map((n, axis) =>
      Math.max(0, Math.min(state.size[axis] - 1, n + moves[event.key][axis])));
    preview();
  }
});

function selectPieces() { options(piece, config.choices[surface.value]); }
room.addEventListener('change', async () => {
  corners = null;
  $('mark').textContent = 'Marcar tablero';
  support.value = '-1';
  if (await change('state', {}, {reset: true, reload: true})) preview();
});
surface.addEventListener('change', async () => {
  corners = null;
  $('mark').textContent = 'Marcar tablero';
  selectPieces();
  if (await change('state', {}, {reset: true})) preview();
});
piece.addEventListener('change', async () => { if (await change()) preview(); });
support.addEventListener('change', () => { hideDraft(); point = firstPoint(); drawOutline(); preview(); });
for (const id of ['scale', 'angle', 'vertical']) $(id).addEventListener('input', () => { hideDraft(); preview(); });
$('lighting').addEventListener('change', async () => { if (await change('state', {}, {reload: true})) preview(); });
$('detect').addEventListener('click', async () => {
  status.textContent = 'Buscando apoyos en la foto…';
  if (await change('detect', {}, {reset: true})) preview();
});
$('mark').addEventListener('click', () => {
  hideDraft();
  corners = corners ? null : [];
  $('mark').textContent = corners ? 'Cancelar marcado' : 'Marcar tablero';
  status.textContent = instructions(); drawOutline();
});
$('remove').addEventListener('click', () => change('remove', {index: Number($('items').value)}, {reload: true}));
$('save').addEventListener('click', async () => {
  if (busy) return;
  try { status.textContent = 'Guardado en ' + (await request('save', settings())).path; }
  catch (error) { status.textContent = error.message; }
});

try {
  config = await (await fetch('config')).json();
  options(room, config.rooms.map(name => [name.replaceAll('_', ' '), name]), config.initial);
  selectPieces();
  if (await change('state', {}, {reset: true, reload: true})) preview();
} catch (error) {
  status.textContent = error.message;
}
