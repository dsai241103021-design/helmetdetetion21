const apiUrl = window.HELMET_API_URL || '/api/predict';
let selectedModel = 'yolo';
let selectedFile = null;

const imageInput = document.getElementById('imageInput');
const cameraInput = document.getElementById('cameraInput');
const dropZone = document.getElementById('dropZone');
const fileName = document.getElementById('fileName');
const analyzeButton = document.getElementById('analyzeButton');
const apiState = document.getElementById('apiState');
const apiStateText = document.getElementById('apiStateText');
const errorMessage = document.getElementById('errorMessage');
const emptyOutput = document.getElementById('emptyOutput');
const resultOutput = document.getElementById('resultOutput');

document.querySelectorAll('.model-button').forEach((button) => {
  button.addEventListener('click', () => {
    selectedModel = button.dataset.model;
    document.querySelectorAll('.model-button').forEach((item) => item.classList.toggle('active', item === button));
    document.getElementById('selectedModelLabel').textContent = selectedModel.toUpperCase();
  });
});

function chooseFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;
  fileName.textContent = file.name;
  analyzeButton.disabled = false;
  errorMessage.textContent = '';
  dropZone.classList.add('has-file');
}

imageInput.addEventListener('change', (event) => chooseFile(event.target.files[0]));
cameraInput.addEventListener('change', (event) => chooseFile(event.target.files[0]));
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));

function setBusy(isBusy) {
  analyzeButton.disabled = isBusy || !selectedFile;
  analyzeButton.classList.toggle('loading', isBusy);
  analyzeButton.querySelector('span').textContent = isBusy ? 'ANALYZING...' : 'RUN DETECTION';
  apiState.classList.toggle('busy', isBusy);
  apiStateText.textContent = isBusy ? 'PROCESSING' : 'API READY';
}

function renderDetections(detections) {
  const list = document.getElementById('detectionList');
  list.innerHTML = '';
  (detections || []).forEach((detection) => {
    const row = document.createElement('div');
    row.className = 'detection-row';
    const score = Number(detection.confidence ?? detection.score ?? 0);
    row.innerHTML = `<span>${detection.label || detection.class || 'Object'}</span><b>${(score <= 1 ? score * 100 : score).toFixed(1)}%</b>`;
    list.appendChild(row);
  });
}

function renderResult(result) {
  const confidence = Number(result.confidence || result.score || 0);
  const percentage = confidence <= 1 ? confidence * 100 : confidence;
  document.getElementById('resultImage').src = result.annotated_image || result.image || result.annotated || '';
  document.getElementById('resultStatus').textContent = result.status || result.label || 'Object detected';
  document.getElementById('resultConfidence').textContent = `${percentage.toFixed(1)}%`;
  document.getElementById('resultDetail').textContent = result.detail || 'Detection completed successfully.';
  renderDetections(result.detections || result.boxes || []);
  emptyOutput.classList.add('hidden');
  resultOutput.classList.remove('hidden');
}

analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  setBusy(true);
  errorMessage.textContent = '';
  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('model', selectedModel);
  try {
    const response = await fetch(apiUrl, { method: 'POST', body: formData });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || result.error || `Request failed (${response.status})`);
    renderResult(result);
  } catch (error) {
    errorMessage.textContent = `Could not reach the inference service. ${error.message}`;
    apiState.classList.add('error');
    apiStateText.textContent = 'API OFFLINE';
  } finally {
    setBusy(false);
  }
});
