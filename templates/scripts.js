// static/scripts.js
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultDiv = document.getElementById('result');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.border = '2px dashed #4CAF50';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.border = '2px dashed #ccc';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.border = '2px dashed #ccc';
    const file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;
    resultDiv.innerHTML = `File selected: ${file.name}`;
});

fileInput.addEventListener('change', () => {
    resultDiv.innerHTML = `File selected: ${fileInput.files[0].name}`;
});
