// static/js/dashboard.js - Simplified chart and UI management

let lossChart = null;
let accChart = null;
let trainingInterval = null;
let chartsReady = false;
let lastEpochCount = 0;

// Wait for DOM and Chart.js
document.addEventListener('DOMContentLoaded', () => {
  console.log('✓ DOM Content Loaded');
  
  // Wait for Chart.js library
  let chartAttempts = 0;
  const checkChart = () => {
    if (typeof Chart !== 'undefined') {
      console.log('✓ Chart.js library loaded');
      initCharts();
      setupEventListeners();
      // Refresh recent history on initial load
      try { setTimeout(() => refreshHistory(), 200); } catch (e) { }
    } else if (chartAttempts < 100) {
      chartAttempts++;
      setTimeout(checkChart, 50);
    } else {
      console.error('✗ Chart.js failed to load after timeout');
    }
  };
  
  checkChart();
});

// Initialize empty charts
function initCharts() {
  console.log('Initializing charts...');
  
  const lossCtx = document.getElementById('lossChart');
  const accCtx = document.getElementById('accChart');
  
  if (!lossCtx) {
    console.error('✗ Loss chart canvas not found');
    return;
  }
  if (!accCtx) {
    console.error('✗ Accuracy chart canvas not found');
    return;
  }
  
  // Initialize loss chart
  try {
    lossChart = new Chart(lossCtx.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Training Loss',
          data: [],
          borderColor: '#1f77b4',
          backgroundColor: 'rgba(31,119,180,0.0)',
          borderWidth: 3,
          tension: 0.2,
          fill: false,
          pointRadius: 6,
          pointBackgroundColor: '#1f77b4',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointStyle: 'circle'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { 
            display: true,
            labels: { color: '#fff', font: { size: 12 }, boxWidth: 12 }
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: '#ff6b6b',
            borderWidth: 1
          }
        },
        scales: {
          y: { 
            beginAtZero: true,
            ticks: { color: '#aaa' },
            grid: { color: 'rgba(255,255,255,0.06)', lineWidth: 1 }
          },
          x: {
            ticks: { color: '#aaa' },
            grid: { color: 'rgba(255,255,255,0.06)', lineWidth: 1 }
          }
        }
      }
    });
    console.log('✓ Loss chart initialized');
  } catch (e) {
    console.error('✗ Error creating loss chart:', e);
    return;
  }
  
  // Initialize accuracy chart
  try {
    accChart = new Chart(accCtx.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Training Accuracy (%)',
          data: [],
          borderColor: '#2ca02c',
          backgroundColor: 'rgba(44,160,44,0.0)',
          borderWidth: 3,
          tension: 0.2,
          fill: false,
          pointRadius: 6,
          pointBackgroundColor: '#2ca02c',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointStyle: 'circle'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { 
            display: true,
            labels: { color: '#fff', font: { size: 12 }, boxWidth: 12 }
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: '#6c5ce7',
            borderWidth: 1
          }
        },
        scales: {
          y: { 
            beginAtZero: true,
            max: 100,
            ticks: { color: '#aaa' },
            grid: { color: 'rgba(255,255,255,0.06)', lineWidth: 1 }
          },
          x: {
            ticks: { color: '#aaa' },
            grid: { color: 'rgba(255,255,255,0.06)', lineWidth: 1 }
          }
        }
      }
    });
    console.log('✓ Accuracy chart initialized');
  } catch (e) {
    console.error('✗ Error creating accuracy chart:', e);
    return;
  }
  
  chartsReady = true;
  loadChartData(); // Load initial data
}

// Live chart polling disabled (update charts only when training finishes)

// Load and display chart data
async function loadChartData() {
  console.log('Loading chart data...');
  
  try {
    const response = await fetch(`/chart_data?t=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) {
      console.error('✗ Failed to fetch chart data:', response.status);
      return;
    }
    
    const data = await response.json();
    console.log('✓ Chart data received:', {
      epochs: data.loss.length,
      losses: data.loss,
      accuracies: data.acc
    });
    
    if (!data.loss || data.loss.length === 0) {
      console.log('⚠ No training data available yet');
      return;
    }
    
    // Update charts with data
    updateChartsWithData(data);
  } catch (error) {
    console.error('✗ Error loading chart data:', error);
  }
}

// Update charts with new data
function updateChartsWithData(data) {
  if (!chartsReady || !lossChart || !accChart) {
    console.warn('⚠ Charts not ready yet');
    return;
  }
  
  console.log('Updating charts with data...');
  
  try {
    // Generate labels
    const labels = data.loss.map((_, i) => `Epoch ${i + 1}`);
    
    // Update loss chart
    lossChart.data.labels = labels;
    // Ensure numeric values
    lossChart.data.datasets[0].data = data.loss.map(v => Number(v));
    // Apply axis metadata from backend or exact matplotlib chart_meta if available
    try {
      if (data.chart_meta && data.chart_meta.loss_ticks) {
        const ticks = data.chart_meta.loss_ticks.map(v => Number(v));
        lossChart.options.scales.y.min = Number(data.chart_meta.loss_ylim[0]);
        lossChart.options.scales.y.max = Number(data.chart_meta.loss_ylim[1]);
        // Use afterBuildTicks to force exact tick positions
        lossChart.options.scales.y.afterBuildTicks = (scale) => {
          scale.ticks = ticks.map(v => ({ value: v, label: String(v) }));
        };
      } else if (data.loss_meta) {
        lossChart.options.scales.y.min = Number(data.loss_meta.min);
        lossChart.options.scales.y.max = Number(data.loss_meta.max);
        lossChart.options.scales.y.ticks = lossChart.options.scales.y.ticks || {};
        lossChart.options.scales.y.ticks.stepSize = Number(data.loss_meta.step);
      } else {
        // fallback: autoscale with small margin
        const vals = lossChart.data.datasets[0].data.filter(v => !isNaN(v));
        const maxLoss = vals.length ? Math.max(...vals) : 0.0;
        const step = Math.max(0.001, (maxLoss || 0.01) / 5.0);
        lossChart.options.scales.y.min = 0;
        lossChart.options.scales.y.max = maxLoss * 1.1;
        lossChart.options.scales.y.ticks = lossChart.options.scales.y.ticks || {};
        lossChart.options.scales.y.ticks.stepSize = step;
      }
    } catch (e) {}
    lossChart.update();
    console.log('✓ Loss chart updated');
    
    // Update accuracy chart
    accChart.data.labels = labels;
    accChart.data.datasets[0].data = data.acc.map(v => Number(v));
    // Apply accuracy axis metadata from backend or matplotlib meta if available
    try {
      if (data.chart_meta && data.chart_meta.acc_ticks) {
        const ticks = data.chart_meta.acc_ticks.map(v => Number(v));
        accChart.options.scales.y.min = Number(data.chart_meta.acc_ylim[0]);
        accChart.options.scales.y.max = Number(data.chart_meta.acc_ylim[1]);
        accChart.options.scales.y.afterBuildTicks = (scale) => {
          scale.ticks = ticks.map(v => ({ value: v, label: String(v) }));
        };
      } else if (data.acc_meta) {
        accChart.options.scales.y.min = Number(data.acc_meta.min);
        accChart.options.scales.y.max = Number(data.acc_meta.max);
        accChart.options.scales.y.ticks = accChart.options.scales.y.ticks || {};
        accChart.options.scales.y.ticks.stepSize = Number(data.acc_meta.step);
      } else {
        accChart.options.scales.y.min = 0;
        accChart.options.scales.y.max = 100;
        accChart.options.scales.y.ticks = accChart.options.scales.y.ticks || {};
        accChart.options.scales.y.ticks.stepSize = 10;
      }
    } catch (e) {}
    accChart.update();
    console.log('✓ Accuracy chart updated');
    // Reload the saved PNG when epoch count changes
    try {
      reloadTrainingCurvesImage();
    } catch (e) {
      console.warn('Could not reload training curves image:', e);
    }
  } catch (error) {
    console.error('✗ Error updating charts:', error);
  }
}

// Refresh recent predictions list (no full reload)
async function refreshHistory() {
  const container = document.getElementById('recentPredictionsList');
  if (!container) return;
  try {
    const resp = await fetch('/history_data');
    if (!resp.ok) return;
    const payload = await resp.json();
    const history = payload.history || [];

    if (!history.length) {
      container.innerHTML = `<div class="text-center py-3"><i class="bi bi-inbox text-muted" style="font-size: 2rem;"></i><p class="text-muted small mt-2 mb-0">No predictions yet.</p><p class="text-muted small">Upload an image to get started!</p></div>`;
      return;
    }

    let html = '';
    for (let i = 0; i < Math.min(5, history.length); i++) {
      const h = history[i];
      const timePart = (h.timestamp || '').split(' ')[1] || '';
      html += `
        <div class="list-group-item bg-dark text-light d-flex align-items-start p-3">
          <img src="/uploads/${h.filename}?t=${Date.now()}" width="50" height="50" class="rounded me-3 object-fit-cover border border-secondary" alt="thumb">
          <div class="flex-grow-1">
            <div class="small text-muted">${timePart}</div>
            <div class="fw-bold text-success">${h.prediction}</div>
            <div class="small">${h.confidence}</div>
          </div>
        </div>`;
    }

    

    container.innerHTML = html;

    // wire up clear history button
    const clearBtn = document.getElementById('clearHistoryBtn');
    if (clearBtn) {
      clearBtn.addEventListener('click', async () => {
        if (!confirm('Clear history?')) return;
        try {
          const r = await fetch('/clear_history', { method: 'POST' });
          const j = await r.json();
          if (j.ok) refreshHistory();
        } catch (e) { console.warn('Clear history failed', e); }
      });
    }
  } catch (e) {
    console.warn('Could not refresh history:', e);
  }
}

// Reload the exact saved training_curves.png with cache-busting
function reloadTrainingCurvesImage(force=false) {
  const img = document.getElementById('trainingCurvesImage');
  if (!img) return;

  const epochCount = (lossChart && lossChart.data && lossChart.data.datasets && lossChart.data.datasets[0].data)
    ? lossChart.data.datasets[0].data.length
    : 0;

  if (!force && epochCount === lastEpochCount) return; // only reload when changed
  lastEpochCount = epochCount;

  const url = `/results/training_curves.png?t=${Date.now()}`;
  img.src = url;
  console.log('Reloaded training_curves.png (cache-busted)');
}

// Reload predictions grid image (cache-busted)
function reloadPredictionsGridImage() {
  const gi = document.getElementById('gridImage');
  if (!gi) return;
  gi.src = `/results/predictions_grid.png?t=${Date.now()}`;
  console.log('Reloaded predictions_grid.png (cache-busted)');
}

// Setup all event listeners
function setupEventListeners() {
  console.log('Setting up event listeners...');
  
  // Sidebar toggle
  const sidebarToggle = document.getElementById('sidebarToggle');
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
      document.getElementById('sidebar-wrapper')?.classList.toggle('collapsed');
    });
  }
  
  // Train Dataset button
  const trainDatasetBtn = document.getElementById('trainDatasetBtn');
  if (trainDatasetBtn) {
    trainDatasetBtn.addEventListener('click', () => {
      if (confirm('Start training on CIFAR-10 dataset (5 epochs)?')) {
        startTraining('dataset');
      }
    });
  }
  
  // Train Custom button
  const trainCustomBtn = document.getElementById('trainCustomBtn');
  if (trainCustomBtn) {
    trainCustomBtn.addEventListener('click', () => {
      if (confirm('Start training on custom uploaded images (3 epochs)?')) {
        startTraining('custom');
      }
    });
  }
  
  // Train button (top navbar)
  const trainBtn = document.getElementById('trainBtn');
  if (trainBtn) {
    trainBtn.addEventListener('click', () => {
      if (confirm('Start training now?')) {
        startTraining('dataset');
      }
    });
  }
  
  // Refresh Charts button
  const refreshChartsBtn = document.getElementById('refreshCharts');
  if (refreshChartsBtn) {
    refreshChartsBtn.addEventListener('click', (e) => {
      e.preventDefault();
      console.log('Manual chart refresh...');
      loadChartData();
    });
  }
  
  // Upload button
  const uploadBtn = document.getElementById('uploadBtn');
  if (uploadBtn) {
    uploadBtn.addEventListener('click', handleUpload);
  }
  
  // Clear uploads button
  const clearUploadsBtn = document.getElementById('clearUploadsBtn');
  if (clearUploadsBtn) {
    clearUploadsBtn.addEventListener('click', () => {
      if (confirm('Clear all uploaded images?')) {
        clearUploads();
      }
    });
  }
}

// Handle image upload and prediction
async function handleUpload() {
  const fileInput = document.getElementById('fileInput');
  const uploadAlert = document.getElementById('uploadAlert');
  const uploadProgress = document.getElementById('uploadProgress');
  const predictionResult = document.getElementById('predictionResult');
  const resultContent = document.getElementById('resultContent');
  
  if (!fileInput || !fileInput.files.length) {
    showAlert('Please select an image', 'warning', uploadAlert);
    return;
  }
  
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  
  if (uploadProgress) uploadProgress.style.display = 'block';
  showAlert('Uploading and predicting...', 'info', uploadAlert);
  
  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (uploadProgress) uploadProgress.style.display = 'none';
    
    if (data.ok) {
      const rec = data.record;
      const timestamp = new Date().getTime();
      let html = `
        <div class="row">
          <div class="col-md-6 text-center">
            <img src="/uploads/${rec.filename}?t=${timestamp}" 
                 class="img-fluid rounded border border-success" 
                 style="max-height: 300px; object-fit: contain;"
                 alt="Uploaded">
          </div>
          <div class="col-md-6">
            <h5 class="text-success">Prediction: ${rec.prediction}</h5>
            <p class="text-muted">Confidence: ${rec.confidence}</p>
      `;
      
      if (data.top3) {
        html += '<h6 class="text-light mt-3">Top Predictions:</h6><ul>';
        data.top3.forEach(pred => {
          html += `<li>${pred.class}: ${pred.prob}%</li>`;
        });
        html += '</ul>';
      }
      
      html += '</div></div>';
      if (resultContent) resultContent.innerHTML = html;
      if (predictionResult) predictionResult.style.display = 'block';
      showAlert('Prediction successful!', 'success', uploadAlert);
      // Refresh predictions grid and recent history to show updated thumbnails
      try { reloadPredictionsGridImage(); } catch (e) { console.warn('Could not reload predictions grid after upload:', e); }
      // Refresh recent predictions list in-place
      try { setTimeout(() => refreshHistory(), 300); } catch (e) { console.warn('Could not refresh history after upload:', e); }
    } else {
      showAlert(data.error || 'Prediction failed', 'danger', uploadAlert);
    }
  } catch (error) {
    console.error('Upload error:', error);
    showAlert('Upload failed: ' + error.message, 'danger', uploadAlert);
    if (uploadProgress) uploadProgress.style.display = 'none';
  }
}

// Start training
async function startTraining(mode) {
  console.log('Starting training with mode:', mode);
  
  try {
    const res = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: mode })
    });
    
    const data = await res.json();
    
    if (data.ok) {
      console.log('Training started, beginning to monitor...');
      monitorTraining();
    } else {
      alert('Error: ' + (data.error || data.msg || 'Unknown error'));
    }
  } catch (error) {
    console.error('Training start error:', error);
    alert('Failed to start training: ' + error.message);
  }
}

// Load and display training summary
async function loadTrainingSummary() {
  console.log('Loading training summary...');
  
  try {
    const response = await fetch('/summary');
    if (!response.ok) {
      console.error('✗ Failed to fetch summary:', response.status);
      return;
    }
    
    const data = await response.json();
    const summaryContainer = document.querySelector('.card-body pre');
    
    if (summaryContainer && data.summary) {
      summaryContainer.textContent = data.summary;
      console.log('✓ Summary updated');
    }
  } catch (error) {
    console.error('✗ Error loading summary:', error);
  }
}

// Monitor training progress
function monitorTraining() {
  console.log('Starting training monitor...');
  
  const trainProgress = document.getElementById('trainProgress');
  const trainMessage = document.getElementById('trainMessage');
  const trainingStatus = document.getElementById('trainingStatus');
  const trainingProgress = document.getElementById('trainingProgress');
  
  if (trainingStatus) trainingStatus.style.display = 'block';
  if (trainingProgress) trainingProgress.style.display = 'block';
  
  trainingInterval = setInterval(async () => {
    try {
      const res = await fetch('/status');
      const status = await res.json();
      
      console.log('Training status:', status);
      
      // Update progress bars
      if (trainProgress) {
        trainProgress.style.width = status.progress + '%';
        trainProgress.textContent = status.progress + '%';
      }
      
      if (trainingProgress) {
        const bar = trainingProgress.querySelector('.progress-bar');
        if (bar) {
          bar.style.width = status.progress + '%';
          bar.textContent = status.progress + '%';
        }
      }
      
      // Update messages
      if (trainMessage) {
        trainMessage.textContent = status.message || 'Training...';
      }
      
      if (trainingStatus) {
        trainingStatus.textContent = status.message || 'Training in progress...';
      }
      
      // Keep summary updated during training
      loadTrainingSummary();
      
      // Stop when training completes
      if (!status.running || status.progress >= 100) {
        console.log('Training complete');
        clearInterval(trainingInterval);
        trainingInterval = null;
        
        if (trainingStatus) trainingStatus.style.display = 'none';
        if (trainingProgress) trainingProgress.style.display = 'none';
        
        // Final chart update
        loadChartData();
        
        // Final summary update
        loadTrainingSummary();
        
        // Refresh saved images and recent history in-place when training finishes
        try { reloadTrainingCurvesImage(true); } catch (e) { console.warn('Could not reload training curves image on completion:', e); }
        try { reloadPredictionsGridImage(); } catch (e) { console.warn('Could not reload predictions grid image on completion:', e); }
        try { setTimeout(() => refreshHistory(), 500); } catch (e) { console.warn('Could not refresh history on completion:', e); }
      }
    } catch (error) {
      console.error('Status check error:', error);
    }
  }, 1000); // Poll every 1 second
}

// Clear uploads
async function clearUploads() {
  try {
    const res = await fetch('/clear_uploads', { method: 'POST' });
    const data = await res.json();
    
    if (data.ok) {
      console.log('Uploads cleared');
      location.reload();
    } else {
      alert('Error: ' + (data.error || 'Unknown error'));
    }
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

// Show alert message
function showAlert(message, type, container) {
  if (!container) return;
  
  const alertHTML = `
    <div class="alert alert-${type} alert-dismissible fade show" role="alert">
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;
  
  container.innerHTML = alertHTML;
}
