// Global Variables to store loaded model data and UI states
let modelData = null;
let currentValues = {};
let distributionChartInstance = null;
let rocChartInstance = null;

// DOM Elements
const sidebarNav = document.querySelector('.nav-menu');
const navItems = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');
const pageTitle = document.getElementById('page-title');
const pageSubtitle = document.getElementById('page-subtitle');
const slidersContainer = document.getElementById('sliders-container');
const resetSlidersBtn = document.getElementById('reset-sliders-btn');
const featureSelector = document.getElementById('feature-selector');
const getStartedBtn = document.getElementById('get-started-btn');
const welcomeScreen = document.getElementById('welcome-screen');
const appContainer = document.querySelector('.app-container');

// Get Started Button Action
if (getStartedBtn) {
    getStartedBtn.addEventListener('click', () => {
        if (welcomeScreen) {
            welcomeScreen.classList.add('fade-out');
            setTimeout(() => {
                welcomeScreen.remove();
            }, 500);
        }
        if (appContainer) {
            appContainer.classList.remove('hidden');
        }
    });
}

// Navigation Tab Routing
sidebarNav.addEventListener('click', (e) => {
    const btn = e.target.closest('.nav-item');
    if (!btn) return;
    
    const tabName = btn.dataset.tab;
    
    // Toggle active classes on sidebar
    navItems.forEach(item => item.classList.remove('active'));
    btn.classList.add('active');
    
    // Toggle active tab sections
    tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        }
    });

    // Update Header Text dynamically
    updateHeaderText(tabName);
    
    // Perform layout recalculations for canvas charts (fix scaling bugs when drawing in hidden tabs)
    if (tabName === 'analytics') {
        setTimeout(() => {
            renderDistributionChart(featureSelector.value);
        }, 100);
    } else if (tabName === 'performance') {
        setTimeout(() => {
            renderRocChart();
        }, 100);
    }
});

function updateHeaderText(tabName) {
    switch (tabName) {
        case 'predictor':
            pageTitle.innerText = "Breast Cancer Risk Predictor";
            pageSubtitle.innerText = "Interactive ML Diagnostic Assistant using the Wisconsin Breast Cancer Dataset";
            break;
        case 'analytics':
            pageTitle.innerText = "Exploratory Data Analysis";
            pageSubtitle.innerText = "Investigating feature distributions and relationships within the cell nuclei dataset";
            break;
        case 'performance':
            pageTitle.innerText = "Model Evaluation & Performance";
            pageSubtitle.innerText = "Comparing predictive performance of Support Vector Machine (SVM) vs Logistic Regression models";
            break;
        case 'dataset-info':
            pageTitle.innerText = "Dataset Information & Structure";
            pageSubtitle.innerText = "Scientific details, clinical backgrounds, and mathematical modeling techniques";
            break;
    }
}

// 1. Fetch data on initialization
async function initDashboard() {
    try {
        const response = await fetch('model_data.json');
        if (!response.ok) {
            throw new Error("Unable to load model_data.json. Make sure the training script has run.");
        }
        modelData = await response.json();
        
        // 1. Setup Feature Sliders
        setupSliders();
        
        // 2. Setup EDA Heatmap
        setupCorrelationHeatmap();
        
        // 3. Setup EDA Dropdown Selector and Chart
        setupDistributionSelector();
        
        // 4. Fill Model Metrics Summary cards
        setupMetricsSummary();
        
        // 5. Fill Confusion Matrices
        setupConfusionMatrices();
        
        // Run initial predictions
        calculateRisk();
        
    } catch (err) {
        console.error("Dashboard Init Error:", err);
        slidersContainer.innerHTML = `
            <div class="alert alert-info" style="grid-column: 1/-1; border-color: var(--color-danger); background-color: var(--color-danger-bg); color: var(--color-danger);">
                <p><strong>Error Loading Model Data:</strong> ${err.message}. Please run the python training script (<code>python train_models.py</code>) first to generate the models and visual metadata.</p>
            </div>
        `;
    }
}

// Dynamically generate sliders
function setupSliders() {
    slidersContainer.innerHTML = '';
    currentValues = {};
    
    modelData.features.forEach((feature, index) => {
        const meta = modelData.features_metadata[feature];
        const defaultValue = meta.mean;
        currentValues[feature] = defaultValue;
        
        const sliderGroup = document.createElement('div');
        sliderGroup.className = 'slider-group';
        
        // Round feature steps depending on scale of values (e.g. area needs less decimals, fractal dimension needs more)
        let step = 0.01;
        if (feature === 'area_mean') step = 1;
        else if (feature === 'radius_mean' || feature === 'perimeter_mean' || feature === 'texture_mean') step = 0.1;
        else if (feature === 'smoothness_mean' || feature === 'fractal_dimension_mean' || feature === 'concave points_mean') step = 0.0001;
        else if (feature === 'concavity_mean' || feature === 'compactness_mean' || feature === 'symmetry_mean') step = 0.001;

        sliderGroup.innerHTML = `
            <div class="slider-info">
                <span class="slider-label">${meta.name}</span>
                <span class="slider-val-box" id="val-${feature}">${defaultValue.toFixed(step.toString().split('.')[1]?.length || 0)}</span>
            </div>
            <div class="slider-input-row">
                <span class="slider-min-lbl">${meta.min.toFixed(step.toString().split('.')[1]?.length || 0)}</span>
                <input type="range" 
                    id="slider-${feature}" 
                    class="custom-range" 
                    min="${meta.min}" 
                    max="${meta.max}" 
                    step="${step}" 
                    value="${defaultValue}">
                <span class="slider-max-lbl">${meta.max.toFixed(step.toString().split('.')[1]?.length || 0)}</span>
            </div>
        `;
        
        slidersContainer.appendChild(sliderGroup);
        
        // Add event listener
        const sliderInput = sliderGroup.querySelector('input[type="range"]');
        sliderInput.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            currentValues[feature] = val;
            document.getElementById(`val-${feature}`).innerText = val.toFixed(step.toString().split('.')[1]?.length || 0);
            calculateRisk();
        });
    });
}

// Reset sliders
resetSlidersBtn.addEventListener('click', () => {
    if (!modelData) return;
    setupSliders();
    calculateRisk();
});

// Perform scaling and calculate real-time scores
function calculateRisk() {
    if (!modelData) return;
    
    // Scale inputs: z = (x - mean) / scale
    const scaledInputs = [];
    modelData.features.forEach((feature, index) => {
        const mean = modelData.scaler.mean[index];
        const scale = modelData.scaler.scale[index];
        const rawValue = currentValues[feature];
        const scaledVal = (rawValue - mean) / scale;
        scaledInputs.push(scaledVal);
    });
    
    // Logistic Regression Predict
    let lrScore = modelData.models.lr.intercept;
    scaledInputs.forEach((x_scaled, index) => {
        lrScore += x_scaled * modelData.models.lr.coef[index];
    });
    const lrProb = 1 / (1 + Math.exp(-lrScore)); // Sigmoid function
    
    // SVM Predict
    let svmScore = modelData.models.svm.intercept;
    scaledInputs.forEach((x_scaled, index) => {
        svmScore += x_scaled * modelData.models.svm.coef[index];
    });
    // Platt-like scaling approximation for SVM margin distance to probability
    const svmProb = 1 / (1 + Math.exp(-svmScore * 1.5));
    
    // Consensus risk (Average of both model probabilities)
    const consensusProb = (lrProb + svmProb) / 2;
    
    // Update Dial Gauge UI
    updateRiskDial(consensusProb);
    
    // Update individual model breakdowns
    updateModelBreakdowns(lrProb, svmProb);
}

function updateRiskDial(prob) {
    const percentage = Math.round(prob * 100);
    const riskLabel = document.getElementById('risk-percentage');
    const gaugeFill = document.getElementById('gauge-fill');
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictText = document.getElementById('verdict-text');
    const verdictDesc = document.getElementById('verdict-desc');
    
    riskLabel.innerText = `${percentage}%`;
    
    // Arc logic: SVG stroke-dasharray = 125.6 (Full half-circle arc length)
    // 0% = dashoffset 125.6, 100% = dashoffset 0
    const offset = 125.6 - (125.6 * prob);
    gaugeFill.style.strokeDashoffset = offset;
    
    if (prob >= 0.50) {
        // Malignant Risk State
        gaugeFill.className.baseVal = "gauge-fill malignant";
        verdictBanner.className = "verdict-banner malignant-banner";
        verdictText.innerText = "MALIGNANT DETECTED";
        verdictDesc.innerText = `Higher risk score of ${percentage}%. Clinical correlation and diagnostic testing are strongly recommended.`;
    } else {
        // Benign State
        gaugeFill.className.baseVal = "gauge-fill benign";
        verdictBanner.className = "verdict-banner benign-banner";
        verdictText.innerText = "BENIGN DETECTED";
        verdictDesc.innerText = `Lower risk score of ${percentage}%. The features strongly align with normal, non-cancerous cells.`;
    }
}

function updateModelBreakdowns(lrProb, svmProb) {
    // Logistic Regression progress bar
    const lrBar = document.getElementById('lr-progress');
    const lrLabel = document.getElementById('lr-prob-label');
    const lrPercent = Math.round(lrProb * 100);
    lrBar.style.width = `${lrPercent}%`;
    lrLabel.innerText = `${lrPercent}%`;
    
    // Set color based on risk levels
    if (lrPercent >= 50) lrBar.style.backgroundColor = 'var(--color-danger)';
    else lrBar.style.backgroundColor = 'var(--color-success)';

    // SVM progress bar
    const svmBar = document.getElementById('svm-progress');
    const svmLabel = document.getElementById('svm-prob-label');
    const svmPercent = Math.round(svmProb * 100);
    svmBar.style.width = `${svmPercent}%`;
    svmLabel.innerText = `${svmPercent}%`;
    
    if (svmPercent >= 50) svmBar.style.backgroundColor = 'var(--color-danger)';
    else svmBar.style.backgroundColor = 'var(--color-success)';
}

// 2. Correlation Heatmap rendering
function setupCorrelationHeatmap() {
    const container = document.getElementById('correlation-heatmap');
    container.innerHTML = '';
    
    const features = modelData.features;
    const meta = modelData.features_metadata;
    const matrix = modelData.correlation_matrix;
    
    // Set grid columns template dynamically (10 data cols + 1 y-label col)
    container.style.gridTemplateColumns = '120px repeat(10, 42px)';
    
    // Row 1: X labels
    // Empty corner cell
    const corner = document.createElement('div');
    corner.className = 'heatmap-cell heatmap-header-cell';
    container.appendChild(corner);
    
    features.forEach(f => {
        const cell = document.createElement('div');
        cell.className = 'heatmap-lbl-cell heatmap-lbl-x';
        cell.innerText = meta[f].name;
        cell.title = meta[f].name;
        container.appendChild(cell);
    });
    
    // Rest of rows: Y labels + Data Cells
    features.forEach((rowFeat, rowIndex) => {
        // Y label
        const yLbl = document.createElement('div');
        yLbl.className = 'heatmap-lbl-cell heatmap-lbl-y';
        yLbl.innerText = meta[rowFeat].name;
        container.appendChild(yLbl);
        
        features.forEach((colFeat, colIndex) => {
            const corrVal = matrix[rowIndex][colIndex];
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.innerText = corrVal.toFixed(2);
            cell.title = `Correlation between ${meta[rowFeat].name} and ${meta[colFeat].name}: ${corrVal.toFixed(4)}`;
            
            // Background color intensity depending on positive/negative correlation
            if (corrVal >= 0) {
                // Purple/violet positive correlation
                cell.style.backgroundColor = `rgba(139, 92, 246, ${corrVal})`;
            } else {
                // Rose/red negative correlation
                cell.style.backgroundColor = `rgba(244, 63, 94, ${Math.abs(corrVal)})`;
            }
            
            // Adjust text color based on cell lightness for readability
            if (Math.abs(corrVal) < 0.45) {
                cell.style.color = 'var(--text-secondary)';
            } else {
                cell.style.color = '#ffffff';
            }
            
            container.appendChild(cell);
        });
    });
}

// 3. EDA Tab Distribution Setup
function setupDistributionSelector() {
    featureSelector.innerHTML = '';
    
    modelData.features.forEach(f => {
        const option = document.createElement('option');
        option.value = f;
        option.innerText = modelData.features_metadata[f].name;
        featureSelector.appendChild(option);
    });
    
    // Add selector listener
    featureSelector.addEventListener('change', (e) => {
        renderDistributionChart(e.target.value);
    });
}

function renderDistributionChart(feature) {
    const canvas = document.getElementById('distribution-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Destroy previous chart instance if it exists
    if (distributionChartInstance) {
        distributionChartInstance.destroy();
    }
    
    const distData = modelData.distributions[feature];
    const featName = modelData.features_metadata[feature].name;
    
    distributionChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: distData.bins.map(v => v.toFixed(3)),
            datasets: [
                {
                    label: 'Benign samples',
                    data: distData.benign,
                    backgroundColor: 'rgba(16, 185, 129, 0.45)',
                    borderColor: 'rgba(16, 185, 129, 0.8)',
                    borderWidth: 1.5,
                    borderRadius: 4
                },
                {
                    label: 'Malignant samples',
                    data: distData.malignant,
                    backgroundColor: 'rgba(244, 63, 94, 0.45)',
                    borderColor: 'rgba(244, 63, 94, 0.8)',
                    borderWidth: 1.5,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 11 }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: (tooltipItems) => `Bin Center: ${tooltipItems[0].label}`
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: `${featName} Measurements`,
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b', font: { family: 'monospace', size: 9 } }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sample Count',
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

// 4. Model Evaluation metrics setups
function setupMetricsSummary() {
    const models = ['lr', 'svm'];
    models.forEach(model => {
        const accuracy = modelData.metrics[model].accuracy;
        const precision = modelData.metrics[model].precision;
        const recall = modelData.metrics[model].recall;
        const f1 = modelData.metrics[model].f1;
        
        document.getElementById(`${model}-metric-accuracy`).innerText = `${(accuracy * 100).toFixed(1)}%`;
        document.getElementById(`${model}-metric-precision`).innerText = `${(precision * 100).toFixed(1)}%`;
        document.getElementById(`${model}-metric-recall`).innerText = `${(recall * 100).toFixed(1)}%`;
        document.getElementById(`${model}-metric-f1`).innerText = `${(f1 * 100).toFixed(1)}%`;
    });
}

// 5. Populate confusion matrices
function setupConfusionMatrices() {
    // Confusion matrices structure: [[tn, fp], [fn, tp]]
    // LR
    const cmLr = modelData.confusion_matrices.lr;
    document.getElementById('cm-lr-tn').innerText = cmLr[0][0]; // TN
    document.getElementById('cm-lr-fn').innerText = cmLr[1][0]; // FN (Actual M predicted B)
    document.getElementById('cm-lr-fp').innerText = cmLr[0][1]; // FP (Actual B predicted M)
    document.getElementById('cm-lr-tp').innerText = cmLr[1][1]; // TP
    
    // SVM
    const cmSvm = modelData.confusion_matrices.svm;
    document.getElementById('cm-svm-tn').innerText = cmSvm[0][0];
    document.getElementById('cm-svm-fn').innerText = cmSvm[1][0];
    document.getElementById('cm-svm-fp').innerText = cmSvm[0][1];
    document.getElementById('cm-svm-tp').innerText = cmSvm[1][1];
}

// 6. Draw ROC Chart
function renderRocChart() {
    const canvas = document.getElementById('roc-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (rocChartInstance) {
        rocChartInstance.destroy();
    }
    
    // Generate diagonal random chance line (x = y)
    const randomChanceData = [];
    for (let i = 0; i <= 10; i++) {
        randomChanceData.push({ x: i / 10, y: i / 10 });
    }
    
    // Parse ROC coordinates from JSON
    const lrRoc = modelData.roc_curves.lr;
    const lrPoints = lrRoc.fpr.map((fprVal, idx) => ({ x: fprVal, y: lrRoc.tpr[idx] }));
    
    const svmRoc = modelData.roc_curves.svm;
    const svmPoints = svmRoc.fpr.map((fprVal, idx) => ({ x: fprVal, y: svmRoc.tpr[idx] }));
    
    rocChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Logistic Regression',
                    data: lrPoints,
                    showLine: true,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 4
                },
                {
                    label: 'Linear SVM',
                    data: svmPoints,
                    showLine: true,
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.2)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 4
                },
                {
                    label: 'Random Guess',
                    data: randomChanceData,
                    showLine: true,
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 11 }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'False Positive Rate (1 - Specificity)',
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'True Positive Rate (Sensitivity)',
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

// Fire when page is loaded
window.addEventListener('DOMContentLoaded', initDashboard);
