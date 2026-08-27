import csv
import math
import json
import os
import random

# Define paths
data_path = os.path.join("data", "clean-data.csv")
web_dir = "web"
output_path = os.path.join(web_dir, "model_data.json")

os.makedirs(web_dir, exist_ok=True)

# 1. Load data using pure Python csv module
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

dataset = []
with open(data_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Encode diagnosis: M = 1, B = 0
        diag = 1 if row['diagnosis'] == 'M' else 0
        
        # Extract features
        feat_vals = {}
        for feat in features:
            feat_vals[feat] = float(row[feat])
        
        dataset.append({
            'diagnosis': diag,
            'features': feat_vals
        })

# Set seed for reproducibility
random.seed(42)
random.shuffle(dataset)

# 2. Split train/test (70% train, 30% test)
split_idx = int(len(dataset) * 0.7)
train_data = dataset[:split_idx]
test_data = dataset[split_idx:]

print(f"Dataset loaded. Train size: {len(train_data)}, Test size: {len(test_data)}")

# 3. Calculate mean and std on train data (StandardScaler logic)
scaler_mean = {}
scaler_std = {}

for feat in features:
    vals = [row['features'][feat] for row in train_data]
    mean_val = sum(vals) / len(vals)
    var_val = sum((x - mean_val) ** 2 for x in vals) / len(vals)
    std_val = math.sqrt(var_val) if var_val > 0 else 1.0
    
    scaler_mean[feat] = mean_val
    scaler_std[feat] = std_val

# Scale helper
def scale_features(feat_dict):
    return [
        (feat_dict[feat] - scaler_mean[feat]) / scaler_std[feat]
        for feat in features
    ]

# Prepare training vectors
X_train = [scale_features(row['features']) for row in train_data]
y_train = [row['diagnosis'] for row in train_data]

X_test = [scale_features(row['features']) for row in test_data]
y_test = [row['diagnosis'] for row in test_data]

# 4. Train Logistic Regression
# Parameters
lr_w = [0.0] * len(features)
lr_b = 0.0
lr_alpha = 0.1
lr_epochs = 2000

for epoch in range(lr_epochs):
    # Gradient accumulators
    dw = [0.0] * len(features)
    db = 0.0
    
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]
        
        # Predict probability
        z = sum(xi[j] * lr_w[j] for j in range(len(features))) + lr_b
        # Sigmoid
        p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else 1.0 - 1.0 / (1.0 + math.exp(z))
        
        error = p - yi
        for j in range(len(features)):
            dw[j] += error * xi[j]
        db += error
        
    # Update weights
    n = len(X_train)
    for j in range(len(features)):
        lr_w[j] -= lr_alpha * (dw[j] / n)
    lr_b -= lr_alpha * (db / n)

# 5. Train Linear SVM (Gradient Descent on Hinge Loss with L2 regularization)
svm_w = [0.0] * len(features)
svm_b = 0.0
svm_alpha = 0.01
svm_epochs = 2000
svm_C = 1.0

# Encode labels as 1 and -1 for SVM
y_train_svm = [1 if y == 1 else -1 for y in y_train]
y_test_svm = [1 if y == 1 else -1 for y in y_test]

for epoch in range(svm_epochs):
    # Subgradient updates
    dw = [w for w in svm_w] # L2 regularization term gradient part 1
    db = 0.0
    
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train_svm[i]
        
        # Decision boundary
        score = sum(xi[j] * svm_w[j] for j in range(len(features))) + svm_b
        
        # Check if margin condition is violated
        if yi * score < 1.0:
            for j in range(len(features)):
                dw[j] -= svm_C * yi * xi[j]
            db -= svm_C * yi
            
    # Update weights
    n = len(X_train)
    for j in range(len(features)):
        svm_w[j] -= svm_alpha * (dw[j] / n)
    svm_b -= svm_alpha * (db / n)

# Prediction helper functions
def predict_lr(xi):
    z = sum(xi[j] * lr_w[j] for j in range(len(features))) + lr_b
    prob = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else 1.0 - 1.0 / (1.0 + math.exp(z))
    return 1 if prob >= 0.5 else 0, prob

def predict_svm(xi):
    score = sum(xi[j] * svm_w[j] for j in range(len(features))) + svm_b
    # Map score to pseudo-probability for ROC and metrics
    prob = 1.0 / (1.0 + math.exp(-score * 1.5))
    return 1 if score >= 0.0 else 0, prob

# 6. Evaluate Models on Test Set
def get_metrics(predictions, actual):
    tp = fp = tn = fn = 0
    for pred, act in zip(predictions, actual):
        if pred == 1 and act == 1: tp += 1
        elif pred == 1 and act == 0: fp += 1
        elif pred == 0 and act == 0: tn += 1
        elif pred == 0 and act == 1: fn += 1
        
    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': [[tn, fp], [fn, tp]]
    }

lr_preds = []
lr_probs = []
svm_preds = []
svm_probs = []

for xi in X_test:
    pred, prob = predict_lr(xi)
    lr_preds.append(pred)
    lr_probs.append(prob)
    
    pred_s, prob_s = predict_svm(xi)
    svm_preds.append(pred_s)
    svm_probs.append(prob_s)

lr_metrics = get_metrics(lr_preds, y_test)
svm_metrics = get_metrics(svm_preds, y_test)

# 7. Generate ROC curve data
def generate_roc(probs, actual):
    # Zip and sort by probability descending
    zipped = sorted(zip(probs, actual), key=lambda x: x[0], reverse=True)
    
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    total_pos = sum(actual)
    total_neg = len(actual) - total_pos
    
    tp_acc = 0
    fp_acc = 0
    
    for prob, act in zipped:
        if act == 1:
            tp_acc += 1
        else:
            fp_acc += 1
            
        tpr = tp_acc / total_pos if total_pos > 0 else 0.0
        fpr = fp_acc / total_neg if total_neg > 0 else 0.0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        
    # Downsample to maximum 50 points to keep JSON small
    indices = [int(i) for i in range(len(fpr_list))]
    if len(fpr_list) > 50:
        step = len(fpr_list) / 50
        indices = [int(i * step) for i in range(50)]
        if (len(fpr_list) - 1) not in indices:
            indices.append(len(fpr_list) - 1)
            
    return {
        'fpr': [fpr_list[i] for i in indices],
        'tpr': [tpr_list[i] for i in indices]
    }

roc_lr = generate_roc(lr_probs, y_test)
roc_svm = generate_roc(svm_probs, y_test)

# 8. Feature stats (min, max, mean, std, name) for Sliders
features_metadata = {}
raw_features_by_col = {feat: [row['features'][feat] for row in dataset] for feat in features}

for feat in features:
    vals = raw_features_by_col[feat]
    mean_val = sum(vals) / len(vals)
    var_val = sum((x - mean_val) ** 2 for x in vals) / len(vals)
    
    features_metadata[feat] = {
        'min': min(vals),
        'max': max(vals),
        'mean': mean_val,
        'std': math.sqrt(var_val),
        'name': feat.replace('_mean', '').replace(' ', ' ').capitalize()
    }

# 9. Correlation matrix calculation in pure Python
corr_matrix = []
for f1 in features:
    row_corr = []
    vals1 = raw_features_by_col[f1]
    mean1 = sum(vals1) / len(vals1)
    std1 = math.sqrt(sum((x - mean1) ** 2 for x in vals1))
    
    for f2 in features:
        vals2 = raw_features_by_col[f2]
        mean2 = sum(vals2) / len(vals2)
        std2 = math.sqrt(sum((x - mean2) ** 2 for x in vals2))
        
        # Covariance
        cov = sum((x1 - mean1) * (x2 - mean2) for x1, x2 in zip(vals1, vals2))
        corr = cov / (std1 * std2) if (std1 * std2) > 0 else 0.0
        row_corr.append(corr)
    corr_matrix.append(row_corr)

# 10. Feature distributions (bins, benign histogram, malignant histogram)
distributions = {}
for feat in features:
    vals = raw_features_by_col[feat]
    min_val = min(vals)
    max_val = max(vals)
    
    # 12 bins
    bin_edges = [min_val + i * (max_val - min_val) / 12 for i in range(13)]
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(12)]
    
    benign_hist = [0] * 12
    malignant_hist = [0] * 12
    
    for row in dataset:
        val = row['features'][feat]
        diag = row['diagnosis']
        
        # Find which bin it belongs to
        bin_idx = 11 # Default to last bin
        for i in range(12):
            if val < bin_edges[i+1]:
                bin_idx = i
                break
                
        if diag == 1:
            malignant_hist[bin_idx] += 1
        else:
            benign_hist[bin_idx] += 1
            
    distributions[feat] = {
        'bins': [round(x, 4) for x in bin_centers],
        'benign': benign_hist,
        'malignant': malignant_hist
    }

# 11. Compile export payload
export_data = {
    'features': features,
    'scaler': {
        'mean': [scaler_mean[f] for f in features],
        'scale': [scaler_std[f] for f in features]
    },
    'models': {
        'lr': {
            'coef': lr_w,
            'intercept': lr_b
        },
        'svm': {
            'coef': svm_w,
            'intercept': svm_b
        }
    },
    'metrics': {
        'lr': {
            'accuracy': lr_metrics['accuracy'],
            'precision': lr_metrics['precision'],
            'recall': lr_metrics['recall'],
            'f1': lr_metrics['f1']
        },
        'svm': {
            'accuracy': svm_metrics['accuracy'],
            'precision': svm_metrics['precision'],
            'recall': svm_metrics['recall'],
            'f1': svm_metrics['f1']
        }
    },
    'confusion_matrices': {
        'lr': lr_metrics['cm'],
        'svm': svm_metrics['cm']
    },
    'roc_curves': {
        'lr': roc_lr,
        'svm': roc_svm
    },
    'features_metadata': features_metadata,
    'correlation_matrix': corr_matrix,
    'distributions': distributions
}

with open(output_path, 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"Pure Python Model training complete! Model data exported successfully to {output_path}!")
