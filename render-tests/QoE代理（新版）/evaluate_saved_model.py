"""使用已保存的 ONNX 模型补算测试集回归指标，无需重新训练。"""

import glob
import json
import os

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = r"D:\Unity\QoEProxy\DatasetOutputs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "qoe_proxy_model.onnx")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler_x.pkl")


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def predict(session, x):
    """兼容当前固定 batch=1 的 ONNX，以及以后导出的动态 batch 模型。"""
    model_input = session.get_inputs()[0]
    if model_input.shape[0] == 1:
        return np.vstack([
            session.run(None, {model_input.name: row[None, :].astype(np.float32)})[0]
            for row in x
        ])
    return session.run(None, {model_input.name: x.astype(np.float32)})[0]


csv_files = glob.glob(
    os.path.join(BASE_DIR, "**", "final_training_dataset.csv"), recursive=True
)
if not csv_files:
    raise FileNotFoundError("没有找到 final_training_dataset.csv")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"没有找到模型: {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"没有找到输入标准化器: {SCALER_PATH}")

df = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
x = df[["res_scale", "qp"]].values
y = df[["vmaf", "bitrate"]].values
x_scaled = joblib.load(SCALER_PATH).transform(x)

# 与原训练脚本保持完全相同的测试集划分规则。
indices = np.arange(len(df))
_, x_test, _, y_test, _, test_indices = train_test_split(
    x_scaled, y, indices, test_size=0.2, random_state=42
)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
y_pred = predict(session, x_test)

vmaf_metrics = calculate_metrics(y_test[:, 0], y_pred[:, 0])
bitrate_metrics = calculate_metrics(y_test[:, 1], y_pred[:, 1])

report = {
    "evaluated_model": MODEL_PATH,
    "dataset": {
        "csv_files": csv_files,
        "total_samples": int(len(df)),
        "test_samples": int(len(x_test)),
        "test_size": 0.2,
        "random_state": 42,
    },
    "vmaf": vmaf_metrics,
    "bitrate_kbps": bitrate_metrics,
}

json_path = os.path.join(SCRIPT_DIR, "evaluation_metrics.json")
csv_path = os.path.join(SCRIPT_DIR, "evaluation_metrics.csv")
predictions_path = os.path.join(SCRIPT_DIR, "test_predictions.csv")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

pd.DataFrame([
    {"target": "vmaf", "unit": "score", **vmaf_metrics},
    {"target": "bitrate", "unit": "Kbps", **bitrate_metrics},
]).to_csv(csv_path, index=False, encoding="utf-8-sig")

pd.DataFrame({
    "sample_index": test_indices,
    "vmaf_true": y_test[:, 0],
    "vmaf_pred": y_pred[:, 0],
    "vmaf_error": y_pred[:, 0] - y_test[:, 0],
    "bitrate_true_kbps": y_test[:, 1],
    "bitrate_pred_kbps": y_pred[:, 1],
    "bitrate_error_kbps": y_pred[:, 1] - y_test[:, 1],
}).to_csv(predictions_path, index=False, encoding="utf-8-sig")

print(f"测试样本数: {len(x_test)}")
print(
    f"VMAF   MAE={vmaf_metrics['mae']:.4f}, "
    f"RMSE={vmaf_metrics['rmse']:.4f}, "
    f"R2={vmaf_metrics['r2']:.4f}, "
    f"Pearson r={vmaf_metrics['pearson_r']:.4f}"
)
print(
    f"Bitrate MAE={bitrate_metrics['mae']:.4f} Kbps, "
    f"RMSE={bitrate_metrics['rmse']:.4f} Kbps, "
    f"R2={bitrate_metrics['r2']:.4f}, "
    f"Pearson r={bitrate_metrics['pearson_r']:.4f}"
)
print(f"指标已保存: {json_path}")
print(f"逐样本预测已保存: {predictions_path}")
