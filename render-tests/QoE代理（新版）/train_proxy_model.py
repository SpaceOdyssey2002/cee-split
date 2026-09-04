import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob  # 🌟 新增：用于快速查找文件
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = 'serif'

# --- 配置路径 ---
BASE_DIR = r"D:\Unity\QoEProxy\DatasetOutputs"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vp9_proxy")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def output_path(filename):
    """将训练产物固定保存到本脚本所在目录。"""
    return os.path.join(OUTPUT_DIR, filename)


def calculate_regression_metrics(y_true, y_pred):
    """计算原始量纲下的常用回归指标。"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }

# 1. 定义核心多任务MLP模型
class QualityProxyMLP(nn.Module):
    def __init__(self):
        super(QualityProxyMLP, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(2, 64),    # res_scale 和 qp
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.vmaf_head = nn.Linear(64, 1)
        self.bitrate_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared_layers(x)
        vmaf = self.vmaf_head(features)
        bitrate = self.bitrate_head(features)
        return torch.cat((vmaf, bitrate), dim=1)

class ExportWrapper(nn.Module):
    def __init__(self, base_model, scaler_y):
        super(ExportWrapper, self).__init__()
        self.base_model = base_model
        self.y_mean = nn.Parameter(torch.FloatTensor(scaler_y.mean_), requires_grad=False)
        self.y_scale = nn.Parameter(torch.FloatTensor(scaler_y.scale_), requires_grad=False)

    def forward(self, x):
        normalized_output = self.base_model(x)
        real_output = normalized_output * self.y_scale + self.y_mean
        return real_output

# 2. 准备数据 (🌟 修改：合并多个文件夹下的 CSV)
print(f"🔍 正在扫描目录: {BASE_DIR}")

# 查找所有子目录下的 final_training_dataset.csv
all_csv_files = glob.glob(os.path.join(BASE_DIR, "**", "final_training_dataset_vp9.csv"), recursive=True)

if not all_csv_files:
    print("❌ 未找到任何 final_training_dataset.csv 文件，请先运行数据处理脚本。")
    exit()

print(f"合计找到 {len(all_csv_files)} 个数据文件:")
for f in all_csv_files:
    print(f" - {f}")

# 读取并合并
df_list = [pd.read_csv(f) for f in all_csv_files]
df = pd.concat(df_list, ignore_index=True)

print(f"📊 数据合并完成，总样本数: {len(df)}")

# 提取特征和标签
# 注意：确保你的 CSV 列名是 res_scale, qp, vmaf, bitrate
X = df[['res_scale', 'qp']].values
Y = df[['vmaf', 'bitrate']].values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, output_path('scaler_x.pkl'))

scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)

all_indices = np.arange(len(df))
X_train, X_test, Y_train, Y_test, _, test_indices = train_test_split(
    X_scaled, Y_scaled, all_indices, test_size=0.2, random_state=42
)

X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)

# 3. 训练模型
torch.manual_seed(42)
np.random.seed(42)
model = QualityProxyMLP()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

epochs = 1500
history_total_loss = []
history_vmaf_loss = []  # 🌟 新增：记录 VMAF 损失

print("\n🚀 开始训练 QoE 代理模型...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)

    loss_vmaf = criterion(predictions[:, 0], Y_train_t[:, 0])
    loss_bitrate = criterion(predictions[:, 1], Y_train_t[:, 1])
    loss = loss_vmaf + loss_bitrate

    loss.backward()
    optimizer.step()

    history_total_loss.append(loss.item())
    history_vmaf_loss.append(loss_vmaf.item())  # 🌟 记录

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d}: Total Loss={loss.item():.4f} | VMAF_Loss={loss_vmaf.item():.4f}")

# 4. 在测试集上进行验证评估
model.eval()
with torch.no_grad():
    pred_test_raw = model(torch.FloatTensor(X_test))
    pred_test_real = scaler_Y.inverse_transform(pred_test_raw.numpy())
    Y_test_real = scaler_Y.inverse_transform(Y_test)

    vmaf_metrics = calculate_regression_metrics(Y_test_real[:, 0], pred_test_real[:, 0])
    bitrate_metrics = calculate_regression_metrics(Y_test_real[:, 1], pred_test_real[:, 1])
    print("\n✅ 测试集精度评估")
    print(
        f"VMAF:   MAE={vmaf_metrics['mae']:.4f} | "
        f"RMSE={vmaf_metrics['rmse']:.4f} | R2={vmaf_metrics['r2']:.4f} | "
        f"Pearson r={vmaf_metrics['pearson_r']:.4f}"
    )
    print(
        f"Bitrate: MAE={bitrate_metrics['mae']:.4f} Kbps | "
        f"RMSE={bitrate_metrics['rmse']:.4f} Kbps | R2={bitrate_metrics['r2']:.4f} | "
        f"Pearson r={bitrate_metrics['pearson_r']:.4f}"
    )

# The online scheduler only observes resolution and QP, so its prediction is the
# expected quality of a configuration rather than the exact quality of an
# individual scene/view. Report both evaluation granularities explicitly.
configuration_table = (
    df.groupby(['res_scale', 'qp'], as_index=False)[['vmaf', 'bitrate']]
      .mean()
)
configuration_X = scaler_X.transform(
    configuration_table[['res_scale', 'qp']].values
)
with torch.no_grad():
    configuration_pred_normalized = model(
        torch.FloatTensor(configuration_X)
    ).numpy()
configuration_pred = scaler_Y.inverse_transform(configuration_pred_normalized)
configuration_vmaf_metrics = calculate_regression_metrics(
    configuration_table['vmaf'].values, configuration_pred[:, 0]
)
configuration_bitrate_metrics = calculate_regression_metrics(
    configuration_table['bitrate'].values, configuration_pred[:, 1]
)
print("\n✅ 25 个配置均值的拟合精度")
print(
    f"VMAF:   MAE={configuration_vmaf_metrics['mae']:.4f} | "
    f"RMSE={configuration_vmaf_metrics['rmse']:.4f} | "
    f"R2={configuration_vmaf_metrics['r2']:.4f}"
)
print(
    f"Bitrate: MAE={configuration_bitrate_metrics['mae']:.4f} Kbps | "
    f"RMSE={configuration_bitrate_metrics['rmse']:.4f} Kbps | "
    f"R2={configuration_bitrate_metrics['r2']:.4f}"
)

# 保存完整指标，避免训练结束后只在终端中显示
metrics_report = {
    "dataset": {
        "csv_files": all_csv_files,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": 0.2,
        "random_state": 42,
    },
    "vmaf": vmaf_metrics,
    "bitrate_kbps": bitrate_metrics,
    "configuration_mean": {
        "vmaf": configuration_vmaf_metrics,
        "bitrate_kbps": configuration_bitrate_metrics,
    },
}

with open(output_path("evaluation_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_report, f, ensure_ascii=False, indent=2)

metrics_table = pd.DataFrame([
    {"target": "vmaf", "unit": "score", **vmaf_metrics},
    {"target": "bitrate", "unit": "Kbps", **bitrate_metrics},
])
metrics_table.to_csv(output_path("evaluation_metrics.csv"), index=False, encoding="utf-8-sig")

predictions_table = pd.DataFrame({
    "sample_index": test_indices,
    "vmaf_true": Y_test_real[:, 0],
    "vmaf_pred": pred_test_real[:, 0],
    "vmaf_error": pred_test_real[:, 0] - Y_test_real[:, 0],
    "bitrate_true_kbps": Y_test_real[:, 1],
    "bitrate_pred_kbps": pred_test_real[:, 1],
    "bitrate_error_kbps": pred_test_real[:, 1] - Y_test_real[:, 1],
})
predictions_table.to_csv(output_path("test_predictions.csv"), index=False, encoding="utf-8-sig")
print(f"📄 评估指标已保存至 {output_path('evaluation_metrics.json')} 和 evaluation_metrics.csv")
print(f"📄 测试集逐样本预测已保存至 {output_path('test_predictions.csv')}")

configuration_predictions_table = configuration_table.copy()
configuration_predictions_table['vmaf_pred'] = configuration_pred[:, 0]
configuration_predictions_table['vmaf_error'] = (
    configuration_pred[:, 0] - configuration_table['vmaf'].values
)
configuration_predictions_table['bitrate_pred_kbps'] = configuration_pred[:, 1]
configuration_predictions_table['bitrate_error_kbps'] = (
    configuration_pred[:, 1] - configuration_table['bitrate'].values
)
configuration_predictions_table.to_csv(
    output_path("configuration_mean_predictions.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 5. 导出 ONNX 模型
export_model = ExportWrapper(model, scaler_Y)
export_model.eval()
dummy_input = torch.randn(1, 2)
torch.onnx.export(export_model, dummy_input, output_path("qoe_proxy_model.onnx"),
                  input_names=['config_params'],
                  output_names=['qoe_metrics'])
print(f"💾 ONNX 模型已保存至 {output_path('qoe_proxy_model.onnx')}")

# 画图
plt.figure(figsize=(6, 5))
plt.scatter(Y_test_real[:, 0], pred_test_real[:, 0], alpha=0.5, color='#3498db', edgecolors='k')
plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal')
plt.title(f'VMAF Prediction', fontweight='bold')
plt.xlabel('Ground Truth VMAF', fontweight='bold')
plt.ylabel('Predicted VMAF', fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig(output_path('vmaf_prediction.png'), dpi=300, bbox_inches='tight')
plt.close()
print("📸 预测散点图已保存至 vmaf_prediction.png")

plt.figure(figsize=(6, 4))
plt.plot(history_vmaf_loss, label='VMAF Training Loss', color='teal')
plt.title('VMAF Loss Evolution', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig(output_path('vmaf_loss_curve.png'), dpi=300)
plt.close()
