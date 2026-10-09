# CEE-Split

云端、边缘和本地协同的 Unity 分层渲染系统，包含手动调控和强化学习训练代码。

第一次运行：按 [快速开始](快速开始.md) 操作。

| 目录 | 内容 |
|---|---|
| `unity-split-render/` | 主系统 Unity 工程，打开这个工程运行分层渲染 |
| `render-tests/` | Python 调控服务器、训练、评估代码、运行模型和数据；见 [说明](render-tests/README.md) |
| `qoe-proxy-unity/` | QoE 校准和数据采集 Unity 工程 |

环境：Unity **6000.4.4f1**、Python **3.10/3.11**、Git LFS；跨机器连接使用 Tailscale。

克隆后执行 `git lfs pull`，下载模型、贴图和 `webserver.exe`。私有仓库需要先添加协作者。

Unity 缓存和打包程序未上传，可自行生成。QoE 采集的原始 PNG 帧另行存档，仓库保留元数据和训练 CSV。

论文、实验结果和训练日志不纳入仓库。
