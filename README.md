# VOSR for me

当前 **v1.5 原生结构基础与选择性视觉纠错**的核心源码，整理于 2026-11-11。以冻结的 VOSR 0.5B 多步模型为基础，两条轨迹共享前 16 步；原生路径读取桥前 RGB，执行可信配准和带符号残差纠正，最后与相机校准及跨尺度视觉估计进行分频纠错。

## 核心入口

| 代码 | 作用 |
| --- | --- |
| `src/native_anchor_bridge_261109_v1/code/infer_system.py` | 当前单图推理入口，共享轨迹、读出与融合 |
| 同目录 `native_model.py` | 九候选位移、可信细频采样、原生结构头和带符号残差 |
| 同目录 `route.py` | 原生估计与视觉估计的分频纠错 |
| `src/visual_evidence_texture_261109_v15/code/` | 同图跨尺度视觉预测及视觉纠错 |
| `src/texture_retention_261109_v15/code/` | 分频工具、四门读出、上游生成调用 |
| `src/material_direction_flow_261108_v1/code/` | 双轨去噪与材质方向约束 |
| `src/material_detail_projection_261108_v1/`、`src/visual_trajectory_mix_261108_v1/` | 按实际服务器目录组织的上游依赖源文件 |

`provenance.json` 记录原交付包身份、文件来源和逐文件 SHA-256；发布源文件保持原字节。部分 `study` 模块同时包含训练/评价函数，推理通过导入复用其中的模型与读出函数。仓库保留这些依赖，避免只上传三个入口而丢失实现。

## 当前配置

`configs/current_recipe.json` 摘录当前已发布配置：`bridge=none`、`method=conf`、`spec=[0.6,0.5,"evidence"]`。它说明原生路径绕过额外平滑桥，采用可信细频采样与证据门控。运行仍读取发布资产中的 `pruned/lock.json`。

## 在既有发布环境运行

当前代码保留原服务器根目录 `/home/zhangxu/hx/HX001` 和模块加载方式。`src/` 对应该根目录下的源码布局。本仓库是源码同步，需要既有冻结运行环境和发布资产；权重、数据集、特征缓存与完整实验结果保留服务器。

在已部署 v1.5 环境执行：

```bash
cd /home/zhangxu/hx/HX001/native_anchor_bridge_261109_v1
/home/zhangxu/hx/HX001/.venv/bin/python code/infer_system.py \
  --input /path/to/lr.png --output /path/to/sr.png --seed 6
```

输入为 `128×128` RGB，输出为 `512×512` RGB。环境为 Linux、PyTorch 与 CUDA，包含 NumPy、Pillow、pyiqa，以及上游 VOSR/DINOv2 运行依赖。冻结模型、相机资产和依赖位置见 [运行依赖](docs/runtime.md)。本次源码同步只做语法、JSON 解析和逐文件散列核验，未在本地重新执行 GPU 推理。

```bash
python tools/verify_source.py
```

## 更新约定

修改当前实现时保持配置、来源清单与说明一致。提出新结构或重训时建立明确的新版本；原生估计、视觉校准和最终融合的作用分别评价。图像感知指标的改善按其指标含义解释，结构恢复结论使用位置、形状与相位证据。
