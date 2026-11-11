# v1.5 运行依赖

代码来源是当前 v1.5 正式源码交付包。源码保持发布实现，不改变冻结训练算法或评价协议。

## 冻结环境与资产

运行根目录为 `/home/zhangxu/hx/HX001`，当前服务器解释器为该目录 `.venv/bin/python`。已有环境还需要：

- 原生结构头：`native_anchor_bridge_261109_v1/geometry.pt`。
- 当前发布锁：`native_anchor_bridge_261109_v1/pruned/lock.json`。
- `visual_evidence_texture_261109_v15` 的跨尺度预测、语义读出、相位组合及其选择配置。
- `texture_retention_261109_v15` 的纹理、颜色和区域纠正权重与选择配置。
- `visual_trajectory_mix_261108_v1`、`material_detail_projection_261108_v1` 的冻结发布配置。
- `visual_region_condition_261108_v1/code/probe.py` 等上游 VOSR 适配器，以及 `camera_feedback_261108_v1`、`paired_detail_readout_261108_v1` 的冻结相机与读出模块。
- 官方 VOSR 0.5B 多步权重、官方轻量解码器 `sd21_lwdecoder.pth`、DINOv2 和对应的相机资产。

权重及完整冻结发布包继续保存在服务器。此仓库不提供这些二进制资产，不能单靠安装 PyTorch 完成独立推理。原生与视觉整合发布包在服务器 `native_anchor_bridge_261109_v1/system_release/native_visual_release.zip`，其历史 SHA-256 为 `ee0e629f5f6e78452bab75a77cd0a114af8ee78f2a90455ab705014f9fc2b7b6`。

## 源码布局

`src/` 按运行目录保留模块层级。上游归档把多个阶段代码合并在一个本地任务目录；这里另外把 `mix_study.py` 和 `refine_study.py` 等按实际运行目录放置相同字节的副本。逐文件来源可在 `provenance.json` 核对。

远程登录、服务器调度、口令、原始数据、权重和实验导出图均不属于本次同步。历史研究模块通过导入复用函数；除当前单图入口外，其余脚本的训练和确认命令须结合对应原协议使用。
