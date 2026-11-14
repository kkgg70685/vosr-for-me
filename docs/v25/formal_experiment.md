# VOSR v2.5 正式实验安排与核心伪代码

日期：2026-11-14。实现版本 `2.5.1-audit`，协议 `V25-OBSERVABLE-TRANSPORT-20261114-R1`。本文件替代 R0 的运行安排；冒烟结果见 [审计报告](audit.md)。**当前有服务器冒烟结果，尚无独立确认集结论。**

## 1. 问题与固定预算

主张是连续几何修订能提高超分后的对应精度，同时保持纹理质量。图像质量变好而匹配变差不算主张成立；没有更新也不算验证有效。HR 和单应矩阵只进入构造/评价，绝不输入修订求解器。

| 阶段 | 父图与输入 | 目的 | 执行状态 |
| --- | --- | --- | --- |
| 实际模型冒烟 | 2 个历史自然父图；2 张实拍 LR；另构造 6 个视图、4 对非恒等单应 | 接口、指标、图像输出和失败处理 | 已完成，结果不作为独立确认 |
| 固定开发 | 8 个已暴露父图，每父图3视图、2关系，共24视图/16对；若有实拍 LR 再做8张真实相机检查 | 检查激活、失败原因和代价，只运行冻结 R1；不做网格搜索 | 待执行 |
| 一次独立确认 | 32 个未用于过去选择/开发的物理父图，96视图/64对；有真实配对时另报告32张真实相机组 | 检验固定主张；确认结果不得反过来修改阈值 | 待核实历史排除清单后执行 |

数据不绑定某一命名数据集。使用服务器已有、来源和许可清楚的 HR 图像即可；不能把同一场景不同裁块/相机/变换算作不同父图。数据不足32个独立父图时如实报告规模，不补抄历史得分、不把结果标成已满足本协议。数量是固定探索预算，不是统计功效保证。

构造器 `prepare.py` 对每张 HR 先做固定中心放大与两种 ±3°、平移、轻微透视变换，然后面积降采样4倍。最终 HR 为512，LR为128；自动检查采样角点在源图有效区域，避免黑边捷径。先生成 HR 视图，再退化，再分别运行生成器。保留原始来源 SHA、变换矩阵、量化规则与父图身份。合成组的相机为 `synthetic-area`，真实组为 `deployed`，两组不得混成一个均值。真实配对中残余配准误差需要写成解释限制。

## 2. 四个固定方法

| 输出名 | 方法 | 对照意义 |
| --- | --- | --- |
| native | 冻结 VOSR；seed=6、25 Euler steps、guidance=-0.5 | 历史 native 配置，不能写成上游官方默认配置 |
| translation_only | 仅二维全局平移；保留相同谱门槛 | 判断全图平移是否足够；若不激活，仅是这个受门控平移的对照 |
| smooth_unfiltered | 相同18维连续场，移除观测分辨率谱门槛，保留数值秩截断 | 判断是否仅普通平滑配准起作用 |
| observable_transport | 完整 R1 | 检验谱筛选的增量 |

每个 LR 只生成一次浮点 native；所有方法共享这份张量。保持相同颜色约束、积分次数、三轮优化和四档回溯。固定 seed=6；不多种子，不进行人工盲评，不搜索多个阈值挑最高分。发现实现 bug 可以修复，但需新协议身份和重跑受影响结果；科研失败不能称为 bug 后悄悄更换口径。

## 3. 主次指标与口径

| 类别 | 指标 | 计算与解释 |
| --- | --- | --- |
| 主指标 | 单应角点误差 AUC@3/5/10 ↑ | 固定 SIFT-2048、ratio=.75、RANSAC3 HR px/2000/.995；四角平均距离；阈值单位为 HR 像素 |
| 主指标分母 | 拟合失败率 ↓ | 无描述子、少于4匹配、无解或非法矩阵都保留为失败，不筛去困难样本 |
| 感知 | LPIPS ↓、DISTS ↓ | 所有方法同样截断到 [0,1]，同一 HR；要求服务器真实预训练权重，缺失直接报错 |
| 像素 | RGB PSNR ↑ | 全图RGB、无裁边、[0,1] 浮点；另保留 raw_RGB_PSNR 诊断越界，不混称 Y-PSNR |
| 高频正确性 | HF_RMSE ↓ | RGB减去固定5阶二项式低通，与HR高频的RMSE，裁4 HR像素；本项目诊断量，不冒称现成论文主指标 |
| 高频强度 | HF_RMS | 同一高通的能量；更大可来自噪声，更小可来自插值模糊，不能单独解释成更好 |
| 激活与安全 | 接受图像数、秩、残差、位移、最小离散Jacobian、逆误差 | 拒绝样本仍进入最终图像/匹配指标分母；秩不是正确概率 |
| 成本 | 每图生成时间、几何时间、峰值显存 | CUDA同步计时；native的几何时间0不代表生成免费；峰值是本进程PyTorch分配，不是整卡占用 |

先在每个物理父图内部汇总，再等权汇总父图；同时给出全pair AUC、parent macro AUC以及四种方法相对base的配对差。所有图像和失败保留，不按接受与否筛出“成功组”作为总分。独立父图为统计单位，不把多个视图当独立样本。

AUC积分方式参照 [SuperGlue 官方 error_auc 实现](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py)，本实验用的是单应角点像素误差，不是其相对位姿角误差，也未运行 SuperGlue。感知指标来源：[LPIPS 作者代码](https://github.com/richzhang/PerceptualSimilarity)、[DISTS 论文](https://arxiv.org/abs/2004.07728)。DISTS 对部分几何变化有容忍性，因此不能用 DISTS 改善替代匹配正确性。

## 4. 要交付的图与表

每次运行自动写 `evaluation.json`、`matching.json`（有pairs时）、`summary.csv`、`report.html`，以及全部样本目录。HTML按原清单顺序展示，包含失败和不动作样本。

1. `base_vs_v25.png`：LR bicubic、base、v2.5、仅平移、无谱筛选、HR、绝对差×8，共用像素坐标。
2. `high_frequency_comparison.png`：同一ROI的RGB与有符号高频；低通核固定 [1,4,6,4,1]/16，编码统一 `0.5+4*HF`，记录显示饱和比例，绝不逐图自动拉伸对比度。ROI按base高频能量在固定3×3候选网格选取，与HR结果和收益无关。
3. `edited_region/high_frequency_comparison.png`：有动作时，按位移能量最大区域另做一张共同ROI图。没有动作就明确标记，不用未修改区域证明修改后纹理保留。
4. `stages.pt`：浮点base、各对照、最后尝试与最终接受结果、速度和位移；`diagnostics.json`保留所有回溯记录。PNG只负责显示及统一8bit匹配，图像质量指标来自浮点张量。

这些文件只留服务器；本地交付文档和小型汇总，不自动下载实验图。打开运行目录中的 `report.html` 即可逐图检查。如在本机通过SSH浏览，可使用已有可信连接把服务器目录的只读HTTP服务转发到localhost；本轮没有启动公网服务。

## 5. 服务器执行命令

环境根为 `<RUNTIME_ROOT>`。代码快照及实际运行位置以 [审计报告](audit.md) 为准。先在服务器准备 `dev_sources.json`、`test_sources.json` 与 `historical_and_dev_parents.json`；前两者格式为 `[ {"parent":"物理场景身份", "hr":"/服务器/来源.png"} ]`，后一份是过去所有看过或用于选择的父图身份列表。历史排除清单的完整性需要来源审计，程序只能检查提交给它的身份交集，不能自动证明从未看过。

```bash
ROOT=${VOSR_RUNTIME_ROOT:?Set runtime root}
CODE=$ROOT/observable_transport_v25_261113/audit_261114/code/354a98b2a9d77c22
PY=$ROOT/.venv/bin/python
export PYTHONPATH="$CODE/src"
export VOSR_RUNTIME_ROOT="$ROOT"
TASK=$ROOT/observable_transport_v25_261113/runs/formal_r1

# 一次开发；输出目录须不存在。
$PY -m observable_transport_v25.prepare --sources "$TASK/dev_sources.json" \
  --output-dir "$TASK/dev_inputs" --phase dev
$PY -m observable_transport_v25.benchmark --manifest "$TASK/dev_inputs/manifest.json" \
  --output-dir "$TASK/dev_outputs" --camera synthetic-area --perceptual

# 先确认来源、冻结代码/协议与测试清单SHA，再运行一次确认。
$PY -m observable_transport_v25.prepare --sources "$TASK/test_sources.json" \
  --excluded-parents "$TASK/historical_and_dev_parents.json" \
  --output-dir "$TASK/test_inputs" --phase test
sha256sum "$TASK/test_inputs/manifest.json" "$CODE/configs/observable_transport_v25.json"
# 将上一步事先登记的两个SHA填入以下参数；不能评价后再回填为预注册。
$PY -m observable_transport_v25.benchmark --manifest "$TASK/test_inputs/manifest.json" \
  --output-dir "$TASK/test_outputs" --camera synthetic-area --perceptual \
  --expected-manifest-sha FROZEN_MANIFEST_SHA --expected-protocol-sha FROZEN_PROTOCOL_SHA
```

真实相机组清单使用 `phase`、`camera:"deployed"`、`rows` 字段；每行 `id,parent,lr,hr` 均为服务器路径，并同样声明测试父图排除列表。不要把构造的面积降采样 LR 配到真实相机后混称真实结果。

## 6. 实验核心伪代码

```text
freeze(code, protocol, seed=6, sampler=Euler25, guidance=-0.5)
assert test_parents ∩ all_historical_and_dev_parents == empty
freeze(source_hashes, manifests, homographies_before_SR)

for parent in manifest_order:
    # 若是几何组：HR源 -> 固定视图 -> camera4x -> LR；H只交给评价器
    for view in parent.views:
        native = FrozenVOSR(view.LR, seed=6)      # float; 仅生成一次
        outputs = {base: native}
        for method in [translation_only, smooth_unfiltered, observable_transport]:
            a = zeros(method.field_dimension)
            repeat at most 3 iterations:
                warped = warp(native, exp(-B*a)) # 始终采样同一native
                r = bounded_positive_RGB_profile(camera(warped), view.LR)
                J = central_difference_of_refitted_residual(a) / sqrt(N)
                U,s,V = SVD(J)
                keep = fixed_spectral_rule(method, s)
                if no retained direction: break
                delta = damped_step(U,s,V,r,keep); cap_step_motion(delta)
                try rates [1, .5, .25, .125]:
                    accept first residual descent with motion/Jacobian/inverse checks
                if no accepted proposal: break
            outputs[method] = accepted_warp_or_exact_native
        evaluate_same_float_range(outputs, HR, PSNR, LPIPS, DISTS, HF_metrics)
        save_all_outputs_and_fixed_gain_panels_on_server()
    for pair in parent.pairs:
        for method in all_four_methods:
            H_hat = fixed_SIFT_RANSAC(quantize(outputs[view0]), quantize(outputs[view1]))
            error = corner_error(H_hat, H_gt) if valid else infinity
            retain_pair_in_denominator(error)
aggregate_by_physical_parent(); report_all_controls_and_paired_differences()
write_summary_CSV_and_HTML_with_every_case()
```

## 7. 决策规则

开发阶段若全不动作、只降低LR残差而恶化匹配、或无谱筛选已经解释全部收益，就先记录未支持主张的证据。可以设计后续新假设，但不把 R1 直接称为已有效版本。确认阶段主表必须同时展示 AUC@3/5/10、失败率和 LPIPS/DISTS 变化；不得拿像素改善代替主任务成功。若结果不一致，报告权衡，不发明加权总分。真实三维位姿、物体连接和缺失结构恢复均不在本协议已验证范围。
