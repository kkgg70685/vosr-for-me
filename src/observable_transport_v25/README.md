# VOSR v2.5：可观测连续输运原型

日期：2026-11-13。新分支从 main `5b116bd` 建立；v1.5 的 50 个发布源文件逐字节保留。v2 `98dbab6` 仅用于理解运行接口与失败原因，不依赖其结构头或速度适配器权重。

**状态（2026-11-14）：20 项测试通过；冻结 VOSR、真实相机、LPIPS/DISTS 的 CUDA 冒烟已完成。新封装与原始 native 逐值一致。两张自然图不动作，六张构造视图图像质量改善但匹配 AUC 略降；没有独立确认集上的有效性结论。** 见 [审计与实测](../../docs/v25/audit.md)。

## 如何工作

1. 固定 seed=6，用冻结 VOSR 的原生 25 步轨迹生成 native，一次解码。
2. 用连续全图余弦基表示平滑二维速度场，默认 18 个系数，保留亚像素量。
3. 将 native 变形后送入同一个冻结相机，与 LR 比较；颜色增益/偏置仅作为观测中的干扰参数消去。
4. 对连续坐标系数做有限差分，SVD 选择在指定观测误差尺度上可分辨的局部方向。
5. 固定三轮 Gauss–Newton 更新，配合四级回溯与离散形变检查。无可用方向或无下降则保持当前结果。
6. 返回 native、最后一次尝试 revised、已接受 final、连续场及过程记录。revised 可能是被拒尝试，不一定等于 final。

模型是一条完整的 **冻结生成器 + 确定性几何求解层** 路径，不再训练潜空间速度残差。全图连续场消除了矩形剪贴接口，但不保证每条结构刚性运动；平滑弯曲仍可能发生，需在下游任务验证。连续输运不是最优输运/Wasserstein 算法。

## 代码地图

| 文件 | 职责 |
| --- | --- |
| geometry.py | HR 坐标、缩放平方积分、采样、离散 Jacobian、逆一致性 |
| model.py | 颜色参数剔除、方向灵敏度、谱截断、信赖步长和回溯 |
| runtime.py | 原生 VOSR API、相机资产散列核验 |
| infer.py | LR 单图端到端入口；可用预生成 native 调试几何层 |
| benchmark.py / prepare.py / artifacts.py | 真实生成、已知单应构造、浮点指标与整图/高频/HTML报告 |
| evaluate.py | 四方法配对、PSNR、可选 LPIPS/DISTS、父级汇总、输出图路径 |
| matching.py | 固定 SIFT/RANSAC 下的单应估计、角点 AUC、失败率 |
| metrics.py | 明确的单位、分母与失败处理 |

研究设计、数学和文献见 [设计说明](../../docs/v25/design.md)，协议见 [评价协议](../../docs/v25/evaluation.md)，图见 [架构图](../../docs/architecture_v25/README.md)。

## 服务器运行

沿用现有服务器环境；新源码与旧运行资产分开。所有输入、输出、权重和完整日志仅留服务器。

```bash
cd <RUNTIME_ROOT>/observable_transport_v25_261113/code
export PYTHONPATH="$PWD/src"
PY=<RUNTIME_ROOT>/.venv/bin/python
$PY -m observable_transport_v25.infer \
  --input /absolute/server/path/lr.png \
  --output-dir <RUNTIME_ROOT>/observable_transport_v25_261113/runs/one_case \
  --runtime-root <RUNTIME_ROOT> --device cuda
```

默认真实相机须匹配保留协议 SHA；缺失即报错。`--camera synthetic-area` 是显式的面积降采样实验相机，不能拿它代替实拍相机而称为真实部署。`--native /path/native.png --device cpu` 可调试几何层，PNG 量化路径必须与浮点原生路径区分。

输出目录必须是新目录；包含浮点 `stages.pt`、三路 PNG 和诊断 JSON。默认相机作用域/边界需在服务器核验：当前固定排除 3 LR 像素边缘，不保证覆盖任意相机的感受野。

```bash
$PY -m observable_transport_v25.evaluate \
  --manifest /absolute/server/path/manifest.json --device cuda --perceptual \
  --output-dir <RUNTIME_ROOT>/observable_transport_v25_261113/runs/paired
$PY -m observable_transport_v25.matching \
  --manifest /absolute/server/path/pairs.json \
  --output-dir <RUNTIME_ROOT>/observable_transport_v25_261113/runs/matching
```

评价脚本基于已缓存的 native，固定比较原生、仅平移、无谱筛选平滑配准和完整模型；默认参数不做网格搜索。所有方法输出图和相对路径写入评价目录，供配对匹配清单引用。LPIPS/DISTS 需服务器已有 pyiqa 与权重；未请求时不计算，不用替代指标冒充。

## 验证与限制

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python tools/verify_source.py
python tools/verify_v25.py
```

Windows 本次使用 `本地 PyTorch 环境的 Python`，PyTorch `2.10.0.dev20251102+cu128` 的 CPU 路径；设置 `MKL_THREADING_LAYER=SEQUENTIAL` 解决既有环境 OpenMP 冲突，不使用重复 OpenMP 运行库强行忽略选项。测试仅在内存构造数组，没有保存实验图像或数据。

SVD 方向是当前 native、相机、形变基和颜色模型下的局部可辨识性，不是 HR 真值概率。亮度可由形变错误拟合、相机失配、重复纹理多解、颜色拟合退化、插值衰减、跨物体运动耦合仍可能导致失败。默认阈值是尚未经实拍校准的初值。离散 Jacobian 正值不证明全局单射；本版本不增删结构、不修复连接/遮挡/缺失构件，不宣称拓扑恢复。

## 正式评测入口

正式运行使用 `python -m observable_transport_v25.benchmark`，现场生成一次浮点 native 并复用给四方法。`evaluate` 保留为缓存数据兼容入口，不能自动验证已有 PNG 的生成身份。数据安排、完整命令、指标与核心伪代码见 [正式实验方案](../../docs/v25/formal_experiment.md)。R1 将颜色增益限定为 [0.5,2]；默认谱门槛保持不变。
