# 固定评价协议与关键指标

协议：V25-OBSERVABLE-TRANSPORT-20261113-R0。日期：2026-11-13。**协议已实现部分入口，但正式服务器评价尚未执行。** 配置初值未经真实相机校准，不称为已验证阈值。

## 1. 要回答的问题

主问题是超分是否改善图像对的几何对应；副问题是材质/感知代价；机制检查只解释它为什么动或不动。人工结构标注、人工盲评、多生成种子和大规模搜索均不作为进入研究的前提。

| 层级 | 指标与方向 | 来源/用途 | 实现状态 |
| --- | --- | --- | --- |
| 下游主指标 | 单应角点误差 AUC@3/5/10 HR px ↑ | 采用匹配文献的累计误差面积口径；同样阈值仅比较同输出尺度 | matching.py 已实现 |
| 主指标分母 | 单应估计失败率 ↓ | 无描述子/不足4匹配/无解均留在分母 | 已实现 |
| 三维外部确认 | 相对位姿误差 AUC@5°/10°/20° ↑ | [LightGlue](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)；角度单位不可混用像素 | 方案，未实现位姿求解 |
| 感知与材质 | LPIPS ↓、DISTS ↓ | [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)、[DISTS](https://arxiv.org/abs/2004.07728) | pyiqa 可选接口，未运行权重 |
| 像素保真 | RGB PSNR ↑ | [0,1]、全图 RGB、无裁边；不混称 Y-PSNR | 已实现 |
| 成本 | 几何层延迟、生成延迟、显存 | 当前 evaluate 只计几何层，native 缓存耗时为0不能解读成生成免费 | 几何计时已实现；GPU显存尚待测量 |
| 数值机制 | 命令终点误差、零命令漂移、离散 Jacobian、逆一致性 | 配准/光流中常见量；不代替最终任务收益 | 单测及诊断已实现 |
| 覆盖 | 接受率、可辨识秩、有效 LR 面积 | 防止全拒/裁边隐藏失败 | 已记录，非正确概率 |

不默认采用 NIQE/MUSIQ/CLIP 分数作为主指标：无参考好看不等于坐标正确。Hallucination Score 作为文献背景，不要求额外 MLLM 评审服务。

## 2. 固定对照

所有方法复用同一个原生输出，生成 seed=6。默认余弦基 3×3、三轮更新、固定四档线搜索；开发完成后以配置散列冻结。

| 名称 | 设置 | 要排除的解释 |
| --- | --- | --- |
| native | 原始冻结生成 | 只看是否比原生好 |
| translation_only | modes=1，保留同样谱筛选与相机 | 全局配准已足够 |
| smooth_unfiltered | 相同18维场，去观测分辨率截断，保留数值秩截断 | 只是普通平滑配准产生收益 |
| observable_transport | 完整默认求解 | 方向可辨识筛选有实际增量 |

这些是结构消融，不是搜索多个候选然后挑最高分。使用一套固定相机；区域、尺度、采样和输入压缩一致。v1.5 若加入，作为额外旧发布对照，不用其历史数值填当前表。

## 3. 无人工标注且不绑定数据集的任务

从服务器已有、来源清楚的 HR 图像构造两视图：原图与固定已知投影变换后的图。每视图通过同一相机退化，再各自生成固定种子的 native。变换矩阵自动提供评测真值，不进入 SR 推理。变换后需固定有效重叠区域，避免黑色填边成为匹配捷径；要先变换 HR 再退化，不能对已超分图变换后宣称恢复正确。

开发父图和确认父图互不重复，所有裁块/变换属于同一个 parent。确认规模按服务器已有资产事先固定，不能看结果后反复追加好样本；不要求特定数据集或多种子。单应只是平面对应能力测试，不能据此证明自然三维位姿或结构连接恢复。

真实多视图已有可靠标定时再检验位姿。没有位姿时可报告匹配数/内点率作为探索，但不能把内点率当几何真值，因为错误重复纹理也能形成自洽匹配。

## 4. 输入清单与运行

单图评价清单是 JSON 数组，路径相对清单。native 必须来自相同 seed 与冻结资产；本入口不从 PNG 自动验证生成身份，应在服务器保留生成诊断与 manifest 来源记录。使用 PNG 的量化路径是单独口径；不能与 infer 的浮点直出混表。

```json
[
  {"id":"scene01_view0","parent":"scene01","lr":"images/lr0.png","native":"images/native0.png","hr":"images/hr0.png"},
  {"id":"scene01_view1","parent":"scene01","lr":"images/lr1.png","native":"images/native1.png","hr":"images/hr1.png"}
]
```

hr 可省略。缺参考时 RGB_PSNR/LPIPS/DISTS 为 null，并报告可用父数；不伪造无参考得分。每个方法输出浮点指标和量化匹配图；文件路径保存在 `evaluation.json`。PSNR 使用未额外截断的求解结果，IQA 使用统一 [0,1] 截断。

匹配清单中的 H 是从 view0 到 view1 的 **HR 像素坐标** 单应矩阵。下面的单位阵只示范格式，正式测试必须填实际构造关系，不能全部用恒等关系制造容易结果。

```json
[
  {
    "id":"scene01_pair","parent":"scene01",
    "H_0_to_1":[[1,0,0],[0,1,0],[0,0,1]],
    "methods":{
      "native":["paired/000000/native.png","paired/000001/native.png"],
      "translation_only":["paired/000000/translation_only.png","paired/000001/translation_only.png"],
      "smooth_unfiltered":["paired/000000/smooth_unfiltered.png","paired/000001/smooth_unfiltered.png"],
      "observable_transport":["paired/000000/observable_transport.png","paired/000001/observable_transport.png"]
    }
  }
]
```

固定 SIFT 2048 特征、ratio=.75、RANSAC=3 HR px/2000次/.995、随机种子6。它是免训练、免下载权重的几何基线，不能写成已经运行 LightGlue。后续冻结 LightGlue 作为匹配器迁移检查可提高结论外推性。

角点误差为参考单应与估计单应在原图四角投影之间的平均欧氏距离。失败按无穷误差进入累计分布，JSON 逐对记录用 null 加 failed=true 表达。报告 pair AUC 与 parent macro AUC；后者先在每个父图内计算再等权汇总，不把多个变换当独立图。

## 5. 如何形成结论

报告所有确认父图、每方法配对差异、失败数与感知代价，不合成一个任意加权总分。固定一次开发、一次确认；发现实现/测量错误可修复并建立新协议身份，但不能悄悄更改成功口径。

无谱筛选对照同样好，说明核心假设未获支持；仅平移同样好，说明高维场没有必要；低残差而匹配变坏，是错误解释 LR 的直接负证据。单种子结论明确只覆盖该固定生成轨迹，不用多种子扩充样本量，也不声称种子鲁棒性。

服务器预定位置：`/home/zhangxu/hx/HX001/observable_transport_v25_261113/runs`。这只是拟定路径，当前尚未创建或执行该批服务器实验。
