# Social-ITP：多方在线引导实验与评测代码说明

本仓库用于支持 Social-ITP 项目的实验推进。项目目标是围绕 Reddit 多方讨论场景，构建一个能够在局部对话中进行在线引导、并对引导效果进行自动化评测的实验框架。目前代码主要包括三部分：

1. **Social-ITP 主实验框架**：面向多方在线引导的策略、世界模型、模拟与结果统计框架；
2. **立场分类器**：用于判断评论对目标对象的 `favor / against / none` 立场；
3. **情感分类器**：用于判断评论的 `negative / neutral / positive` 情绪极性。

其中，立场分类器和情感分类器是后续评测体系的重要组成部分：立场分类器用于度量引导前后目标用户或旁观者的立场变化，情感分类器用于度量讨论情绪是否趋于缓和或恶化。

---

## 1. 推荐目录结构

建议仓库根目录保持如下结构：

```text
social_itp_stance_full/
├── social_itp/
│   ├── classifiers/
│   │   ├── stance/
│   │   └── emotion/
│   ├── data/
│   ├── envs/
│   ├── models/
│   ├── policies/
│   ├── simulation/
│   └── evaluation/
├── scripts/
├── configs/
├── data/
│   └── MCSD/
│       ├── biden/
│       ├── Bitcoin/
│       ├── BMW/
│       ├── costco/
│       ├── tesla/
│       └── trump/
├── outputs/
├── requirements.txt
└── README.md
```

其中：

```text
data/MCSD/
```

存放原始 Reddit 多话题数据；

```text
outputs/
```

存放所有构建后的数据集、模型权重、评测结果和中间输出。

建议不要把 `outputs/` 中的大模型权重、embedding cache、API 返回缓存提交到 Git。可以在 `.gitignore` 中加入：

```gitignore
outputs/**/model/
outputs/**/checkpoint-*/
outputs/**/embedding_cache/
outputs/**/qwen_raw_predictions.jsonl
*.pt
*.bin
*.safetensors
```

---

## 2. 环境安装

建议使用干净的 conda 环境：

```powershell
conda create -n social_itp python=3.10 -y
conda activate social_itp
pip install -r .\requirements_all.txt
```

如果你只训练情感分类器，可使用情感分类器代码包中的：

```powershell
pip install -r .\requirements_emotion_goemotions.txt
```

如果在 Windows 下遇到 OpenMP 冲突：

```text
OMP: Error #15: Initializing libiomp5md.dll...
```

可临时设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

如果使用 Hugging Face 模型时遇到 `torch.load` 安全限制，建议安装较新版本的 PyTorch，并优先使用 `safetensors` 权重。本项目的 transformer 训练脚本默认使用 `use_safetensors=True`。

---

## 3. Social-ITP 主实验代码概览

Social-ITP 主实验部分用于模拟多方在线讨论中的引导策略，并对策略效果进行评测。当前项目中的主实验代码大致可以按以下模块理解。

### 3.1 数据与环境模块

建议将原始数据放在：

```text
data/MCSD/
```

其中六个主题为：

```text
biden, Bitcoin, BMW, costco, tesla, trump
```

每个主题目录下包含 Reddit 讨论树 JSON 文件。典型对话文件中包含：

```text
post_id / title / text / comments
```

每条 comment 中通常包含：

```text
id
parent_id
text
created_time
user
anno_label.stance
```

目前 stance classifier 主要使用其中的 `anno_label.stance` 字段作为监督标签。

### 3.2 策略模块

Social-ITP 主实验中，可以按论文结构区分若干策略：

```text
Random      随机选择引导动作
Reactive    基于当前观察进行即时反应
LookAhead   利用世界模型预测后续局部讨论展开，再选择动作
```

建议在代码中将策略实现放在：

```text
social_itp/policies/
```

每个策略应遵循统一接口，例如：

```python
policy.act(observation) -> action
```

其中 `observation` 表示当前局部讨论状态，`action` 表示系统将要插入或选择的引导回复。

### 3.3 世界模型模块

世界模型用于估计：

```text
当前 observation + action 后，局部讨论将如何继续展开
```

建议放在：

```text
social_itp/models/
```

或：

```text
social_itp/world_model/
```

论文中可以将其形式化为：

```text
M_phi: p_phi(y_t | o_t, a_t)
```

其中：

```text
o_t: 当前局部讨论观察
a_t: 引导动作
y_t: 后续讨论展开结果
```

当前实现中，复杂模型可以先保留 dummy / rule-based 版本，保证主实验流程先跑通。

### 3.4 模拟与评测模块

建议将模拟运行脚本放在：

```text
scripts/run_*.py
```

将评测逻辑放在：

```text
social_itp/evaluation/
```

后续评测可以结合：

```text
stance classifier：立场变化
emotion classifier：情绪变化
conversation statistics：讨论规模、回复深度、用户参与度等
```

---

## 4. 立场分类器

立场分类器用于判断评论对目标对象的立场：

```text
favor
against
none
```

它是当前项目中最成熟的评测组件之一。我们已经尝试过多种路线：

```text
TF-IDF + Logistic Regression
Frozen Embedding + Logistic Regression
Fine-tuned DeBERTa
Qwen zero-shot reference judge
```

目前推荐使用：

```text
microsoft/deberta-v3-base fine-tuning
```

作为主立场评测器。

---

### 4.1 构建立场分类数据集

当前推荐输入模板为：

```text
input_template = stage1
max_ancestor_depth = 1
```

也就是保留：

```text
目标对象 target entity
话题 topic
post title
父评论 parent comment
当前评论 current comment
```

构建命令：

```powershell
python .\scripts\build_stance_dataset.py `
  --data_root .\data\MCSD `
  --topics biden,Bitcoin,BMW,costco,tesla,trump `
  --out_dir .\outputs\stance_dataset_stage1_depth1 `
  --seed 42 `
  --split_by_topic `
  --input_template stage1 `
  --max_ancestor_depth 1
```

输出：

```text
outputs/stance_dataset_stage1_depth1/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
├── dataset_summary.json
└── scan_report.json
```

每条样本主要字段包括：

```text
row_id
thread_id
topic
target_entity
post_title
parent_text
comment_text
input_text
stance
```

---

### 4.2 训练 DeBERTa 三分类立场分类器

基础训练命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

python .\scripts\train_transformer_stance_classifier.py `
  --task three_class `
  --dataset_dir .\outputs\stance_dataset_stage1_depth1 `
  --out_dir .\outputs\stance_classifier\deberta_v3_base_three_class `
  --model_name microsoft/deberta-v3-base `
  --max_length 384 `
  --learning_rate 2e-5 `
  --num_train_epochs 3 `
  --per_device_train_batch_size 8 `
  --per_device_eval_batch_size 16 `
  --fp16
```

如果显存不足：

```powershell
--per_device_train_batch_size 4 `
--per_device_eval_batch_size 8 `
--gradient_accumulation_steps 2
```

如果 fp16 报错，则去掉：

```powershell
--fp16
```

---

### 4.3 训练集扩增与保守解码

由于各 topic 内的三类样本可能不平衡，可以只对训练集做少数类扩增：

```powershell
python .\scripts\train_transformer_stance_classifier.py `
  --task three_class `
  --dataset_dir .\outputs\stance_dataset_stage1_depth1 `
  --out_dir .\outputs\stance_classifier\deberta_v3_base_three_class_aug_cons `
  --model_name microsoft/deberta-v3-base `
  --max_length 384 `
  --learning_rate 2e-5 `
  --num_train_epochs 3 `
  --per_device_train_batch_size 8 `
  --per_device_eval_batch_size 16 `
  --fp16 `
  --augment_train `
  --augment_scope topic `
  --augment_target max `
  --augment_methods duplicate,prefix `
  --augment_prefixes "oh,|well,|actually,|literally,|to be fair,|honestly," `
  --augment_max_multiplier 3.0 `
  --stance_confidence_threshold 0.60
```

其中：

```text
--augment_train
```

表示只扩增 train split，不扩增 dev/test。

```text
--augment_scope topic
```

表示在每个 topic 内部平衡类别。

```text
--stance_confidence_threshold 0.60
```

表示如果模型原始预测为 `favor/against`，但置信度低于 0.60，则最终输出回退为 `none`。

保守解码规则：

```text
raw prediction = favor/against 且 confidence < threshold
    => final prediction = none

raw prediction = none 且 confidence < threshold
    => 仍保持 none，只标记 low_confidence_none
```

这个策略用于降低 `none -> stance` 的危险误判。

---

### 4.4 评估已有立场模型

如果已经训练好模型，只想测试不同置信度阈值：

```powershell
python .\scripts\evaluate_transformer_stance_classifier.py `
  --task three_class `
  --dataset_jsonl .\outputs\stance_dataset_stage1_depth1\test.jsonl `
  --model_dir .\outputs\stance_classifier\deberta_v3_base_three_class\model\final `
  --out_dir .\outputs\stance_classifier\deberta_v3_base_three_class_eval_thr060 `
  --split_name test `
  --max_length 384 `
  --stance_confidence_threshold 0.60
```

建议尝试：

```text
0.55, 0.60, 0.65, 0.70
```

重点比较：

```text
macro-F1
none recall
none -> stance error
favor / against recall
```

---

### 4.5 Qwen 对比实验

为了证明监督微调分类器的必要性，可以将本地 DeBERTa 与 Qwen zero-shot 进行对比。

设置 API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的实际 API Key"
```

运行：

```powershell
python .\scripts\compare_qwen_and_transformer_classifier.py `
  --dataset_jsonl .\outputs\stance_dataset_stage1_depth1\test.jsonl `
  --local_model_dir .\outputs\stance_classifier\deberta_v3_base_three_class\model\final `
  --out_dir .\outputs\stance_classifier\qwen_comparison_deberta_sample180 `
  --n_per_topic_label 10 `
  --seed 42 `
  --qwen_model qwen-plus `
  --qwen_base_url https://dashscope.aliyuncs.com/compatible-mode/v1
```

该实验会在 test 中按 topic × stance 分层抽样，输出：

```text
comparison_summary.json
comparison_predictions.csv
local_metrics.json
qwen_metrics.json
```

目前实验结论是：在 180 条分层样本上，微调 DeBERTa 更贴近数据集标注规则，而 Qwen zero-shot 更保守，倾向于将隐性 stance 判为 none。

---

## 5. 情感分类器

情感分类器用于判断评论本身的情绪极性：

```text
negative
neutral
positive
```

它不同于 stance classifier：

```text
stance: 评论对目标对象的立场
emotion: 评论本身表达出的情绪 valence
```

当前情感分类器使用 GoEmotions 数据集映射为三类进行训练。

---

### 5.1 GoEmotions 三分类映射

当前采用如下映射：

```text
positive:
admiration, amusement, approval, caring, desire, excitement,
gratitude, joy, love, optimism, pride, relief

negative:
anger, annoyance, confusion, disappointment, disapproval, disgust,
embarrassment, fear, grief, nervousness, remorse, sadness

neutral:
neutral, curiosity, realization, surprise
```

其中：

```text
confusion -> negative
```

这是根据当前评测体系设置的，因为 confusion 往往表示讨论中的困惑、冲突或消极反应。

---

### 5.2 构建 GoEmotions 三分类数据集

每类训练集约 5k：

```powershell
python .\scripts\build_goemotions_three_class_dataset.py `
  --out_dir .\outputs\emotion_goemotions_3class_5k `
  --train_per_class 5000 `
  --dev_per_class 800 `
  --test_per_class 800 `
  --multi_label_policy skip_conflict `
  --seed 42 `
  --allow_less
```

输出：

```text
outputs/emotion_goemotions_3class_5k/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── dataset_summary.json
```

其中 `skip_conflict` 表示：如果一个 GoEmotions 样本同时映射到多个粗类，例如 positive 和 negative，则跳过，降低训练噪声。

---

### 5.3 训练情感分类器

推荐使用 DeBERTa：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

python .\scripts\train_transformer_emotion_classifier.py `
  --dataset_dir .\outputs\emotion_goemotions_3class_5k `
  --out_dir .\outputs\emotion_classifier\deberta_v3_base_goemotions_3class `
  --model_name microsoft/deberta-v3-base `
  --max_length 256 `
  --learning_rate 2e-5 `
  --num_train_epochs 3 `
  --per_device_train_batch_size 8 `
  --per_device_eval_batch_size 16 `
  --fp16
```

显存不足：

```powershell
--per_device_train_batch_size 4 `
--per_device_eval_batch_size 8 `
--gradient_accumulation_steps 2
```

想先快速跑通流程，可改用：

```powershell
--model_name distilroberta-base
```

---

### 5.4 情感分类器保守解码

情感分类器也支持低置信度回退：

```powershell
--emotion_confidence_threshold 0.60
```

规则：

```text
raw prediction = negative/positive 且 confidence < threshold
    => final emotion = neutral

raw prediction = neutral 且 confidence < threshold
    => 仍保持 neutral，只标记 low_confidence_neutral
```

原因是，在评测中错误制造明显负面情绪变化比漏掉弱情绪更危险。

---

### 5.5 评估已有情感模型

不重新训练，只评估不同阈值：

```powershell
python .\scripts\evaluate_transformer_emotion_classifier.py `
  --dataset_jsonl .\outputs\emotion_goemotions_3class_5k\test.jsonl `
  --model_dir .\outputs\emotion_classifier\deberta_v3_base_goemotions_3class\model\final `
  --out_dir .\outputs\emotion_classifier\deberta_v3_base_goemotions_eval_thr060 `
  --split_name test `
  --max_length 256 `
  --emotion_confidence_threshold 0.60
```

建议尝试：

```text
0.55, 0.60, 0.65, 0.70
```

重点看：

```text
negative precision
negative recall
neutral recall
macro-F1
```

---

### 5.6 当前情感分类器结果记录

当前扩大训练集和轮数后的 GoEmotions 三分类结果约为：

```text
accuracy   ≈ 0.752
macro-F1   ≈ 0.748
weighted-F1 ≈ 0.748
```

其中：

```text
positive F1 ≈ 0.827
negative F1 ≈ 0.762
neutral F1  ≈ 0.654
```

说明模型已经能够较稳定地区分明显正负情绪，但 neutral 与弱情绪之间仍存在混淆。因此后续接入评测体系时，建议启用 conservative decoding。

---

## 6. 如何在最终评测中使用两个分类器

### 6.1 立场评测

对每条评论得到：

```text
P(favor), P(against), P(none)
```

可统计：

```text
favor_ratio
against_ratio
none_ratio
stance_shift = after_favor_prob - before_favor_prob
polarization = max(P(favor), P(against)) - P(none)
```

具体指标应结合目标用户、旁观者、整体讨论线程分别计算。

### 6.2 情绪评测

对每条评论得到：

```text
P(negative), P(neutral), P(positive)
```

可统计：

```text
mean_negative_prob
negative_ratio = proportion(P(negative) > threshold)
mean_positive_prob
emotion_shift = after_negative_prob - before_negative_prob
```

对于动态引导系统，更建议使用概率均值和比例指标，而不是只用单条硬标签。

---

## 7. 结果文件管理

建议保留：

```text
outputs/stance_dataset_stage1_depth1/
outputs/stance_classifier/deberta_v3_base_three_class*/reports/
outputs/stance_classifier/qwen_comparison_deberta_sample180/
outputs/emotion_goemotions_3class_5k/
outputs/emotion_classifier/deberta_v3_base_goemotions_3class*/reports/
```

可以归档旧实验：

```powershell
mkdir .\outputs\archive

Move-Item .\outputs\stance_classifier\context_char .\outputs\archive\ -ErrorAction SilentlyContinue
Move-Item .\outputs\stance_classifier\context_word_char .\outputs\archive\ -ErrorAction SilentlyContinue
Move-Item .\outputs\stance_classifier\embedding_minilm_* .\outputs\archive\ -ErrorAction SilentlyContinue
Move-Item .\outputs\stance_classifier\two_stage_* .\outputs\archive\ -ErrorAction SilentlyContinue
```

不要删除：

```text
data/MCSD/
outputs/stance_dataset_stage1_depth1/
outputs/emotion_goemotions_3class_5k/
最终要写进论文的 reports/*.json 和 *.csv
```

---

## 8. 常见问题

### 8.1 OpenMP DLL 冲突

报错：

```text
OMP: Error #15: Initializing libiomp5md.dll...
```

临时解决：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

长期建议：使用干净 conda 环境。

### 8.2 `torch.load` 安全限制

如果报错要求 `torch>=2.6`，建议：

```powershell
pip install --upgrade torch transformers safetensors
```

本项目脚本默认优先使用 safetensors。

### 8.3 `Trainer.__init__()` 不接受 `tokenizer`

不同 transformers 版本中，`Trainer` 的接口可能从 `tokenizer` 改为 `processing_class`。当前脚本已做兼容处理。

### 8.4 fp16 报错

如果遇到：

```text
ValueError: Attempting to unscale FP16 gradients.
```

当前脚本会尽量保持模型参数为 FP32，让 Trainer 使用 AMP。若仍失败，可去掉：

```powershell
--fp16
```

---

## 9. 推荐实验顺序

当前阶段建议按以下顺序推进：

```text
1. 固定 stance classifier：DeBERTa 三分类 + 可选 conservative decoding
2. 固定 emotion classifier：GoEmotions 三分类 + conservative decoding
3. 在 MCSD 上抽样，检查 emotion classifier 的领域迁移效果
4. 将 stance/emotion classifier 接入 Random / Reactive / LookAhead 评测
5. 记录各策略在 stance shift 与 negative emotion shift 上的差异
```

如果时间紧，优先保证：

```text
立场分类器：可用、可复现、指标已报告
情感分类器：GoEmotions 上可用，MCSD 小样本验证
主实验：至少能输出几组策略对比结果
```
