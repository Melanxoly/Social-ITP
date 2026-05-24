# social_itp stance classifier transfer package

这个包是可迁移的代码包，包含当前 stance classifier 实验所需的 `social_itp/`、`scripts/`、`configs/` 和依赖说明。它不包含你的原始数据和已训练模型；请把这些目录单独复制到新电脑。

## 1. 推荐目录结构

在新 Windows 台式机上建议保持：

```text
social_itp_project/
├── social_itp/
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
└── requirements.txt
```

你需要额外复制：

```text
data/MCSD/
outputs/stance_classifier/deberta_v3_base_three_class/model/final/   # 如果要直接评估已训练模型
```

如果准备在新机器上重新训练，则只需要复制 `data/MCSD/`。

## 2. 安装环境

建议新建干净环境：

```powershell
conda create -n social_itp python=3.10 -y
conda activate social_itp
pip install -r .\requirements_all.txt
```

如果你的 GPU/CUDA 环境有特殊要求，可以先按 PyTorch 官网方式安装对应 CUDA 版 torch，再执行：

```powershell
pip install -r .\requirements_all.txt
```

Windows 下如果遇到 OpenMP DLL 冲突，可以临时设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 3. 构建当前推荐数据集

当前推荐输入是简化版 Stage 1 模板，保留 target、topic、post title、父评论与当前评论：

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

## 4. 训练 DeBERTa 三分类：带少数类扩增 + 保守解码

这一步实现两个新功能：

1. 训练集少数类扩增：按 topic 内部把少数标签 oversample 到相对平衡；
2. conservative decoding：如果模型预测 favor/against 但置信度低于阈值，则最终输出 none。

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

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

如果显存不足：

```powershell
--per_device_train_batch_size 4 `
--per_device_eval_batch_size 8 `
--gradient_accumulation_steps 2
```

如果 fp16 仍然出问题，去掉：

```powershell
--fp16
```

## 5. 如何理解 conservative decoding

默认规则：

```text
raw prediction = favor/against 且 confidence < threshold  => final prediction = none
raw prediction = none 且 confidence < threshold           => 仍保持 none，只在 predictions.csv 标记 low_confidence_none
```

原因：你认为“把 none 判断成 stance”比“把 stance 判断成 none”更危险，因此低置信度 stance 统一回退到 none。对于本来预测为 none 且低置信度的样本，不反向改成立场类，因为那会增加误报风险；它们只适合作人工/Qwen 复核。

输出文件中会同时保留 raw 与 final：

```text
reports/test_raw_metrics.json              # 阈值前原始预测
reports/test_metrics.json                  # 阈值后最终预测
reports/test_raw_predictions.csv
reports/test_predictions.csv
```

## 6. 不重新训练，直接对已有模型测试不同置信度阈值

如果你已经有一个训练好的模型，可以只评估不同阈值：

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

你可以把 `0.55, 0.60, 0.65, 0.70` 分别评估，选择在 `none_to_stance` 和 macro-F1 之间更合适的阈值。

## 7. Qwen 对比

设置 API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的实际API_KEY"
```

运行 180 条左右分层样本对比：

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

## 8. 输出管理建议

建议保留：

```text
outputs/stance_dataset_stage1_depth1/
outputs/stance_classifier/deberta_v3_base_three_class*/reports/
outputs/stance_classifier/qwen_comparison_deberta_sample180/
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
最终要写进论文的 reports/*.json 和 *.csv
```
