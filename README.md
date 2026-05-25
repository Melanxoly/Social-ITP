# GoEmotions → 3-class emotion classifier package

这个包用于把 GoEmotions 映射成三类情感数据集，并训练本地情感分类器：

```text
negative / neutral / positive
```

其中按你的要求：

```text
confusion -> negative
```

## 1. 数据量会不会太大？

不会。GoEmotions 总量约 58k Reddit comments，原始标签是 27 个细粒度 emotion 加 neutral。我们这里会重新映射并抽样，每类 train 只保留约 5k，因此训练集大约：

```text
5k negative + 5k neutral + 5k positive = 15k train rows
```

这比你现在 stance 数据训练 DeBERTa 的规模通常更可控。

## 2. 新电脑环境安装

```powershell
conda create -n emotion python=3.10 -y
conda activate emotion
pip install -r .\requirements_emotion_goemotions.txt
```

Windows 如遇 OpenMP 冲突，可临时设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 3. 构建三分类 GoEmotions 数据集

推荐命令：

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
  train.jsonl
  dev.jsonl
  test.jsonl
  dataset_summary.json
```

### 标签映射

positive:

```text
admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
```

negative:

```text
anger, annoyance, confusion, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
```

neutral:

```text
neutral, curiosity, realization, surprise
```

### 多标签样本怎么处理？

默认：

```text
--multi_label_policy skip_conflict
```

如果一个样本同时含 positive 和 negative 粗类标签，就跳过。这样训练集更干净。

如果你希望保留更多样本，并且把冲突样本偏向 negative，可以改成：

```powershell
--multi_label_policy priority_negative --no_prefer_single_label
```

不过第一版不建议这么做，因为会引入更多噪声。

## 4. 训练情感分类器

算力够时：

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

如果显存不足：

```powershell
python .\scripts\train_transformer_emotion_classifier.py `
  --dataset_dir .\outputs\emotion_goemotions_3class_5k `
  --out_dir .\outputs\emotion_classifier\deberta_v3_base_goemotions_3class_bs4 `
  --model_name microsoft/deberta-v3-base `
  --max_length 256 `
  --learning_rate 2e-5 `
  --num_train_epochs 3 `
  --per_device_train_batch_size 4 `
  --per_device_eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --fp16
```

如果想先快速跑通流程，可以换小模型：

```powershell
--model_name distilroberta-base
```

## 5. 低置信度正/负情绪回退 neutral

训练时或评估时可以启用 conservative decoding：

```powershell
--emotion_confidence_threshold 0.60
```

规则：

```text
raw prediction = negative/positive 且 confidence < threshold => final emotion = neutral
raw prediction = neutral 且 confidence < threshold           => 仍保持 neutral，仅标记 low_confidence_neutral
```

原因：在你的评测体系中，错误制造明显正/负情绪变化比漏掉弱情绪更危险。

建议先训练不带阈值的模型，然后单独评估不同阈值。

## 6. 不重新训练，评估不同阈值

```powershell
python .\scripts\evaluate_transformer_emotion_classifier.py `
  --dataset_jsonl .\outputs\emotion_goemotions_3class_5k\test.jsonl `
  --model_dir .\outputs\emotion_classifier\deberta_v3_base_goemotions_3class\model\final `
  --out_dir .\outputs\emotion_classifier\deberta_v3_base_goemotions_eval_thr060 `
  --split_name test `
  --max_length 256 `
  --emotion_confidence_threshold 0.60
```

可以依次测试：

```text
0.55, 0.60, 0.65, 0.70
```

## 7. 输出文件

训练输出：

```text
outputs/emotion_classifier/deberta_v3_base_goemotions_3class/
  train_config.json
  model/final/
  reports/
    metrics_all.json
    test_metrics.json
    test_raw_metrics.json
    test_confusion_matrix.csv
    test_raw_confusion_matrix.csv
    test_predictions.csv
```

其中：

```text
test_raw_metrics.json = 阈值前原始分类结果
test_metrics.json     = 阈值后 conservative decoding 结果
```

## 8. 后续如何接入你的评测体系

建议不要只用硬标签，而是保存概率：

```text
P(negative), P(neutral), P(positive)
```

后续可以统计：

```text
mean_negative_prob
negative_ratio = proportion(P(negative) > threshold)
mean_positive_prob
emotion_shift = after_negative_prob - before_negative_prob
```

这样比单条评论硬标签更稳定。
