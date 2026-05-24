# Makefile for social_itp project
# Usage examples:
#   make help
#   make setup
#   make build_dataset
#   make train
#   make eval
#   make compare_qwen

PY ?= python
PIP ?= pip
CONDA_ENV ?= social_itp
REQ ?= requirements_all.txt

# -------- Cross-platform helpers --------
ifeq ($(OS),Windows_NT)
define MKDIR_P
	powershell -NoProfile -Command "New-Item -ItemType Directory -Force '$(1)' | Out-Null"
endef
define RM_RF
	powershell -NoProfile -Command "if (Test-Path '$(1)') { Remove-Item -Recurse -Force '$(1)' }"
endef
else
define MKDIR_P
	mkdir -p '$(1)'
endef
define RM_RF
	rm -rf '$(1)'
endef
endif
# ----------------------------------------

DATA_ROOT ?= data/MCSD
TOPICS ?= biden,Bitcoin,BMW,costco,tesla,trump
OUT_DIR ?= outputs/stance_dataset_stage1_depth1
SEED ?= 42
INPUT_TEMPLATE ?= stage1
MAX_ANCESTOR_DEPTH ?= 1

MODEL_NAME ?= microsoft/deberta-v3-base
TRAIN_OUT_DIR ?= outputs/stance_classifier/deberta_v3_base_three_class_aug_x3_thr050
MAX_LENGTH ?= 384
LR ?= 2e-5
EPOCHS ?= 3
TRAIN_BATCH ?= 8
EVAL_BATCH ?= 16
FP16 ?= --fp16
MULTIPLIER ?= 3.0
THR ?= 0.50

.PHONY: all help setup install-requirements build_dataset train eval compare_qwen clean

help:
	@echo "Makefile targets for the social_itp project"
	@echo "  setup                 Create/prepare environment (pip install)"
	@echo "  install-requirements  Install Python requirements via pip"
	@echo "  build_dataset         Build stance dataset (uses scripts/build_stance_dataset.py)"
	@echo "  train                 Train the transformer stance classifier"
	@echo "  eval                  Evaluate a trained model (conservative decoding threshold)"
	@echo "  compare_qwen          Run Qwen vs local model comparison (uses scripts/compare_qwen_and_transformer_classifier.py)"
	@echo "  clean                 Remove common output folders"

setup: install-requirements
	@echo "If you want a conda env, create one manually: conda create -n $(CONDA_ENV) python=3.10 -y"
	@echo "Then activate it and run: make install-requirements"

install-requirements:
	@echo "Installing requirements..."

ifeq ($(wildcard requirements_all.txt),)
ifeq ($(wildcard requirements.txt),)
$(error No requirements file found (tried requirements_all.txt and requirements.txt))
else
	@$(PY) -m pip install -r requirements.txt
endif
else
	@$(PY) -m pip install -r requirements_all.txt
endif

build_dataset:
	$(PY) scripts/build_stance_dataset.py \
		--data_root $(DATA_ROOT) \
		--topics $(TOPICS) \
		--out_dir $(OUT_DIR) \
		--seed $(SEED) \
		--split_by_topic \
		--input_template $(INPUT_TEMPLATE) \
		--max_ancestor_depth $(MAX_ANCESTOR_DEPTH)

train:
	$(PY) scripts/train_transformer_stance_classifier.py \
		--task three_class \
		--dataset_dir $(OUT_DIR) \
		--out_dir $(TRAIN_OUT_DIR) \
		--model_name $(MODEL_NAME) \
		--max_length $(MAX_LENGTH) \
		--learning_rate $(LR) \
		--num_train_epochs $(EPOCHS) \
		--per_device_train_batch_size $(TRAIN_BATCH) \
		--per_device_eval_batch_size $(EVAL_BATCH) \
		$(FP16) \
		--augment_train \
		--augment_scope topic \
		--augment_target max \
		--augment_final_multiplier $(MULTIPLIER) `
		--augment_max_multiplier 0 `
		--augment_methods duplicate,prefix,casing,synonym,mixed `
		--augment_prefixes "oh,|well,|actually,|literally,|to be fair,|honestly,|tbh,|imo," \
		--augment_case_modes lower,upper,title `
		--augment_chain_min 1 `
		--augment_chain_max 2 `
		--stance_confidence_threshold $(THR)

sweep-stance-thresholds:
	$(PY) scripts/sweep_transformer_stance_thresholds.py \
		--dataset_jsonl $(OUT_DIR)/test.jsonl \
		--model_dir $(TRAIN_OUT_DIR)/model/final \
		--out_dir $(TRAIN_OUT_DIR)_threshold_sweep \
		--split_name test \
		--max_length $(MAX_LENGTH) \
		--per_device_eval_batch_size $(EVAL_BATCH) \
		--thresholds 0.0,0.5,0.55,0.6,0.65,0.7,0.75

compare_qwen:
	$(PY) scripts/compare_qwen_and_transformer_classifier.py \
		--dataset_jsonl $(OUT_DIR)/test.jsonl \
		--local_model_dir $(TRAIN_OUT_DIR)/model/final \
		--out_dir outputs/stance_classifier/qwen_comparison_deberta_sample180 \
		--n_per_topic_label 10 \
		--seed 42 \
		--qwen_model qwen-plus \
		--qwen_base_url https://dashscope.aliyuncs.com/compatible-mode/v1


clean:
	@echo "Removing common outputs (be careful)"
	@$(call RM_RF,outputs/stance_dataset_stage1_depth1)
	@$(call RM_RF,outputs/stance_classifier/*_eval_thr*)

all: install-requirements build_dataset train sweep-stance-thresholds
	@echo "Completed 'make all' (install -> build_dataset -> train -> sweep-stance-thresholds)"
