#!/bin/bash
# 모델 비교 학습 실행 스크립트

cd /data2/jeOn9/fsdp_practices

# FSDP 실행
accelerate launch --config_file accelerate_config.yaml model_comparison_train.py

