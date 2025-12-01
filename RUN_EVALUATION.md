# 평가 스크립트 실행 방법

## 방법 1: 직접 실행 (가장 간단)

```bash
cd /data2/jeOn9/fsdp_practices
python evaluate_models.py
```

## 방법 2: 실행 스크립트 사용

```bash
cd /data2/jeOn9/fsdp_practices
bash run_evaluation.sh
```

또는 실행 권한 부여 후:

```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

## 방법 3: 전체 프로세스 (학습 + 평가)

```bash
cd /data2/jeOn9/fsdp_practices
python run_full_comparison.py
```

또는:

```bash
bash run_full_comparison.sh
```

## 방법 4: 백그라운드 실행

```bash
cd /data2/jeOn9/fsdp_practices
nohup python evaluate_models.py > evaluation.log 2>&1 &
```

로그 확인:
```bash
tail -f evaluation.log
```

## 평가할 모델 설정

`evaluate_models.py` 파일의 `MODELS_TO_EVALUATE` 리스트에서 평가할 모델을 설정할 수 있습니다:

```python
MODELS_TO_EVALUATE = [
    {
        "name": "google/gemma-2b-it",
        "base_model": "google/gemma-2b-it",
        "checkpoint": "outputs/google_gemma_2b_it/checkpoint-50",
        "transformer_layer_cls": "GemmaDecoderLayer",
    },
    # ... 다른 모델들
]
```

## 평가 샘플 수 변경

기본값은 100개 샘플입니다. 변경하려면:

```python
# evaluate_models.py의 main() 함수에서
result = evaluate_model(model_config, num_samples=50)  # 50개로 변경
```

## 결과 확인

평가 완료 후:
- **JSON 파일**: `evaluation_results_{timestamp}.json`
- **콘솔 출력**: ROUGE-1, ROUGE-2, ROUGE-L 점수

## 필수 라이브러리

```bash
pip install rouge-score datasets transformers peft torch
```

## 주의사항

1. **체크포인트 경로**: `outputs/{model_name}/checkpoint-50` 경로에 체크포인트가 있어야 합니다
2. **Hugging Face 인증**: 일부 모델은 접근 권한이 필요할 수 있습니다
3. **GPU 메모리**: 평가는 모델을 메모리에 로드하므로 충분한 GPU 메모리가 필요합니다

