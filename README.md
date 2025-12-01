# LLM 모델 성능 비교 - FSDP 파인튜닝

`google/gemma-2b-it`와 비슷한 크기의 여러 LLM 모델들을 동일한 설정으로 학습하여 성능을 비교하는 프로젝트입니다.

## 비교 모델

1. **google/gemma-2b-it** (2B)
2. **microsoft/Phi-3-mini-4k-instruct** (3.8B)
3. **meta-llama/Llama-3.2-3B-Instruct** (3B)
4. **Qwen/Qwen2.5-3B-Instruct** (3B)

## 설정

모든 모델은 동일한 설정으로 학습됩니다:

- **데이터셋**: `daekeun-ml/naver-news-summarization-ko`
- **LoRA 설정**:
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.05
- **학습 설정**:
  - Batch size: 1 (per device)
  - Gradient accumulation: 8
  - Max steps: 50
  - Learning rate: 2e-4
  - Mixed precision: bf16
- **FSDP**: FULL_SHARD with auto_wrap

## 실행 방법

### 1. 환경 설정

```bash
pip install transformers==4.56.2 accelerate==1.10.1 peft==0.17.1 trl==0.23.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets bitsandbytes rouge-score
```

### 2. Accelerate 설정 확인

`accelerate_config.yaml`에서 GPU 개수를 확인하고 필요시 수정:

```yaml
num_processes: 4  # 실제 GPU 개수에 맞게 수정
```

### 3. 전체 프로세스 실행 (권장)

학습부터 평가까지 자동으로 실행:

```bash
cd /data2/jeOn9/fsdp_practices
./run_full_comparison.sh
```

또는:

```bash
python run_full_comparison.py
```

### 4. 단계별 실행

#### 학습만 실행:
```bash
accelerate launch --config_file accelerate_config.yaml model_comparison_train.py
```

#### 평가만 실행 (학습 완료 후):
```bash
python evaluate_models.py
```

## 결과

학습 완료 후:

- 각 모델별 체크포인트: `outputs/{model_name}/checkpoint-50`
- 비교 결과 JSON: `comparison_results_{timestamp}.json`
- 결과에는 각 모델의 학습 시간, 체크포인트 경로 등이 포함됩니다.

## 성능 평가

학습된 모델들의 성능을 평가하려면:

### 1. 평가 라이브러리 설치

```bash
pip install rouge-score
pip install bert-score
```

### 2. 평가 실행

```bash
python evaluate_models.py
```

### 평가 지표

- **ROUGE-1**: 단어 단위 유사도
- **ROUGE-2**: 2-gram 유사도
- **ROUGE-L**: 가장 긴 공통 부분 수열 기반 유사도
- **BERTScore-F1**: 각 단어의 의미(vector embedding)를 비교해서 의미적 유사도를 측정하는 지표

평가 결과는 `evaluation_results_{timestamp}.json` 파일에 저장되며, 각 모델의 ROUGE 점수가 비교됩니다.

## 주의사항

- 각 모델은 모델별 프롬프트 포맷을 사용합니다 (Gemma, Phi-3, Llama-3, Qwen)
- 모델별 transformer layer class가 자동으로 설정됩니다
- 일부 모델은 Hugging Face 접근 권한이 필요할 수 있습니다