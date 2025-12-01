"""
여러 LLM 모델 성능 비교 스크립트
google/gemma-2b-it와 비슷한 크기의 모델들을 동일한 설정으로 학습하여 성능 비교
"""

import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Hugging Face 인증 처리
def get_hf_token():
    """Hugging Face 토큰 가져오기 (환경 변수 또는 로그인)"""
    # 환경 변수에서 토큰 확인
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if token:
        return token
    
    # 환경 변수가 없으면 huggingface-cli login으로 로그인했는지 확인
    try:
        from huggingface_hub import whoami
        try:
            whoami()
            print("Hugging Face에 이미 로그인되어 있습니다 (huggingface-cli login).")
            return None  # None이면 transformers가 자동으로 캐시된 토큰 사용
        except Exception:
            # 로그인되지 않음
            pass
    except ImportError:
        pass
    
    # 토큰이 없고 로그인도 안 되어 있으면 None 반환
    # transformers가 자동으로 처리하거나 에러 메시지에서 안내
    return None

# 비교할 모델 리스트 (gemma-2b-it와 비슷한 크기)
# 필요에 따라 모델을 추가/제거하거나 설정을 변경할 수 있습니다
MODELS_TO_COMPARE = [
    {
        "name": "google/gemma-2b-it",
        "transformer_layer_cls": "GemmaDecoderLayer",
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "enabled": True,  # False로 설정하면 이 모델은 스킵됩니다
    },
    {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "transformer_layer_cls": "Phi3DecoderLayer",
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "enabled": True,
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "transformer_layer_cls": "LlamaDecoderLayer",
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "enabled": True,
        # 주의: Llama 모델은 Hugging Face 접근 권한이 필요할 수 있습니다
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "transformer_layer_cls": "Qwen2DecoderLayer",
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "enabled": True,
    },
]

# Hugging Face 인증 초기화
hf_token = get_hf_token()

# 데이터셋 로드
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

def formatting_func(example):
    """프롬프트 포맷팅 함수 (gemma 형식)"""
    text = f"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{example['document']}<end_of_turn>
<start_of_turn>model
{example['summary']}<end_of_turn><eos>"""
    return text

def get_model_specific_formatting(model_name):
    """모델별 프롬프트 포맷팅 함수 반환"""
    if "gemma" in model_name.lower():
        return formatting_func
    elif "phi" in model_name.lower():
        def phi_formatting(example):
            return f"""<|user|>
다음 글을 요약해주세요:

{example['document']}<|end|>
<|assistant|>
{example['summary']}<|end|>"""
        return phi_formatting
    elif "llama" in model_name.lower():
        def llama_formatting(example):
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

다음 글을 요약해주세요:

{example['document']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['summary']}<|eot_id|>"""
        return llama_formatting
    elif "qwen" in model_name.lower():
        def qwen_formatting(example):
            return f"""<|im_start|>user
다음 글을 요약해주세요:

{example['document']}<|im_end|>
<|im_start|>assistant
{example['summary']}<|im_end|>"""
        return qwen_formatting
    else:
        return formatting_func  # 기본값

def train_model(model_config):
    """단일 모델 학습 함수"""
    import gc
    import torch
    
    model_name = model_config["name"]
    transformer_layer_cls = model_config["transformer_layer_cls"]
    target_modules = model_config["target_modules"]
    
    print(f"\n{'='*80}")
    print(f"학습 시작: {model_name}")
    print(f"{'='*80}\n")
    
    # GPU 메모리 정리 (더 철저하게)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # GPU 메모리 상태 출력
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: 할당됨 {allocated:.2f} GB, 예약됨 {reserved:.2f} GB")
    
    # 모델명에서 디렉토리명 생성 (슬래시를 언더스코어로)
    model_dir_name = model_name.replace("/", "_").replace("-", "_")
    output_dir = f"outputs/{model_dir_name}"
    
    # 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,  # Hugging Face 인증 토큰 추가
        )
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"토크나이저 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. Hugging Face 계정에 모델 접근 권한을 요청하세요:")
        print(f"   https://huggingface.co/{model_name}")
        print("2. 환경 변수에 토큰을 설정하세요:")
        print("   export HF_TOKEN='your_token_here'")
        print("3. 또는 huggingface-cli login을 실행하세요")
        return None
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    # 모델 로드
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=hf_token,  # Hugging Face 인증 토큰 추가
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. Hugging Face 계정에 모델 접근 권한을 요청하세요:")
        print(f"   https://huggingface.co/{model_name}")
        print("2. 환경 변수에 토큰을 설정하세요:")
        print("   export HF_TOKEN='your_token_here'")
        print("3. 또는 huggingface-cli login을 실행하세요")
        return None
    
    # FSDP 설정
    # 주의: accelerate_config.yaml의 fsdp_config와 함께 사용됨
    # TrainingArguments의 fsdp_config가 transformer_layer_cls_to_wrap를 모델별로 설정
    # accelerate_config.yaml의 다른 FSDP 설정들(offload, sharding strategy 등)도 함께 적용됨
    fsdp_config = {
        "transformer_layer_cls_to_wrap": [transformer_layer_cls],
        "activation_checkpointing": True,
    }
    
    # 모델별 포맷팅 함수
    formatting_func_model = get_model_specific_formatting(model_name)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=50,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        push_to_hub=False,
        report_to='none',
        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config,
        # 메모리 최적화
        dataloader_pin_memory=False,  # 메모리 절약
        remove_unused_columns=False,  # 필요한 컬럼 유지
    )
    
    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        formatting_func=formatting_func_model,
    )
    
    trainer.processing_class = tokenizer
    
    # 학습 시작 시간 기록
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        # 학습 완료 시간 기록
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60  # 분 단위
        
        # 메모리 정리
        import gc
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        # 결과 저장
        result = {
            "model_name": model_name,
            "output_dir": output_dir,
            "training_time_minutes": training_time,
            "status": "success",
            "checkpoint": f"{output_dir}/checkpoint-50",
        }
        
        print(f"\n학습 완료: {model_name}")
        print(f"학습 시간: {training_time:.2f} 분")
        print(f"체크포인트: {result['checkpoint']}\n")
        
        return result
        
    except Exception as e:
        print(f"학습 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 메모리 정리
        import gc
        try:
            del model, trainer
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
        }

def main():
    """메인 함수: 단일 모델 학습 (accelerate launch로 각 모델마다 호출됨)"""
    import os
    import sys
    
    # 환경 변수에서 모델 인덱스 가져오기
    model_idx = int(os.environ.get("MODEL_INDEX", "0"))
    
    if model_idx >= len(MODELS_TO_COMPARE):
        print(f"모델 인덱스 {model_idx}가 범위를 벗어났습니다.")
        return
    
    model_config = MODELS_TO_COMPARE[model_idx]
    
    # enabled가 False인 모델은 스킵
    if not model_config.get("enabled", True):
        print(f"\n모델 {model_config['name']}는 비활성화되어 있어 스킵합니다.")
        return
    
    print("="*80)
    print("LLM 모델 성능 비교 학습 시작")
    print("="*80)
    print(f"\n[설정 확인]")
    print(f"- accelerate_config.yaml 사용 중 (--config_file 옵션으로 로드됨)")
    print(f"- FSDP 설정: accelerate_config.yaml의 전역 설정 + TrainingArguments의 모델별 설정")
    print(f"- 모델별 transformer_layer_cls: {model_config['transformer_layer_cls']}")
    
    # Gated 모델에 대한 인증 안내
    if "llama" in model_config['name'].lower() or "meta-llama" in model_config['name'].lower():
        if hf_token is None:
            print(f"\n[인증 안내]")
            print(f"- 모델 {model_config['name']}는 gated 모델입니다.")
            print(f"- Hugging Face 계정에 접근 권한을 요청하세요: https://huggingface.co/{model_config['name']}")
            print(f"- 인증 방법:")
            print(f"  1. export HF_TOKEN='your_token_here' (환경 변수 설정)")
            print(f"  2. huggingface-cli login (CLI 로그인)")
    
    print("="*80)
    
    result = train_model(model_config)
    
    if result:
        # 결과 저장
        results_file = f"comparison_results_{model_config['name'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([result], f, indent=2, ensure_ascii=False)
        
        print(f"\n상세 결과는 {results_file} 파일에 저장되었습니다.")
        
        # 결과 요약 출력
        print("\n" + "="*80)
        print("학습 결과 요약")
        print("="*80)
        if result.get("status") == "success":
            print(f"\n모델: {result['model_name']}")
            print(f"  학습 시간: {result.get('training_time_minutes', 'N/A'):.2f} 분")
            print(f"  체크포인트: {result.get('checkpoint', 'N/A')}")
        else:
            print(f"\n모델: {result['model_name']}")
            print(f"  상태: 실패")
            print(f"  오류: {result.get('error', 'N/A')}")

if __name__ == "__main__":
    main()

