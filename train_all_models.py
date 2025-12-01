"""
모든 모델을 순차적으로 학습하는 스크립트
각 모델마다 accelerate launch를 개별적으로 실행
"""

import subprocess
import sys
import os
import json
import signal
import gc
import torch
from datetime import datetime

# model_comparison_train.py에서 모델 리스트 가져오기
MODELS_TO_COMPARE = [
    {
        "name": "google/gemma-2b-it",
        "transformer_layer_cls": "GemmaDecoderLayer",
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "enabled": True,
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
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "transformer_layer_cls": "Qwen2DecoderLayer",
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "enabled": True,
    },
]

def train_single_model(model_idx, model_config):
    """단일 모델 학습 (accelerate launch 사용)"""
    model_name = model_config["name"]
    
    print(f"\n{'='*80}")
    print(f"모델 {model_idx + 1}/{len(MODELS_TO_COMPARE)}: {model_name}")
    print(f"{'='*80}\n")
    
    # GPU 메모리 정리 (학습 전)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU 메모리 정리 완료\n")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env["MODEL_INDEX"] = str(model_idx)
    
    try:
        # Popen을 사용하여 시그널 전달 가능하도록
        process = subprocess.Popen(
            ["accelerate", "launch", "--config_file", "accelerate_config.yaml", "model_comparison_train.py"],
            cwd="/data2/jeOn9/fsdp_practices",
            env=env
        )
        
        # 시그널 핸들러 설정
        def signal_handler(sig, frame):
            print(f"\n\n{model_name} 학습 중단 요청 받음. 프로세스 종료 중...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 프로세스 완료 대기
        result = process.wait()
        
        if result != 0:
            print(f"\n{model_name} 학습 실패 (exit code: {result})")
            return False
        
        print(f"\n{model_name} 학습 완료!")
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n{model_name} 학습 중단됨")
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

def main():
    """메인 함수: 모든 모델 순차 학습"""
    print("="*80)
    print("LLM 모델 전체 비교 학습 시작")
    print("="*80)
    print("Ctrl+C를 누르면 현재 모델 학습 후 중단됩니다.\n")
    
    results = []
    
    try:
        # 각 모델 순차 학습
        for idx, model_config in enumerate(MODELS_TO_COMPARE):
            if not model_config.get("enabled", True):
                print(f"\n모델 {model_config['name']}는 비활성화되어 있어 스킵합니다.")
                continue
            
            success = train_single_model(idx, model_config)
            
            # 학습 후 GPU 메모리 정리
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if success:
                model_dir_name = model_config["name"].replace("/", "_").replace("-", "_")
                results.append({
                    "model_name": model_config["name"],
                    "output_dir": f"outputs/{model_dir_name}",
                    "checkpoint": f"outputs/{model_dir_name}/checkpoint-50",
                    "status": "success",
                })
            else:
                results.append({
                    "model_name": model_config["name"],
                    "status": "failed",
                })
                
    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다.")
        # 현재까지의 결과 저장
        if results:
            results_file = f"all_models_comparison_results_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"부분 결과는 {results_file} 파일에 저장되었습니다.")
        sys.exit(1)
    
    # 전체 결과 저장
    results_file = f"all_models_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 결과 요약
    print("\n" + "="*80)
    print("전체 학습 결과 요약")
    print("="*80)
    for result in results:
        if result.get("status") == "success":
            print(f"\n✓ {result['model_name']}")
            print(f"  체크포인트: {result.get('checkpoint', 'N/A')}")
        else:
            print(f"\n✗ {result['model_name']} - 실패")
    
    print(f"\n상세 결과는 {results_file} 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()

