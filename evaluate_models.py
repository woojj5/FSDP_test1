"""
학습된 모델들의 성능 평가 스크립트
ROUGE 점수와 BERTScore를 사용하여 요약 성능을 비교합니다.
"""

import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score

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

# 평가할 모델 리스트 (학습된 체크포인트 경로)
MODELS_TO_EVALUATE = [
    {
        "name": "google/gemma-2b-it",
        "base_model": "google/gemma-2b-it",
        "checkpoint": "outputs/google_gemma_2b_it/checkpoint-50",
        "transformer_layer_cls": "GemmaDecoderLayer",
    },
    {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "checkpoint": "outputs/microsoft_Phi_3_mini_4k_instruct/checkpoint-50",
        "transformer_layer_cls": "Phi3DecoderLayer",
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "checkpoint": "outputs/meta_llama_Llama_3.2_3B_Instruct/checkpoint-50",
        "transformer_layer_cls": "LlamaDecoderLayer",
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "checkpoint": "outputs/Qwen_Qwen2.5_3B_Instruct/checkpoint-50",
        "transformer_layer_cls": "Qwen2DecoderLayer",
    },
]

# Hugging Face 인증 초기화
hf_token = get_hf_token()

# 테스트 데이터셋 로드
try:
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="test")
    print(f"데이터셋 로드 완료: {len(dataset)}개 샘플")
    # 첫 번째 샘플 구조 확인
    if len(dataset) > 0:
        first_sample = dataset[0]
        print(f"샘플 타입: {type(first_sample)}")
        if isinstance(first_sample, dict):
            print(f"샘플 키: {list(first_sample.keys())}")
except Exception as e:
    print(f"데이터셋 로드 실패: {e}")
    dataset = None

def get_model_specific_formatting(model_name):
    """모델별 프롬프트 포맷팅 함수 반환"""
    if "gemma" in model_name.lower():
        def gemma_formatting(document):
            return f"""<start_of_turn>user
다음 글을 요약해주세요:

{document}<end_of_turn>
<start_of_turn>model
"""
        return gemma_formatting
    elif "phi" in model_name.lower():
        def phi_formatting(document):
            return f"""<|user|>
다음 글을 요약해주세요:

{document}<|end|>
<|assistant|>
"""
        return phi_formatting
    elif "llama" in model_name.lower():
        def llama_formatting(document):
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

다음 글을 요약해주세요:

{document}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return llama_formatting
    elif "qwen" in model_name.lower():
        def qwen_formatting(document):
            return f"""<|im_start|>user
다음 글을 요약해주세요:

{document}<|im_end|>
<|im_start|>assistant
"""
        return qwen_formatting
    else:
        def default_formatting(document):
            return f"""다음 글을 요약해주세요:

{document}

요약: """
        return default_formatting

def clean_summary(summary):
    """요약 텍스트 정리"""
    # 이상한 패턴 제거
    patterns_to_remove = ["municipi :", "qed", "QED", "Q.E.D.", "<end_of_turn>", "<eos>", "<|end|>", "<|im_end|>"]
    for pattern in patterns_to_remove:
        if pattern in summary:
            summary = summary.split(pattern)[0].strip()
    
    # 끝부분 정리
    summary = summary.rstrip(" .").strip()
    return summary

def evaluate_model(model_config, num_samples=100):
    """단일 모델 평가 함수"""
    model_name = model_config["name"]
    base_model = model_config["base_model"]
    checkpoint = model_config["checkpoint"]
    
    print(f"\n{'='*80}")
    print(f"평가 시작: {model_name}")
    print(f"체크포인트: {checkpoint}")
    print(f"{'='*80}\n")
    
    # 데이터셋 확인
    if dataset is None:
        print("데이터셋이 로드되지 않았습니다.")
        return None
    
    # 데이터셋 확인
    if dataset is None:
        print("데이터셋이 로드되지 않았습니다.")
        return None
    
    # 체크포인트 존재 확인
    if not os.path.exists(checkpoint):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint}")
        return None
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=hf_token,  # Hugging Face 인증 토큰 추가
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"토크나이저 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. Hugging Face 계정에 모델 접근 권한을 요청하세요:")
        print(f"   https://huggingface.co/{base_model}")
        print("2. 환경 변수에 토큰을 설정하세요:")
        print("   export HF_TOKEN='your_token_here'")
        print("3. 또는 huggingface-cli login을 실행하세요")
        return None
    
    try:
        # 모델 로드
        print("베이스 모델 로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,  # Hugging Face 인증 토큰 추가
        )
        
        print("LoRA 어댑터 로드 중...")
        model = PeftModel.from_pretrained(model, checkpoint)
        model.eval()
        
        # 프롬프트 포맷팅 함수
        formatting_func = get_model_specific_formatting(model_name)
        
        # ROUGE 스코어러 초기화
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 평가 수행
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # BERTScore를 위한 참조 요약과 생성 요약 리스트
        reference_summaries = []
        generated_summaries = []
        
        num_eval_samples = min(num_samples, len(dataset))
        print(f"테스트 데이터 {num_eval_samples}개 샘플 평가 중...")
        
        for i in range(num_eval_samples):
            if (i + 1) % 20 == 0:
                print(f"진행 중: {i + 1}/{num_eval_samples}")
            
            try:
                # 데이터셋에서 직접 인덱스로 접근
                example = dataset[i]
                
                # example이 딕셔너리인지 확인
                if not isinstance(example, dict):
                    print(f"경고: 샘플 {i}가 딕셔너리가 아닙니다. 타입: {type(example)}")
                    if isinstance(example, str):
                        print(f"  값 (처음 200자): {example[:200]}")
                    continue
                
                # 필수 필드 확인
                if 'document' not in example or 'summary' not in example:
                    print(f"경고: 샘플 {i}에 'document' 또는 'summary' 필드가 없습니다. 키: {list(example.keys())}")
                    continue
                
                document = example['document']
                reference_summary = example['summary']
                
                # document와 summary가 문자열인지 확인
                if not isinstance(document, str) or not isinstance(reference_summary, str):
                    print(f"경고: 샘플 {i}의 document 또는 summary가 문자열이 아닙니다.")
                    continue
                    
            except Exception as e:
                print(f"샘플 {i} 처리 중 오류: {e}")
                continue
            
            # 프롬프트 생성
            prompt = formatting_func(document)
            
            # 토크나이징
            inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            
            # 생성
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 요약 추출
            if "<start_of_turn>model" in generated_text:
                summary = generated_text.split("<start_of_turn>model")[-1]
            elif "<|assistant|>" in generated_text:
                summary = generated_text.split("<|assistant|>")[-1]
            elif "assistant" in generated_text.lower():
                summary = generated_text.split("assistant")[-1]
            else:
                summary = generated_text[len(prompt):]
            
            summary = clean_summary(summary)
            
            # ROUGE 점수 계산
            scores = scorer.score(reference_summary, summary)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # BERTScore 계산을 위한 데이터 수집
            reference_summaries.append(reference_summary)
            generated_summaries.append(summary)
        
        # 평균 점수 계산
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        
        # BERTScore 계산
        print("BERTScore 계산 중...")
        try:
            P, R, F1 = bert_score(
                generated_summaries,
                reference_summaries,
                lang='ko',  # 한국어 데이터셋이므로 한국어 모델 사용
                verbose=False,
                device=next(model.parameters()).device
            )
            avg_bertscore_precision = P.mean().item()
            avg_bertscore_recall = R.mean().item()
            avg_bertscore_f1 = F1.mean().item()
        except Exception as e:
            print(f"BERTScore 계산 실패: {e}")
            # BERTScore 계산 실패 시 기본값 사용
            avg_bertscore_precision = 0.0
            avg_bertscore_recall = 0.0
            avg_bertscore_f1 = 0.0
        
        result = {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "num_samples": len(rouge1_scores),
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "bertscore_precision": avg_bertscore_precision,
            "bertscore_recall": avg_bertscore_recall,
            "bertscore_f1": avg_bertscore_f1,
            "status": "success",
        }
        
        print(f"\n평가 완료: {model_name}")
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")
        print(f"BERTScore Precision: {avg_bertscore_precision:.4f}")
        print(f"BERTScore Recall: {avg_bertscore_recall:.4f}")
        print(f"BERTScore F1: {avg_bertscore_f1:.4f}\n")
        
        return result
        
    except Exception as e:
        print(f"평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
        }

def main():
    """메인 함수: 모든 모델 평가 및 결과 비교"""
    print("="*80)
    print("LLM 모델 성능 평가 시작")
    print("="*80)
    
    results = []
    
    # 각 모델 평가
    for model_config in MODELS_TO_EVALUATE:
        result = evaluate_model(model_config, num_samples=100)
        if result:
            results.append(result)
    
    # 결과 저장
    results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 결과 요약 출력
    print("\n" + "="*80)
    print("평가 결과 요약")
    print("="*80)
    print(f"{'모델':<40} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'BERT-F1':<10}")
    print("-"*80)
    
    for result in results:
        if result.get("status") == "success":
            print(f"{result['model_name']:<40} {result['rouge1']:<10.4f} {result['rouge2']:<10.4f} {result['rougeL']:<10.4f} {result.get('bertscore_f1', 0.0):<10.4f}")
        else:
            print(f"{result['model_name']:<40} {'실패':<10}")
    
    # 최고 성능 모델 찾기
    successful_results = [r for r in results if r.get("status") == "success"]
    if successful_results:
        best_rouge1 = max(successful_results, key=lambda x: x['rouge1'])
        best_rouge2 = max(successful_results, key=lambda x: x['rouge2'])
        best_rougeL = max(successful_results, key=lambda x: x['rougeL'])
        best_bertscore_f1 = max(successful_results, key=lambda x: x.get('bertscore_f1', 0.0))
        
        print("\n" + "="*80)
        print("최고 성능 모델")
        print("="*80)
        print(f"ROUGE-1 최고: {best_rouge1['model_name']} ({best_rouge1['rouge1']:.4f})")
        print(f"ROUGE-2 최고: {best_rouge2['model_name']} ({best_rouge2['rouge2']:.4f})")
        print(f"ROUGE-L 최고: {best_rougeL['model_name']} ({best_rougeL['rougeL']:.4f})")
        print(f"BERTScore F1 최고: {best_bertscore_f1['model_name']} ({best_bertscore_f1.get('bertscore_f1', 0.0):.4f})")
    
    print(f"\n상세 결과는 {results_file} 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()

