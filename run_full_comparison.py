"""
전체 프로세스 자동화: 학습 → 평가 → 결과 비교
"""

import subprocess
import sys
import os
import signal

def run_training():
    """모델 학습 실행 (각 모델을 순차적으로 학습)"""
    print("="*80)
    print("1단계: 모델 학습 시작")
    print("="*80)
    
    try:
        # Popen을 사용하여 시그널 전달 가능하도록
        process = subprocess.Popen(
            [sys.executable, "train_all_models.py"],
            cwd="/data2/jeOn9/fsdp_practices"
        )
        
        # 시그널 핸들러 설정
        def signal_handler(sig, frame):
            print("\n\n중단 요청 받음. 프로세스 종료 중...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 프로세스 완료 대기
        result = process.wait()
        
        if result != 0:
            print("학습 중 오류가 발생했습니다.")
            return False
        
        print("\n학습 완료!\n")
        return True
        
    except KeyboardInterrupt:
        print("\n\n중단됨")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return False

def run_evaluation():
    """모델 평가 실행"""
    print("="*80)
    print("2단계: 모델 평가 시작")
    print("="*80)
    
    try:
        process = subprocess.Popen(
            [sys.executable, "evaluate_models.py"],
            cwd="/data2/jeOn9/fsdp_practices"
        )
        
        def signal_handler(sig, frame):
            print("\n\n중단 요청 받음. 프로세스 종료 중...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        result = process.wait()
        
        if result != 0:
            print("평가 중 오류가 발생했습니다.")
            return False
        
        print("\n평가 완료!\n")
        return True
        
    except KeyboardInterrupt:
        print("\n\n중단됨")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return False

def main():
    """메인 함수: 학습 → 평가 순차 실행"""
    print("\n" + "="*80)
    print("LLM 모델 전체 비교 프로세스 시작")
    print("="*80 + "\n")
    
    # 1. 학습
    if not run_training():
        print("학습 실패로 인해 프로세스를 중단합니다.")
        sys.exit(1)
    
    # 2. 평가
    if not run_evaluation():
        print("평가 실패로 인해 프로세스를 중단합니다.")
        sys.exit(1)
    
    print("="*80)
    print("전체 프로세스 완료!")
    print("="*80)
    print("\n결과 파일:")
    print("- 학습 결과: comparison_results_*.json")
    print("- 평가 결과: evaluation_results_*.json")

if __name__ == "__main__":
    main()

