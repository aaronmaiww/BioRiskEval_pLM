# ESM2 Evaluation Optimization Guide

## 주요 개선사항

### 1. 진짜 배치 처리 (True Batch Processing)
- 기존: 시퀀스를 하나씩 순차 처리
- 개선: 유사한 길이의 시퀀스들을 그룹화하여 배치로 처리
- 결과: GPU 활용률 대폭 증가

### 2. 메모리 최적화
- FP16 정밀도 사용으로 메모리 사용량 50% 감소
- 벡터화된 연산으로 중복 계산 제거
- 주기적인 GPU 메모리 정리

### 3. 배치 크기 최적화
- 32GB GPU 기준 권장 배치 크기: 512-1024
- 동적 배치 크기 조정 (시퀀스 길이에 따라)
- 메모리 모니터링으로 OOM 방지

## 사용법

### 기본 사용법 (32GB GPU 최적화)
```bash
cd bioriskeval/gen
./run_optimized_eval.sh
```

### 커스텀 설정
```bash
# 배치 크기 1024, tier 2 평가
./run_optimized_eval.sh 1024 2

# 다른 모델 사용
./run_optimized_eval.sh 512 1 "given131/150M_T1"

# 최대 시퀀스 길이 조정
./run_optimized_eval.sh 512 1 "facebook/esm2_t6_8M_UR50D" 512
```

### 직접 Python 실행
```bash
python eval_ppl_esm2.py \
    --tier 1 \
    --batch-size 1024 \
    --max-seq-len 1024 \
    --use-fp16 \
    --ckpt-path "facebook/esm2_t6_8M_UR50D"
```

## 성능 튜닝 가이드

### GPU 메모리별 권장 설정

| GPU 메모리 | 배치 크기 | 최대 시퀀스 길이 | FP16 |
|-----------|----------|----------------|------|
| 8GB       | 128      | 512            | 필수 |
| 16GB      | 256      | 1024           | 권장 |
| 24GB      | 512      | 1024           | 권장 |
| 32GB      | 1024     | 1024           | 권장 |
| 40GB+     | 2048     | 1024           | 선택 |

### 메모리 부족 시 해결방법
1. 배치 크기 줄이기: `--batch-size 256`
2. 시퀀스 길이 제한: `--max-seq-len 512`
3. FP16 활성화: `--use-fp16` (기본값)

### 속도 최적화
1. 큰 배치 크기 사용 (메모리 허용 범위 내)
2. 시퀀스 길이 제한으로 패딩 최소화
3. CUDA 최적화 환경변수 설정

## 모니터링

스크립트 실행 중 다음 정보가 출력됩니다:
- GPU 메모리 사용량 (로딩 전/후, 처리 중, 완료 후)
- 배치별 처리 속도 (sequences/second)
- 전체 처리 시간 및 통계

## 주요 변경사항

### 새로운 함수들
- `compute_pseudo_ppl_hf_batch()`: 최적화된 배치 처리
- `process_sequence_group_batch()`: 길이별 그룹 처리
- `compute_batch_pseudo_ppl()`: 진짜 배치 연산
- `compute_position_likelihoods_vectorized()`: 벡터화된 위치별 계산
- `print_gpu_memory_info()`: GPU 메모리 모니터링
- `cleanup_gpu_memory()`: 메모리 정리

### 새로운 파라미터들
- `--max-seq-len`: 최대 시퀀스 길이 제한
- `--use-fp16`: FP16 정밀도 사용 여부

## 예상 성능 향상

- **속도**: 5-10배 향상 (배치 크기에 따라)
- **GPU 활용률**: 90%+ (기존 10% 미만에서)
- **메모리 효율성**: FP16으로 50% 절약
- **처리량**: 1000+ sequences/second (32GB GPU 기준)
