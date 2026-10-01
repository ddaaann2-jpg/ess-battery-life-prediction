"""평가 지표와 성능표."""
import numpy as np
import pandas as pd

TARGET_MAPE = 9.1       # 원논문 Regression 성능 (Test 오차 %)


def mape(y_log, pred_log):
    """log10 수명의 정답/예측 -> 원 스케일 MAPE(%)."""
    t, p = 10 ** np.asarray(y_log, float), 10 ** np.asarray(pred_log, float)
    return float(np.mean(np.abs(p - t) / t) * 100)


def signed_bias(y_log, pred_log):
    """부호 있는 평균 오차(%): + 이면 예측이 실제보다 길다."""
    t, p = 10 ** np.asarray(y_log, float), 10 ** np.asarray(pred_log, float)
    return float(np.mean((p - t) / t) * 100)


def extrapolation_flag(train_X, X):
    """입력 피처가 학습 데이터 범위(min~max)를 벗어나는 행 표시 (외삽 경고)."""
    lo, hi = train_X.min(), train_X.max()
    return ((X < lo) | (X > hi)).any(axis=1).values


def performance_table(train_cv, valid, test, test_general, test_new, target=TARGET_MAPE):
    """가이드 Reporting format (Regression): Train(B1 CV) / Valid(B1 Hold-out) / Test(B2) + Gap 3종.
    Gap 정의(+ = 나쁜 방향): Train-Valid = Valid - Train (과적합 의심), Valid-Test = Test - Valid (배치 간 일반화 저하),
    Target-Test = Test - Target (논문 9.1% 대비)."""
    rows = [
        ('Train (Batch 1 CV)', train_cv, '프로토콜 단위 GroupKFold(5) out-of-fold'),
        ('Valid (Batch 1 Hold-out)', valid, '프로토콜 단위 Hold-out (시드 42)'),
        ('Test (Batch 2)', test, '39셀, 최종 모델 1회 평가'),
        ('Gap (Train-Valid)', valid - train_cv, '(+): 과적합 의심'),
        ('Gap (Valid-Test)', test - valid, '(+): 배치 간 일반화 저하 의심'),
        ('Gap (Target-Test)', test - target, f'Target: 원논문 {target}% (분할 조건이 달라 참고 지표)'),
        ('  Test: B2 일반 30셀', test_general, '기존 구조 셀 (B1과 유사 구조)'),
        ('  Test: B2 newstructure 9셀', test_new, 'newstructure 셀'),
    ]
    return pd.DataFrame(rows, columns=['구분', 'MAPE (%)', '비고'])
