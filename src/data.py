"""분석용 데이터 구성: 셀 메타 + 초기 사이클 피처 + 라벨 처리 정책.

한 행 = 한 배터리 셀. 사이클 단위 행이 아니라 셀 단위로 모델링한다 (타깃이 셀당 1개).
"""
import numpy as np
import pandas as pd

from src.config import BATCHES, SPECIAL, UNFINISHED_DROP, LOWER_BOUND, ADD_LEN
from src.preprocess import load_cache
from src.features import build_features, parse_policy

TARGET = 'cycle_life'


def load_all():
    return {b: load_cache(b) for b in BATCHES}


def cell_table(data):
    """셀 메타(+ 분석 대상 여부 valid)."""
    rows = []
    for b, d in data.items():
        for c in d['cells']:
            rows.append(dict(batch=b, cell_id=c['cell_id'], cycle_life=c['cycle_life'],
                             policy=c['policy'], n_cycles=c['n_cycles'],
                             special=any(s in c['policy'] for s in SPECIAL)))
    df = pd.DataFrame(rows)
    df['unfinished'] = (df.batch == 'batch1') & df.cell_id.isin(UNFINISHED_DROP)
    df['lower_bound'] = (df.batch == 'batch1') & df.cell_id.isin(LOWER_BOUND)
    df['valid'] = df.cycle_life.notna() & ~df.special & ~df.unfinished     # 분석 대상 셀
    return df


def build_dataset(data=None):
    """전 배치 셀 단위 테이블 (메타 + 피처 + 충전 프로토콜 파라미터 + log 변환 피처)."""
    data = data or load_all()
    cells = cell_table(data)
    feats = []
    for b in BATCHES:
        f = build_features(data[b]); f['batch'] = b
        feats.append(f)
    feat = pd.concat(feats, ignore_index=True).drop(columns=['cycle_life', 'policy'])
    ds = cells.merge(feat, on=['batch', 'cell_id'], how='left')
    pol = pd.DataFrame([parse_policy(p) for p in ds.policy])
    ds = pd.concat([ds, pol.drop(columns=['newstructure'])], axis=1)
    ds['newstructure'] = ds.policy.str.contains('newstructure')
    ds['protocol'] = ds.policy.str.replace('-newstructure', '', regex=False)       # 분할(그룹) 단위
    ds['b2_group'] = np.where(ds.batch != 'batch2', '', np.where(ds.newstructure, 'newstructure', 'general'))
    # 로그 변환 피처 (ΔQ(V) 통계는 자릿수가 다양해 log10 으로 다룬다)
    ds['log_dq_var'] = np.log10(ds.dq_var)
    ds['log_dq_min'] = np.log10(-ds.dq_min.where(ds.dq_min < 0))
    ds['log_dq_abs_mean'] = np.log10(ds.dq_mean.abs())
    ds['y'] = np.log10(ds.cycle_life)                                              # 모델 타깃 (log10 수명)
    return ds


# ── 피처 세트 (Day 1 전략) ─────────────────────────────────────────────────
FEATURE_SETS = {
    'A': ['log_dq_var'],
    'B': ['log_dq_var', 'log_dq_min', 'log_dq_abs_mean'],
    'C': ['log_dq_var', 'qd_slope_91_100', 'switch', 'tmax_mean', 'dq_kurt', 'qd_2'],
    'D': ['log_dq_var', 'log_dq_min', 'log_dq_abs_mean', 'dq_skew', 'dq_kurt', 'dq_2v', 'qd_2', 'qd_max_minus_2',
          'qd_slope_2_100', 'qd_slope_91_100', 'qd_100', 'tavg_mean', 'tmax_mean', 'chargetime_mean', 'switch'],
    # 'E' 는 학습 데이터(B1) 내부의 프로토콜 단위 부호 안정성으로 매번 새로 선별 (src/feature_selection.py)
}
FEATURE_SET_DOC = {
    'A': 'Var[ΔQ] 1개', 'B': 'ΔQ 통계 3종', 'C': 'Day 1 선별 6개(B2·B3 EDA 참고)',
    'D': '전체 후보 15개', 'E': 'B1 내부 안정성 선별(프로토콜 서브샘플링)',
}


# ── 학습 풀 / 테스트 ────────────────────────────────────────────────────────
def training_pool(ds, label_policy='36'):
    """Batch 1 학습 풀.
    '36'        : 하한값 가능 0~4번 제외, 관측 라벨만 (기본)
    '41'        : 원 라벨 41셀 (민감도)
    '41_addlen' : 0~4번 라벨에 논문 코드의 add_len 을 더해 보정한 41셀 (민감도)
    """
    b1 = ds[(ds.batch == 'batch1') & ds.valid].copy()
    if label_policy == '36':
        b1 = b1[~b1.lower_bound]
    elif label_policy == '41_addlen':
        b1['cycle_life'] = b1.cycle_life + b1.cell_id.map(ADD_LEN).fillna(0)
        b1['y'] = np.log10(b1.cycle_life)
    elif label_policy != '41':
        raise ValueError(label_policy)
    return b1.reset_index(drop=True)


def test_set(ds):
    """Batch 2 테스트셋: cycle_life 가 있는 표준 셀 39개 (일반 30 + newstructure 9)."""
    return ds[(ds.batch == 'batch2') & ds.valid].reset_index(drop=True)
