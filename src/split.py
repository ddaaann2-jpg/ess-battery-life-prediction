"""데이터 분할: 프로토콜 단위로 분리해 누수를 막는다.

같은 충전 프로토콜의 셀은 수명이 비슷해(내 CV 6~7% vs 간 23~40%, Day 1 Q4) 셀 단위 랜덤 분할을 하면
쌍둥이 셀이 train/valid 양쪽에 걸려 성능이 부풀려진다. 그래서 분할 단위를 '프로토콜'로 한다.
"""
import numpy as np
from sklearn.model_selection import GroupKFold

OFFICIAL_SEED = 42
EXTREME_PROTOCOL = '5.4C(80%)-5.4C'      # 가장 짧은 수명 구간(534·559)을 대표 -> 항상 Train 에 둬 학습 범위를 확보


def holdout_split(df, n_holdout=5, seed=OFFICIAL_SEED, fixed_train=(EXTREME_PROTOCOL,)):
    """프로토콜 단위 Hold-out. 반환: (train 위치 인덱스, valid 위치 인덱스, hold-out 프로토콜 목록)"""
    candidates = sorted(p for p in df.protocol.unique() if p not in fixed_train)
    rng = np.random.default_rng(seed)
    hold = sorted(rng.choice(candidates, size=n_holdout, replace=False))
    mask = df.protocol.isin(hold).values
    return np.where(~mask)[0], np.where(mask)[0], hold


def group_kfold_splits(groups, n_splits=5):
    """프로토콜 단위 GroupKFold 분할(결정적). groups: 프로토콜 배열."""
    groups = np.asarray(groups)
    n = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups))
