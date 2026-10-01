"""초기 사이클 기반 피처 추출 (Severson et al. 2019 의 피처 정의를 기반으로 함).

사이클 번호는 배열 인덱스 i = cycle i+1 (Batch 1 의 cycle 1 은 비어 있음).
  cycle 10 -> idx 9, cycle 100 -> idx 99
"""
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

I10, I100 = 9, 99


def delta_q(Qdlin, a=I100, b=I10):
    """ΔQ(V) = Q_a(V) - Q_b(V), shape (n_cells, 1000)"""
    return Qdlin[:, a, :] - Qdlin[:, b, :]


def dq_stats(dq):
    """ΔQ(V) 곡선 -> 스칼라 통계."""
    out = {}
    out['dq_var'] = np.nanvar(dq, axis=1)
    out['dq_min'] = np.nanmin(dq, axis=1)
    out['dq_mean'] = np.nanmean(dq, axis=1)
    out['dq_skew'] = skew(dq, axis=1, nan_policy='omit')
    out['dq_kurt'] = kurtosis(dq, axis=1, nan_policy='omit')
    out['dq_2v'] = dq[:, -1]                        # 2.0V 지점의 용량 변화
    return pd.DataFrame(out)


def summary_features(cells, lo=1, hi=100):
    """summary(사이클별 스칼라) 기반 초기 사이클 피처. 인덱스 lo..hi-1 = cycle 2..100."""
    rows = []
    x = np.arange(lo, hi)
    for c in cells:
        s = c['summary']
        qd = np.asarray(s['QDischarge'], float)[lo:hi]
        ir = np.asarray(s['IR'], float)[lo:hi]
        ir = np.where(ir > 0, ir, np.nan)           # IR=0 은 측정 실패
        tavg = np.asarray(s['Tavg'], float)[lo:hi]
        tmax = np.asarray(s['Tmax'], float)[lo:hi]
        ct = np.asarray(s['chargetime'], float)[lo:hi]
        p_all = np.polyfit(x, qd, 1)
        p_late = np.polyfit(x[-10:], qd[-10:], 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # IR 측정 실패(전부 NaN) 셀 -> 결측으로 둔다
            rows.append(dict(
                qd_2=qd[0], qd_max_minus_2=qd.max() - qd[0], qd_slope_2_100=p_all[0], qd_slope_91_100=p_late[0],
                qd_100=qd[-1],
                ir_2=ir[0], ir_min=np.nanmin(ir), ir_100_minus_2=ir[-1] - ir[0], ir_mean=np.nanmean(ir),
                tavg_mean=np.nanmean(tavg), tmax_mean=np.nanmean(tmax),
                chargetime_mean=np.nanmean(ct[:5]),
            ))
    return pd.DataFrame(rows)


def build_features(d):
    """한 배치(캐시 dict) -> 셀 단위 피처 DataFrame."""
    dq = delta_q(d['Qdlin'])
    f = pd.concat([dq_stats(dq), summary_features(d['cells'])], axis=1)
    f.insert(0, 'cell_id', [c['cell_id'] for c in d['cells']])
    f.insert(1, 'cycle_life', [c['cycle_life'] for c in d['cells']])
    f.insert(2, 'policy', [c['policy'] for c in d['cells']])
    return f


_POL = re.compile(r'^([\d.]+)C\(([\d.]+)%\)-([\d.]+)C')


def parse_policy(p):
    """'4.8C(80%)-3.6C-newstructure' -> 충전 파라미터.
    1단계: 0~x% 구간을 c1 C-rate, 2단계: x%~80% 구간을 c2 C-rate (이후 80~100%는 1C CC-CV).
    avg_c = 0~80% 구간 평균 C-rate (조화평균, 시간 가중), t80 = 0~80% 충전 시간(분)."""
    m = _POL.match(p)
    if not m:
        return dict(c1=np.nan, switch=np.nan, c2=np.nan, avg_c=np.nan, t80_min=np.nan, newstructure='newstructure' in p)
    c1, sw, c2 = float(m[1]), float(m[2]) / 100, float(m[3])
    sw = min(sw, 0.8)
    t = sw / c1 + (0.8 - sw) / c2                  # hours
    return dict(c1=c1, switch=sw, c2=c2, avg_c=0.8 / t, t80_min=t * 60, newstructure='newstructure' in p)
