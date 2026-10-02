"""세트 C 가 B2 에서 나빠지는 원인 분해: 피처를 하나씩 더하고 빼며 B1 Valid 와 B2 성능을 본다.
(36셀, 선형 회귀, 공식 Hold-out 시드 42)

※ 이 실험은 B2 결과를 본 뒤 하는 **사후 분석**이다. 모델 선택에는 사용하지 않고 해석에만 쓴다.
실행: python -m src.analysis.feature_ablation -> results/feature_ablation.csv
"""
import numpy as np
import pandas as pd

from src.config import R
from src.data import FEATURE_SETS, build_dataset, test_set, training_pool
from src.evaluate import mape
from src.models import fit_model
from src.split import OFFICIAL_SEED, holdout_split

C = FEATURE_SETS['C']                       # log_dq_var, qd_slope_91_100, switch, tmax_mean, dq_kurt, qd_2
CONFIGS = {
    'A (Var만)': ['log_dq_var'],
    'A + 초기 용량(qd_2)': ['log_dq_var', 'qd_2'],
    'A + 전환 SOC': ['log_dq_var', 'switch'],
    'A + Tmax': ['log_dq_var', 'tmax_mean'],
    'C 전체': C,
    'C − 초기 용량': [f for f in C if f != 'qd_2'],
    'C − 전환 SOC': [f for f in C if f != 'switch'],
    'C − Tmax': [f for f in C if f != 'tmax_mean'],
}


def main():
    ds = build_dataset(); pool = training_pool(ds, '36'); test = test_set(ds)
    tr, va, _ = holdout_split(pool, seed=OFFICIAL_SEED)
    g = (test.b2_group == 'general').values
    rows = []
    for name, cols in CONFIGS.items():
        m, _ = fit_model('LinReg', pool.iloc[tr][cols], pool.iloc[tr].y, pool.iloc[tr].protocol.values)
        valid = mape(pool.iloc[va].y, m.predict(pool.iloc[va][cols]))
        mf, _ = fit_model('LinReg', pool[cols], pool.y, pool.protocol.values)        # B1 36셀 전체로 재학습
        p = mf.predict(test[cols])
        rows.append(dict(구성=name, B1_Valid=round(valid, 1), B2_전체=round(mape(test.y, p), 1),
                         B2_일반=round(mape(test.y[g], p[g]), 1), B2_new=round(mape(test.y[~g], p[~g]), 1)))
    out = pd.DataFrame(rows); out.to_csv(R('feature_ablation.csv'), index=False)
    print(out.to_string(index=False))
    # C 전체 모델의 표준화 계수 (어떤 피처가 예측을 움직이는가)
    mf, _ = fit_model('LinReg', pool[C], pool.y, pool.protocol.values)
    coef = pd.Series(mf[-1].coef_, index=C).round(4); coef.to_csv(R('feature_ablation_coef_C.csv'), header=['표준화 계수'])
    print('\nC 전체 표준화 계수:'); print(coef.to_string())
    # 초기 용량 배치별 평균 (B2 일반 셀은 초기 용량이 더 높은데 수명은 짧다)
    ds_v = ds[ds.valid]
    for k, v in initial_capacity_shift().items(): print(f'  {k}: {v}')
    print('\n외삽 플래그 구분력:'); print(flag_discrimination().to_string(index=False))
    print('\n초기 용량 qd_2 평균:', ds_v.groupby('batch').qd_2.mean().round(3).to_dict(),
          '| B2 일반:', round(ds_v[(ds_v.batch == 'batch2') & (ds_v.b2_group == 'general')].qd_2.mean(), 3),
          '| B1 학습 풀(36셀):', round(pool.qd_2.mean(), 3))


def initial_capacity_shift():
    """초기 용량의 배치 간 분포 이동이 B2 오차를 만드는지 확인 (사후 분석, 해석용)."""
    ds = build_dataset(); pool = training_pool(ds, '36'); test = test_set(ds)
    gen = test[test.b2_group == 'general']
    mu, sd, mx = pool.qd_2.mean(), pool.qd_2.std(), pool.qd_2.max()
    out = {'B1 학습 풀 초기 용량 평균': round(mu, 3), 'B1 표준편차': round(sd, 4), 'B2 일반 평균': round(gen.qd_2.mean(), 3),
           'B2 일반 평균의 z(B1 기준)': round((gen.qd_2.mean() - mu) / sd, 2), 'B1 최댓값': round(mx, 3),
           'B2 일반 중 B1 최댓값 초과': f'{int((gen.qd_2 > mx).sum())}/{len(gen)}'}
    # A + 초기 용량 모델: B2 일반 셀의 초기 용량만 B1 평균으로 바꿔 예측
    cols = ['log_dq_var', 'qd_2']
    m, _ = fit_model('LinReg', pool[cols], pool.y, pool.protocol.values)
    g = gen.copy()
    out['A+초기용량, B2 일반 MAPE (실제 입력)'] = round(mape(g.y, m.predict(g[cols])), 1)
    g['qd_2'] = mu
    out['A+초기용량, B2 일반 MAPE (초기용량을 B1 평균으로)'] = round(mape(g.y, m.predict(g[cols])), 1)
    # B1 안에서 초기 용량-수명 상관: 단순 vs ΔQ 분산을 통제한 편상관
    r_simple = np.corrcoef(pool.qd_2, pool.y)[0, 1]
    res = lambda v: v - np.polyval(np.polyfit(pool.log_dq_var, v, 1), pool.log_dq_var)
    out['B1 초기용량-log수명 상관(단순)'] = round(r_simple, 2)
    out['B1 초기용량-log수명 편상관(Var[ΔQ] 통제)'] = round(np.corrcoef(res(pool.qd_2), res(pool.y))[0, 1], 2)
    pd.Series(out).to_csv(R('initial_capacity_shift.csv'), header=['값'])
    return out


def flag_discrimination():
    """외삽 플래그(피처 하나라도 학습 범위 밖)가 켜진 셀과 꺼진 셀의 오차 비교 (error_analysis_b2.csv 사용)."""
    e = pd.read_csv(R('error_analysis_b2.csv')); rows = []
    for tag, nm in (('m1', '모델 1'), ('m2', '모델 2')):
        for scope, d in (('전체 39셀', e), ('일반 30셀', e[e.b2_group == 'general'])):
            for flag in (True, False):
                x = d[d[f'extrapolation_{tag}'] == flag]
                rows.append(dict(모델=nm, 범위=scope, 플래그='켜짐' if flag else '꺼짐', 셀수=len(x),
                                 MAPE=round(x[f'abs_pct_err_{tag}'].mean(), 1) if len(x) else None))
    out = pd.DataFrame(rows); out.to_csv(R('extrapolation_flag_check.csv'), index=False); return out


if __name__ == '__main__':
    main()
