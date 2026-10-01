"""피처-수명 상관의 배치별 Spearman ρ 와 p-value (부호 반전 vs 0과 구분 불가를 구별하기 위함).
실행: python sign_stability.py -> ../results/sign_stability.csv, 그리고 knee 이후 기울기 vs 평균 C-rate"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.config import LOWER_BOUND, R

q2 = pd.read_csv(R('eda/q2_degradation.csv'))
q3 = pd.read_csv(R('eda/q3_features.csv')); q4 = pd.read_csv(R('eda/q4_policy_table.csv'))[['batch', 'cell_id', 'avg_c']]
v = q3.merge(q4, on=['batch', 'cell_id']).merge(q2[['batch', 'cell_id', 'early_slope', 'slope_after']], on=['batch', 'cell_id'])
v['y'] = np.log10(v.cycle_life)
feats = {'chargetime_mean': '충전시간 평균', 'avg_c': '평균 C-rate', 'dq_skew': 'ΔQ 왜도',
         'qd_slope_2_100': 'QD 기울기(2~100)', 'early_slope': 'QD 기울기(20~100)', 'dq_kurt': 'ΔQ 첨도', 'switch': '전환 SOC'}
if 'switch' not in v: v = v.merge(pd.read_csv(R('eda/q4_policy_table.csv'))[['batch', 'cell_id', 'switch']], on=['batch', 'cell_id'])
rows = []
for f, nm in feats.items():
    row = {'feature': nm}
    for b, g in v.groupby('batch'):
        r, p = spearmanr(g[f], g.y, nan_policy='omit'); row[b] = f'{r:+.2f} (p={p:.3f})'
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(R('eda/sign_stability.csv'), index=False); print(out.to_string(index=False))

print('\n=== knee 이후 기울기(slope_after, 음수 = 감소) vs 평균 C-rate ===')
print('slope_after 범위: %.5f ~ %.5f Ah/cycle' % (v.slope_after.min(), v.slope_after.max()))
print('부호 규약: |기울기|(감소 속도)와의 상관 = -(slope_after와의 ρ).  + 이면 빠르게 충전할수록 knee 이후 더 가파르게 감소')
for b, g in v.groupby('batch'):
    r, p = spearmanr(g.avg_c, -g.slope_after); print(f'{b}: n={len(g)} ρ(|slope|)={r:+.2f} p={p:.3f}')
g = v[(v.batch == 'batch1') & ~v.cell_id.isin(LOWER_BOUND)]; r, p = spearmanr(g.avg_c, -g.slope_after)
print(f'batch1 (0~4번 제외 36셀): n={len(g)} ρ(|slope|)={r:+.2f} p={p:.3f}')
