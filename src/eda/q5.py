from src.eda.common import *
from scipy.stats import spearmanr

q3 = pd.read_csv(R('eda/q3_features.csv'))
q4 = pd.read_csv(R('eda/q4_policy_table.csv'))[['batch', 'cell_id', 'avg_c', 'c1', 'c2', 'switch', 't80_min', 'newstructure']]
v = q3.merge(q4, on=['batch', 'cell_id'])
v['log_life'] = np.log10(v.cycle_life)
FEATS = ['log_dq_var', 'log_dq_min', 'log_dq_abs_mean', 'dq_skew', 'dq_kurt', 'dq_2v',
         'qd_2', 'qd_max_minus_2', 'qd_slope_2_100', 'qd_slope_91_100', 'qd_100',
         'ir_2', 'ir_min', 'ir_100_minus_2', 'ir_mean', 'tavg_mean', 'tmax_mean', 'chargetime_mean',
         'avg_c', 'c1', 'c2', 'switch']
print('NaN/inf 현황:', v[FEATS].replace([np.inf, -np.inf], np.nan).isna().sum()[lambda s: s > 0].to_dict())
v[FEATS] = v[FEATS].replace([np.inf, -np.inf], np.nan)

# 1) 배치별 Spearman (vs log10 수명)
rho = pd.DataFrame({b: [spearmanr(g[f], g.log_life, nan_policy='omit')[0] for f in FEATS] for b, g in v.groupby('batch')}, index=FEATS)
rho['min_abs'] = rho[BATCHES].abs().min(1)
rho['same_sign'] = (np.sign(rho[BATCHES]).nunique(axis=1) == 1)
rho = rho.sort_values('batch1', key=abs, ascending=False)
print('\n=== 피처 vs log10(cycle_life) 상관 (Spearman) ===')
print(rho.round(2).to_string())

# 2) Batch1 피처 간 상관 -> 다중공선성
b1 = v[v.batch == 'batch1'].copy(); b1 = b1.fillna(b1.median(numeric_only=True))
C = b1[FEATS].corr(method='spearman')
pairs = [(a, b, C.loc[a, b]) for i, a in enumerate(FEATS) for b in FEATS[i + 1:] if abs(C.loc[a, b]) >= 0.9]
print('\n=== |r|>=0.9 피처 쌍 (Batch1, Spearman) ===')
for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])): print(f'{a:18s} {b:18s} {r:+.2f}')

def vif(df):
    X = (df - df.mean()) / df.std(); out = {}
    for c in X.columns:
        y = X[c].values; A = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]; r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)
print('\n=== VIF (Batch1) : 전체 후보 22개 ===')
vf = vif(b1[FEATS]).sort_values(ascending=False); print(vf.round(1).to_string())

# 3) 안정 피처 선별: 세 배치 부호 일치 + min|rho| 큰 순 -> 상관 0.9 넘는 중복 제거 -> VIF 확인
cand = rho[rho.same_sign].sort_values('min_abs', ascending=False)
chosen = []
for f in cand.index:
    if all(abs(C.loc[f, c]) < 0.9 for c in chosen): chosen.append(f)
print('\n=== 부호 일치 + 중복(|r|>=0.9) 제거 후 후보 ===')
print(rho.loc[chosen, BATCHES + ['min_abs']].round(2).to_string())
print('\n=== VIF (선별 후보) ===')
print(vif(b1[chosen]).sort_values(ascending=False).round(1).to_string())

# Fig A: Batch1 상관 히트맵 (수명 상관 순)
order = ['log_life'] + list(rho.index)
Cm = b1.assign(log_life=b1.log_life)[order].corr(method='spearman')
fig, ax = plt.subplots(figsize=(10.5, 9))
im = ax.imshow(Cm, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
ax.set_xticklabels(order, rotation=60, ha='right', fontsize=8); ax.set_yticklabels(order, fontsize=8); ax.grid(False)
for i in range(len(order)):
    for j in range(len(order)):
        val = Cm.iloc[i, j]; ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6, color='white' if abs(val) > .6 else 'black')
fig.colorbar(im, ax=ax, shrink=.8); ax.set_title('Batch 1 : 피처 간 / 수명과의 Spearman 상관')
print(savefig(fig, 'q5_corr_heatmap.png'))

# Fig B: 배치별 상관 막대 (안정성)
top = rho.sort_values('min_abs', ascending=False).head(12).index
fig, ax = plt.subplots(figsize=(11, 4.3)); w = .27
for i, b in enumerate(BATCHES): ax.bar(np.arange(len(top)) + (i - 1) * w, rho.loc[top, b], w, color=BCOLOR[b], label=BLABEL[b], alpha=.85)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=40, ha='right', fontsize=8); ax.axhline(0, color='k', lw=.6)
ax.set_ylabel('Spearman vs log10 cycle_life'); ax.set_title('수명과 상관 상위 피처 : 배치 간 안정성'); ax.legend(fontsize=8)
print(savefig(fig, 'q5_feature_stability.png'))
