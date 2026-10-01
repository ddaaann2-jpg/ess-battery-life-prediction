from src.eda.common import *
from src.features import *
from scipy.stats import spearmanr, pearsonr

data = load_all(); cells = cell_table(data)
F = {}
for b in BATCHES:
    f = build_features(data[b]); f['batch'] = b
    f = f.merge(cells[cells.batch == b][['cell_id', 'valid']], on='cell_id')
    F[b] = f
feat = pd.concat(F.values(), ignore_index=True)
v = feat[feat.valid].copy()
v['log_life'] = np.log10(v.cycle_life)
v['log_dq_var'] = np.log10(v.dq_var); v['log_dq_min'] = np.log10(-v.dq_min.where(v.dq_min < 0))
v['log_dq_abs_mean'] = np.log10(v.dq_mean.abs())
v.to_csv(R('eda/q3_features.csv'), index=False)

# ΔQ(V) 곡선 shape
print('=== ΔQ(V) (cycle100 - cycle10) 부호/크기 요약 ===')
for b in BATCHES:
    dq = delta_q(data[b]['Qdlin']); m = cells[cells.batch == b].valid.values
    print(b, 'mean min=%.4f  var med=%.2e  | ΔQ<0 비율=%.2f' % (np.nanmedian(dq[m].min(1)), np.nanmedian(np.nanvar(dq[m], 1)), (np.nanmin(dq[m], 1) < 0).mean()))

print('\n=== ΔQ 통계 vs log10(cycle_life) 상관 (Spearman) ===')
cols = ['log_dq_var', 'log_dq_min', 'log_dq_abs_mean', 'dq_skew', 'dq_kurt', 'dq_2v']
res = pd.DataFrame({b: [spearmanr(g[c], g.log_life, nan_policy='omit')[0] for c in cols] for b, g in v.groupby('batch')}, index=cols)
res['B1+B2'] = [spearmanr(v[v.batch.isin(['batch1', 'batch2'])][c], v[v.batch.isin(['batch1', 'batch2'])].log_life, nan_policy='omit')[0] for c in cols]
print(res.round(3))
print('\n=== log10(var ΔQ) vs log10(cycle_life) Pearson ===')
for b, g in v.groupby('batch'):
    print(b, round(pearsonr(g.log_dq_var, g.log_life)[0], 3))

# 장수명 vs 단수명: 배치 내 상/하위 3분위
print('\n=== 배치 내 하위/상위 3분위 ΔQ 통계 중앙값 ===')
for b, g in v.groupby('batch'):
    lo, hi = g.cycle_life.quantile([1/3, 2/3])
    s, l = g[g.cycle_life <= lo], g[g.cycle_life >= hi]
    print(b, 'short life med=%d  log_dq_var=%.2f  min=%.4f | long life med=%d  log_dq_var=%.2f  min=%.4f' %
          (s.cycle_life.median(), s.log_dq_var.median(), s.dq_min.median(), l.cycle_life.median(), l.log_dq_var.median(), l.dq_min.median()))

# Figure A: ΔQ(V) 곡선 (색 = cycle_life)
Vd = data['batch1']['Vdlin']
norm = plt.Normalize(v.cycle_life.min(), v.cycle_life.max()); cmap = plt.cm.coolwarm_r
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for a, b in zip(ax, BATCHES):
    dq = delta_q(data[b]['Qdlin']); m = cells[cells.batch == b].valid.values
    cl = cells[cells.batch == b].cycle_life.values
    for i in np.where(m)[0]: a.plot(Vd, dq[i], color=cmap(norm(cl[i])), lw=.9, alpha=.85)
    a.set_title(BLABEL[b]); a.set_xlabel('전압 V'); a.axhline(0, color='k', lw=.5)
ax[0].set_ylabel('ΔQ(V) = Q100(V) - Q10(V)  (Ah)')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='cycle_life', shrink=.85)
print(savefig(fig, 'q3_deltaQ_curves.png'))

# Figure B: log10 var(ΔQ) vs log10 cycle_life, 배치별 + B1 회귀선
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
b1 = v[v.batch == 'batch1']; k = np.polyfit(b1.log_dq_var, b1.log_life, 1)
xs = np.linspace(v.log_dq_var.min(), v.log_dq_var.max(), 50)
for b in BATCHES:
    g = v[v.batch == b]; ax[0].scatter(g.log_dq_var, g.log_life, s=22, color=BCOLOR[b], label=BLABEL[b], alpha=.85)
ax[0].plot(xs, np.polyval(k, xs), 'k--', lw=1, label='Batch 1 선형 피팅'); ax[0].legend(fontsize=8)
ax[0].set_xlabel('log10 Var[ΔQ(V)]'); ax[0].set_ylabel('log10 cycle_life'); ax[0].set_title('핵심 피처 vs 수명')
for b in BATCHES:
    g = v[v.batch == b]; ax[1].scatter(g.log_dq_min, g.log_life, s=22, color=BCOLOR[b], label=BLABEL[b], alpha=.85)
ax[1].set_xlabel('log10 |min ΔQ(V)|'); ax[1].set_ylabel('log10 cycle_life'); ax[1].set_title('log10 |min ΔQ| vs 수명')
print(savefig(fig, 'q3_deltaQ_feature_scatter.png'))
print('B1 fit: log_life = %.3f * log_dq_var + %.3f' % tuple(k))
# Batch1 선형모형을 Batch2/3에 적용했을 때 MAPE (간이 확인)
for b in ['batch2', 'batch3']:
    g = v[v.batch == b]; pred = 10 ** np.polyval(k, g.log_dq_var)
    print(b, 'B1단순회귀 적용 MAPE = %.1f%%' % (np.mean(np.abs(pred - g.cycle_life) / g.cycle_life) * 100))
