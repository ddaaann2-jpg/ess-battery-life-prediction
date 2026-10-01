from src.eda.common import *
from src.features import *
from scipy.stats import spearmanr

data = load_all(); cells = cell_table(data)
q2 = pd.read_csv(R('eda/q2_degradation.csv'))
q3 = pd.read_csv(R('eda/q3_features.csv'))
v = cells[cells.valid].copy()
pp = pd.DataFrame([parse_policy(p) for p in v.policy], index=v.index); v = pd.concat([v, pp], axis=1)
v['base_policy'] = v.policy.str.replace('-newstructure', '', regex=False)
v = v.merge(q2[['batch', 'cell_id', 'qd10', 'early_slope']], on=['batch', 'cell_id'])
v = v.merge(q3[['batch', 'cell_id', 'tavg_mean', 'tmax_mean', 'chargetime_mean']], on=['batch', 'cell_id'])
v.to_csv(R('eda/q4_policy_table.csv'), index=False)
print('파싱 실패:', v.c1.isna().sum(), '| newstructure 셀:', v.groupby('batch').newstructure.sum().to_dict())

print('\n=== 배치별 프로토콜 수 / 셀 수 / 프로토콜당 셀 수 ===')
print(v.groupby('batch').agg(cells=('cell_id', 'count'), protocols=('policy', 'nunique')).assign(per=lambda x: x.cells / x.protocols).round(2))

print('\n=== 프로토콜 내 수명 변동(반복성) vs 프로토콜 간 변동 ===')
for b, g in v.groupby('batch'):
    w = g.groupby('policy').cycle_life.agg(['mean', 'std', 'count']); w = w[w['count'] > 1]
    cv_in = (w['std'] / w['mean']).mean(); cv_between = w['mean'].std() / w['mean'].mean()
    print(b, 'within-protocol CV=%.1f%% | between-protocol CV=%.1f%%' % (cv_in * 100, cv_between * 100))

print('\n=== 배치 1 프로토콜별 평균 수명 (상/하위) ===')
t1 = v[v.batch == 'batch1'].groupby('policy').agg(life=('cycle_life', 'mean'), n=('cycle_life', 'count'), avg_c=('avg_c', 'first'), t80=('t80_min', 'first')).sort_values('life')
print(t1.round(1).head(5)); print(t1.round(1).tail(5))

print('\n=== 고속충전과 수명 (Spearman, 배치별) ===')
cols = ['avg_c', 'c1', 'c2', 'switch', 't80_min', 'tavg_mean', 'chargetime_mean']
tab = pd.DataFrame({b: [spearmanr(g[c], g.cycle_life, nan_policy='omit')[0] for c in cols] for b, g in v.groupby('batch')}, index=cols)
tab['B1+B2'] = [spearmanr(v[v.batch != 'batch3'][c], v[v.batch != 'batch3'].cycle_life, nan_policy='omit')[0] for c in cols]
print(tab.round(2))
print('\n=== 같은 프로토콜의 배치 간 수명 차이 (base policy 평균 수명) ===')
pt = v.pivot_table(index='base_policy', columns='batch', values='cycle_life', aggfunc='mean')
ov = pt.dropna(thresh=2); print(ov.round(0).to_string())
print('\n=== 열화속도 상관은 knee 이후 기울기(slope_after)로 측정 -> sign_stability.py 참고 (loss_rate=(QD10-0.88)/수명은 수명과 기계적으로 묶여 폐기) ===')
print('\n=== avg_c 와 평균온도 상관 ===')
for b, g in v.groupby('batch'): print(b, round(spearmanr(g.avg_c, g.tavg_mean, nan_policy='omit')[0], 2))
print('\n=== newstructure 효과 (동일 base policy, 같은 배치) ===')
for b in ['batch2', 'batch3']:
    g = v[v.batch == b]; print(b, g.groupby('newstructure').cycle_life.agg(['mean', 'count']).round(0).to_dict('index'))

# Fig A: 평균 C-rate vs 수명
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for b in BATCHES:
    g = v[v.batch == b]
    ax[0].scatter(g.avg_c, g.cycle_life, s=np.where(g.newstructure, 38, 22), marker='o', color=BCOLOR[b], alpha=.8, label=BLABEL[b], edgecolor=np.where(g.newstructure, 'k', 'none'), linewidth=.6)
ax[0].set_xlabel('0~80% 평균 충전 C-rate'); ax[0].set_ylabel('cycle_life'); ax[0].set_title('충전 속도 vs 수명 (검은 테두리 = newstructure)'); ax[0].legend(fontsize=8)
for b in BATCHES:
    g = v[v.batch == b]; ax[1].scatter(g.t80_min, g.cycle_life, s=22, color=BCOLOR[b], alpha=.8)
ax[1].set_xlabel('0~80% 충전 시간 (분)'); ax[1].set_title('충전 시간 vs 수명')
for b in BATCHES:
    g = v[v.batch == b]; ax[2].scatter(g.avg_c, g.tavg_mean, s=22, color=BCOLOR[b], alpha=.8)
ax[2].set_xlabel('0~80% 평균 충전 C-rate'); ax[2].set_ylabel('초기 100사이클 평균온도 (°C)'); ax[2].set_title('충전 속도 vs 온도')
print(savefig(fig, 'q4_crate_vs_life.png'))

# Fig B: Batch1 프로토콜별 평균 수명 (정렬) + 충전 전류 파형
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})
t = v[v.batch == 'batch1'].groupby('policy').cycle_life.agg(['mean', 'std', 'count']).sort_values('mean')
ax[0].barh(range(len(t)), t['mean'], xerr=t['std'].fillna(0), color=BCOLOR['batch1'], alpha=.75, error_kw=dict(ecolor='gray', capsize=2))
ax[0].set_yticks(range(len(t))); ax[0].set_yticklabels(t.index, fontsize=8.5); ax[0].set_xlabel('평균 cycle_life (±1 std, 프로토콜당 셀 2~3개)'); ax[0].set_title('Batch 1 프로토콜별 평균 수명')
# 대표 셀 충전 전류 파형 (cycle 10)
b1 = {c['cell_id']: c for c in data['batch1']['cells']}
sel = v[v.batch == 'batch1'].sort_values('cycle_life'); picks = [sel.iloc[0], sel.iloc[len(sel)//2], sel.iloc[-1]]
for pk, col in zip(picks, ['#c53030', '#718096', '#2b6cb0']):
    r = b1[pk.cell_id]['raw'][10]; tt, I = r['t'], r['I']
    m = tt < 30
    ax[1].plot(tt[m], I[m], color=col, lw=1.4, label=f"{pk.policy} (수명 {int(pk.cycle_life)})")
ax[1].set_xlabel('시간 (분)'); ax[1].set_ylabel('전류 I (A)'); ax[1].set_title('cycle 10 충전 전류 파형 (최단/중앙/최장 수명 셀)'); ax[1].legend(fontsize=8)
print(savefig(fig, 'q4_protocol_and_current.png'))
