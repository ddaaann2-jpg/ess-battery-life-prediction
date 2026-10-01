from src.eda.common import *

data = load_all()
cells = cell_table(data)
v = cells[cells.valid]

print('=== 셀 구성 ===')
print(cells.groupby('batch').agg(total=('cell_id', 'count'), valid=('valid', 'sum'), special=('special', 'sum'),
                                 nan_cl=('cycle_life', lambda x: x.isna().sum())))
print('\n=== 분석 대상 셀 cycle_life 통계 ===')
st = v.groupby('batch').cycle_life.describe().round(1)
st['skew'] = v.groupby('batch').cycle_life.skew().round(2)
print(st)
print('\n=== 장수명(>1000) / 단수명(<500) 비율 ===')
for b, g in v.groupby('batch'):
    print(b, f'n={len(g)}  long(>1000)={int((g.cycle_life>1000).sum())} ({(g.cycle_life>1000).mean():.0%})  '
             f'short(<500)={int((g.cycle_life<500).sum())} ({(g.cycle_life<500).mean():.0%})  '
             f'<550={int((g.cycle_life<550).sum())}')
print('\n=== IQR 이상치 (배치별) ===')
for b, g in v.groupby('batch'):
    q1, q3 = g.cycle_life.quantile([.25, .75]); iqr = q3 - q1
    out = g[(g.cycle_life < q1 - 1.5*iqr) | (g.cycle_life > q3 + 1.5*iqr)]
    print(b, f'fence=({q1-1.5*iqr:.0f}, {q3+1.5*iqr:.0f})'); print(out[['cell_id','cycle_life','policy']].to_string(index=False))
print('\n=== 가장 짧은 셀 5개 / 가장 긴 셀 5개 (배치별) ===')
for b, g in v.groupby('batch'):
    print(b, 'SHORT'); print(g.nsmallest(5, 'cycle_life')[['cell_id','cycle_life','policy']].to_string(index=False))
    print(b, 'LONG');  print(g.nlargest(3, 'cycle_life')[['cell_id','cycle_life','policy']].to_string(index=False))
print('\n=== 제외 셀 ===')
print(cells[~cells.valid][['batch','cell_id','cycle_life','n_cycles','policy']].to_string(index=False))

# --- Figure: 배치별 히스토그램(동일 bin) + 박스/스트립
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.6, 1]})
bins = np.arange(300, 2101, 100)
for b in BATCHES:
    g = v[v.batch == b].cycle_life
    ax[0].hist(g, bins=bins, alpha=.55, color=BCOLOR[b], label=f'{BLABEL[b]} (n={len(g)}, 중앙값 {g.median():.0f})', edgecolor='white')
ax[0].axvline(500, color='gray', ls=':'); ax[0].axvline(1000, color='gray', ls=':')
ax[0].text(500, ax[0].get_ylim()[1]*.95, ' 500', color='gray', va='top'); ax[0].text(1000, ax[0].get_ylim()[1]*.95, ' 1000', color='gray', va='top')
ax[0].set_xlabel('cycle_life (사이클)'); ax[0].set_ylabel('셀 수'); ax[0].set_title('배치별 수명 분포'); ax[0].legend(fontsize=8)
data_box = [v[v.batch == b].cycle_life for b in BATCHES]
bp = ax[1].boxplot(data_box, tick_labels=[BLABEL[b] for b in BATCHES], patch_artist=True, widths=.5)
for p, b in zip(bp['boxes'], BATCHES): p.set_facecolor(BCOLOR[b]); p.set_alpha(.55)
for i, b in enumerate(BATCHES):
    g = v[v.batch == b].cycle_life
    ax[1].scatter(np.random.default_rng(0).normal(i+1, .05, len(g)), g, s=10, color=BCOLOR[b], alpha=.8)
ax[1].set_ylabel('cycle_life'); ax[1].set_title('박스플롯')
print(savefig(fig, 'q1_cycle_life_dist.png'))
