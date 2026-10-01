from src.eda.common import *
from scipy.signal import medfilt
from scipy.stats import spearmanr, pearsonr

data = load_all()
cells = cell_table(data)

def qd_curve(c):
    """사이클별 방전용량 (index 0은 자리표시자, 스파이크는 median filter로 제거)."""
    qd = np.asarray(c['summary']['QDischarge'], float)[1:]
    qd = np.where((qd > 0.6) & (qd < 1.3), qd, np.nan)
    s = pd.Series(qd).interpolate(limit_direction='both').values
    return medfilt(s, 7)

def two_line_knee(y, lo=0.15, hi=0.9):
    """2구간 선형 피팅(piecewise linear)으로 knee(기울기 변화점) 탐색."""
    n = len(y); x = np.arange(n); best = (np.inf, None)
    for k in range(int(n*lo), int(n*hi)):
        sse = 0
        for sl in (slice(0, k), slice(k, n)):
            p = np.polyfit(x[sl], y[sl], 1); sse += ((np.polyval(p, x[sl]) - y[sl])**2).sum()
        if sse < best[0]: best = (sse, k)
    k = best[1]
    s1 = np.polyfit(x[:k], y[:k], 1)[0]; s2 = np.polyfit(x[k:], y[k:], 1)[0]
    return k, s1, s2

rows = []
curves = {}
for b in BATCHES:
    for c in data[b]['cells']:
        row = cells[(cells.batch == b) & (cells.cell_id == c['cell_id'])].iloc[0]
        if not row.valid: continue
        y = qd_curve(c); cl = int(row.cycle_life)
        y = y[:cl]                                   # EOL까지
        curves[(b, c['cell_id'])] = (y, cl)
        k, s1, s2 = two_line_knee(y)
        # 초기 열화: cycle 20~100 기울기 (Ah/cycle)
        e = np.polyfit(np.arange(20, 100), y[19:99], 1)[0]
        rows.append(dict(batch=b, cell_id=c['cell_id'], cycle_life=cl, policy=c['policy'],
                         qd10=y[9], qd100=y[99], dq_10_100=y[99]-y[9], early_slope=e,
                         knee=k, knee_frac=k/cl, slope_before=s1, slope_after=s2, accel=s2/s1 if s1 else np.nan))
df = pd.DataFrame(rows)
df.to_csv(R('eda/q2_degradation.csv'), index=False)

print('=== 열화 가속/knee 요약 (배치별 중앙값) ===')
print(df.groupby('batch')[['knee','knee_frac','slope_before','slope_after','accel']].median().round(5))
print('\n=== knee 후 기울기가 앞의 2배 이상인 셀 비율(가속 셀) ===')
print(df.groupby('batch').apply(lambda g: (g.accel > 2).mean()).round(2))
print('\n=== 초기(20~100) 열화 기울기 vs cycle_life 상관 (Spearman / Pearson) ===')
for b, g in df.groupby('batch'):
    print(b, round(spearmanr(g.early_slope, g.cycle_life)[0], 3), round(pearsonr(g.early_slope, g.cycle_life)[0], 3),
          '| qd100-qd10 vs life:', round(spearmanr(g.dq_10_100, g.cycle_life)[0], 3))
print('\n=== 초기 용량(QD10) 범위 ===')
print(df.groupby('batch').qd10.describe().round(3))

# Fig A: 배치별 QD 곡선 (색 = cycle_life, 공통 컬러스케일)
allcl = df.cycle_life
norm = plt.Normalize(allcl.min(), allcl.max()); cmap = plt.cm.coolwarm_r
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for a, b in zip(ax, BATCHES):
    for (bb, cid), (y, cl) in curves.items():
        if bb == b: a.plot(np.arange(1, len(y)+1), y, color=cmap(norm(cl)), lw=.9, alpha=.85)
    a.axhline(0.88, color='red', ls='--', lw=1); a.set_title(BLABEL[b]); a.set_xlabel('cycle')
ax[0].set_ylabel('방전 용량 QD (Ah)'); ax[0].text(5, .885, 'EOL 0.88Ah (80%)', color='red', fontsize=8, va='bottom')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); fig.colorbar(sm, ax=ax, label='cycle_life', shrink=.85)
print(savefig(fig, 'q2_degradation_curves.png'))

# Fig B: 정규화 곡선(수명 대비 위치) + knee
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for (b, cid), (y, cl) in curves.items():
    ax[0].plot(np.arange(len(y))/cl, y/y[9], color=BCOLOR[b], lw=.6, alpha=.5)
ax[0].set_xlabel('cycle / cycle_life'); ax[0].set_ylabel('QD / QD(10)'); ax[0].set_title('수명 대비 정규화 곡선')
for b in BATCHES:
    g = df[df.batch == b]; ax[1].scatter(g.cycle_life, g.knee, s=18, color=BCOLOR[b], label=BLABEL[b], alpha=.8)
lim = [0, df.cycle_life.max()]; ax[1].plot(lim, lim, 'k:', lw=.8)
ax[1].set_xlabel('cycle_life'); ax[1].set_ylabel('knee 위치(사이클)'); ax[1].set_title('knee point vs 수명'); ax[1].legend(fontsize=8)
for b in BATCHES:
    g = df[df.batch == b]; ax[2].scatter(g.early_slope*1000, g.cycle_life, s=18, color=BCOLOR[b], alpha=.8, label=BLABEL[b])
ax[2].set_xlabel('초기(20~100) 열화 기울기 (mAh/cycle)'); ax[2].set_ylabel('cycle_life'); ax[2].set_title('초기 열화속도 vs 수명')
print(savefig(fig, 'q2_knee_and_early_slope.png'))
