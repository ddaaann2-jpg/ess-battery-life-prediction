"""모델링 결과 시각화·오류 분석 요약 (results/*.csv 를 읽어 그림을 만든다).   python -m src.report"""
import json
import os

import numpy as np
import pandas as pd

from src.config import FIG_DIR, R, setup_matplotlib
from src.data import build_dataset, training_pool

plt = setup_matplotlib()
plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.25, 'font.size': 11})
C_GEN, C_NEW = '#dd6b20', '#2f855a'
C_TRAIN, C_HOLD, C_FIX = '#2b6cb0', '#63b3ed', '#718096'


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = FIG_DIR / name; fig.savefig(path, bbox_inches='tight'); plt.close(fig); return path


def fig_split(pool=None, meta=None):
    """프로토콜 단위 분할 구조: 공식 Hold-out(시드 42) 프로토콜과 Train 에 고정한 극단 프로토콜."""
    pool = pool if pool is not None else training_pool(build_dataset(), '36')
    meta = meta or json.load(open(R('selected_model.json')))
    hold = set(meta['official_holdout_protocols'])
    g = pool.groupby('protocol').cycle_life.agg(['mean', 'count']).sort_values('mean')
    colors = [C_HOLD if p in hold else (C_FIX if p == '5.4C(80%)-5.4C' else C_TRAIN) for p in g.index]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(range(len(g)), g['mean'], color=colors)
    for i, (m, c) in enumerate(zip(g['mean'], g['count'])): ax.text(m + 8, i, f'n={c}', va='center', fontsize=8, color='#555')
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=8); ax.set_xlabel('프로토콜 평균 cycle_life')
    ax.set_title('Batch 1 학습 풀(36셀, 20개 프로토콜)의 프로토콜 단위 분할')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_TRAIN, label='Train'), Patch(color=C_HOLD, label='Hold-out (시드 42)'),
                       Patch(color=C_FIX, label='극단 프로토콜: Train 고정')], loc='lower right', fontsize=9)
    return _save(fig, 'm1_split_protocols.png')


def fig_candidates():
    """B1 Hold-out(선택 기준) vs (사후 참고) B2: 피처 세트 × 모델."""
    h = pd.read_csv(R('candidates_holdout.csv')); b2 = pd.read_csv(R('candidates_b2_reference.csv'))
    models = ['LinReg', 'ElasticNet', 'RandomForest', 'GradBoost', 'LightGBM']; sets = ['A', 'B', 'C', 'D', 'E']
    H = h.pivot(index='feat_set', columns='model', values='holdout_mean').loc[sets, models]
    B = b2.pivot(index='feat_set', columns='model', values='overall').loc[sets, models]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    for a, M, title, vmax in ((ax[0], H, 'B1 Hold-out MAPE (선택 기준, 30회 평균)', 12), (ax[1], B, 'B2 MAPE (사후 참고, 선택에 미사용)', 55)):
        im = a.imshow(M.values, cmap='YlOrRd', vmin=0, vmax=vmax); a.grid(False)
        a.set_xticks(range(5)); a.set_xticklabels(models, rotation=25, ha='right'); a.set_yticks(range(5)); a.set_yticklabels(sets)
        a.set_title(title, fontsize=11)
        for i in range(5):
            for j in range(5): a.text(j, i, f'{M.values[i, j]:.1f}', ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=a, shrink=.8)
    return _save(fig, 'm2_candidates_b1_vs_b2.png')


def fig_pred_vs_true(err=None):
    err = err if err is not None else pd.read_csv(R('error_analysis_b2.csv'))
    meta = json.load(open(R('selected_model.json')))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for a, tag, ttl in ((ax[0], 'm1', f"모델 1: {meta['model_1']['feature_set']}+{meta['model_1']['model']} (선택)"),
                        (ax[1], 'm2', f"모델 2: {meta['model_2_dq_only']['feature_set']}+{meta['model_2_dq_only']['model']} (ΔQ 전용)")):
        for grp, col in (('general', C_GEN), ('newstructure', C_NEW)):
            d = err[err.b2_group == grp]
            a.scatter(d.cycle_life, d[f'pred_{tag}'], s=28, color=col, alpha=.85, label=f'B2 {grp} ({len(d)})')
        lim = [300, 1400]; x = np.array(lim)
        a.plot(x, x, 'k-', lw=1); a.plot(x, x * 1.2, 'k:', lw=.8); a.plot(x, x * .8, 'k:', lw=.8)
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel('실제 cycle_life'); a.set_title(f"{ttl}\nMAPE {err[f'abs_pct_err_{tag}'].mean():.1f}%", fontsize=10.5)
    ax[0].set_ylabel('예측 cycle_life'); ax[0].legend(fontsize=8.5, loc='upper left')
    return _save(fig, 'm3_b2_pred_vs_true.png')


def fig_error_vs_var(err=None, pool=None):
    err = err if err is not None else pd.read_csv(R('error_analysis_b2.csv'))
    pool = pool if pool is not None else training_pool(build_dataset(), '36')
    lo, hi = pool.log_dq_var.min(), pool.log_dq_var.max()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, tag, ttl in ((ax[0], 'm1', '모델 1 (선택)'), (ax[1], 'm2', '모델 2 (ΔQ 전용)')):
        a.axvspan(lo, hi, color='#bee3f8', alpha=.5, label='B1 학습 범위(log Var[ΔQ])')
        for grp, col in (('general', C_GEN), ('newstructure', C_NEW)):
            d = err[err.b2_group == grp]; a.scatter(d.log_dq_var, d[f'signed_pct_err_{tag}'], s=28, color=col, alpha=.85, label=grp)
        a.axhline(0, color='k', lw=.8); a.set_xlabel('log10 Var[ΔQ(V)]'); a.set_title(ttl, fontsize=10.5)
    ax[0].set_ylabel('부호 있는 오차 (%)  (+ = 예측이 실제보다 김)'); ax[0].legend(fontsize=8.5, loc='upper left')
    return _save(fig, 'm4_b2_error_vs_feature.png')


def fig_worst_dq(err=None):
    """오류 분석: (왼쪽) B2 일반 셀 중 오차가 가장 큰/작은 5개의 ΔQ(V) 곡선 형태를 B1 학습 풀 범위와 비교,
    (오른쪽) 셀별 부호 있는 오차 vs 초기 용량 (B1 학습 범위 표시)."""
    from src.data import load_all, test_set
    from src.features import delta_q
    err = err if err is not None else pd.read_csv(R('error_analysis_b2.csv'))
    data = load_all(); ds = build_dataset(); pool = training_pool(ds, '36')
    Vd = data['batch1']['Vdlin']
    dq1 = delta_q(data['batch1']['Qdlin'])[pool.cell_id.values]; dq2 = delta_q(data['batch2']['Qdlin'])
    gen = err[err.b2_group == 'general'].sort_values('abs_pct_err_m1')
    best, worst = gen.head(5), gen.tail(5)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].fill_between(Vd, np.nanmin(dq1, 0), np.nanmax(dq1, 0), color='#bee3f8', alpha=.6, label='B1 학습 풀 범위(min–max)')
    for i, row in worst.iterrows(): ax[0].plot(Vd, dq2[int(row.cell_id)], color='#c53030', lw=1, alpha=.85, label='B2 일반: 오차 큰 5개' if i == worst.index[0] else None)
    for i, row in best.iterrows(): ax[0].plot(Vd, dq2[int(row.cell_id)], color='#2f855a', lw=1, alpha=.85, label='B2 일반: 오차 작은 5개' if i == best.index[0] else None)
    ax[0].axhline(0, color='k', lw=.5); ax[0].set_xlabel('전압 V'); ax[0].set_ylabel('ΔQ(V) = Q100 − Q10 (Ah)'); ax[0].legend(fontsize=8.5, loc='lower left')
    ax[0].set_title('ΔQ(V) 곡선 형태: 오차 큰 셀 vs 작은 셀', fontsize=10.5)
    lo, hi = pool.qd_2.min(), pool.qd_2.max()
    ax[1].axvspan(lo, hi, color='#bee3f8', alpha=.5, label='B1 학습 범위(초기 용량)')
    for grp, col in (('general', C_GEN), ('newstructure', C_NEW)):
        d = err[err.b2_group == grp]; ax[1].scatter(d.qd_2, d.signed_pct_err_m1, s=26, color=col, alpha=.85, label=grp)
    ax[1].axhline(0, color='k', lw=.8); ax[1].set_xlabel('초기 용량 QD(2) (Ah)'); ax[1].set_ylabel('모델 1 부호 있는 오차 (%)')
    ax[1].legend(fontsize=8.5, loc='upper left'); ax[1].set_title('셀별 오차 vs 초기 용량', fontsize=10.5)
    return _save(fig, 'm5_worst_cells_dq_and_capacity.png')


def error_summary(err=None):
    """오류 분석 요약표: 집단/범위 이탈/수명 구간별 MAPE·편향 (모델 1·2)."""
    err = err if err is not None else pd.read_csv(R('error_analysis_b2.csv'))
    rows = []
    def add(name, mask):
        d = err[mask]
        if len(d): rows.append(dict(구분=name, 셀수=len(d), **{f'M{t[-1]} MAPE': round(d[f'abs_pct_err_{t}'].mean(), 1) for t in ('m1', 'm2')},
                                    **{f'M{t[-1]} 편향': round(d[f'signed_pct_err_{t}'].mean(), 1) for t in ('m1', 'm2')}))
    add('B2 전체', err.cycle_life > 0)
    add('일반 30셀', err.b2_group == 'general'); add('newstructure 9셀', err.b2_group == 'newstructure')
    add('수명 < 450', err.cycle_life < 450); add('수명 450~550', err.cycle_life.between(450, 550)); add('수명 > 550', err.cycle_life > 550)
    add('Var[ΔQ] 학습 범위 밖', err.var_out_of_range); add('Var[ΔQ] 학습 범위 안', ~err.var_out_of_range)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for f in (fig_split, fig_candidates, fig_pred_vs_true, fig_error_vs_var, fig_worst_dq): print(f())
    print(error_summary().to_string(index=False))
