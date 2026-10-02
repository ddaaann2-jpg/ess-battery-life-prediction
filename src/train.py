"""모델 학습·선택·평가 파이프라인.   실행:  python -m src.train  [--repeats 30]

흐름 (Batch 2 는 선택 단계에서 전혀 사용하지 않는다)
  1) 후보(피처 세트 A~E × 모델 5종) 를 Batch 1 의 '프로토콜 단위 Hold-out' 30회 반복으로 평가
  2) 반복 Hold-out 평균 MAPE 로 선택 (최고 성능의 1 표준오차 이내 후보 중 가장 단순한 것)
  3) 선택된 후보: Train(프로토콜 GroupKFold OOF) / Valid(공식 Hold-out, 시드 42) 성능 계산
  4) B1 학습 풀 전체로 재학습 -> Batch 2 에 1회 평가 (일반 30셀 / newstructure 9셀 분리 보고)
  5) (참고) 모든 후보의 B2 성능, 라벨 처리 민감도, 오류 분석용 셀별 예측 저장
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import R, SEED
from src.data import FEATURE_SETS, build_dataset, test_set, training_pool
from src.evaluate import extrapolation_flag, mape, performance_table, signed_bias
from src.feature_selection import stable_features
from src.models import MODEL_ORDER, fit_model
from src.split import OFFICIAL_SEED, group_kfold_splits, holdout_split

SETS = ['A', 'B', 'C', 'D', 'E']
N_REPEATS = 30


def resolve_features(fs, train_df):
    """피처 세트 -> 컬럼 목록 (세트 E 는 학습 데이터에서 매번 새로 선별)."""
    return stable_features(train_df)[0] if fs == 'E' else list(FEATURE_SETS[fs])


def fit_predict(fs, model, train_df, eval_df, seed=SEED):
    cols = resolve_features(fs, train_df)
    m, info = fit_model(model, train_df[cols], train_df.y, train_df.protocol.values, seed)
    return m.predict(eval_df[cols]), cols, info, m


# ── 1) 후보 탐색 (B1 내부 Hold-out 반복) ─────────────────────────────────────
def _holdout_task(pool, fs, model, split_seed):
    tr, va, _ = holdout_split(pool, seed=split_seed)
    p, cols, _, _ = fit_predict(fs, model, pool.iloc[tr], pool.iloc[va])
    return dict(feat_set=fs, model=model, split_seed=split_seed, mape=mape(pool.iloc[va].y, p), n_feat=len(cols))


def search_candidates(pool, n_repeats=N_REPEATS, n_jobs=-1):
    seeds = list(range(n_repeats)) + [OFFICIAL_SEED]
    tasks = [(fs, m, s) for fs in SETS for m in MODEL_ORDER for s in seeds]
    res = Parallel(n_jobs=n_jobs)(delayed(_holdout_task)(pool, fs, m, s) for fs, m, s in tasks)
    raw = pd.DataFrame(res)
    rep = raw[raw.split_seed != OFFICIAL_SEED]
    agg = rep.groupby(['feat_set', 'model']).agg(holdout_mean=('mape', 'mean'), holdout_std=('mape', 'std'),
                                                  n_feat=('n_feat', 'mean')).reset_index()
    off = raw[raw.split_seed == OFFICIAL_SEED].set_index(['feat_set', 'model']).mape.rename('valid_official')
    agg = agg.merge(off.reset_index(), on=['feat_set', 'model'])
    agg['holdout_se'] = agg.holdout_std / np.sqrt(n_repeats)
    return agg


def select_best(agg):
    """반복 Hold-out 평균 MAPE 가 최고 성능의 1 표준오차 이내인 후보 중 가장 단순한 것 (피처 수 -> 모델 복잡도 순)."""
    best = agg.loc[agg.holdout_mean.idxmin()]
    ok = agg[agg.holdout_mean <= best.holdout_mean + best.holdout_se].copy()
    ok['complexity'] = ok.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    row = ok.sort_values(['n_feat', 'complexity', 'holdout_mean']).iloc[0]
    return row.feat_set, row.model, best


# ── 3~4) 선택 모델 평가 ──────────────────────────────────────────────────────
def oof_train_mape(pool, fs, model, n_splits=5):
    oof = np.zeros(len(pool))
    for tr, va in group_kfold_splits(pool.protocol.values, n_splits):
        p, _, _, _ = fit_predict(fs, model, pool.iloc[tr], pool.iloc[va]); oof[va] = p
    return mape(pool.y, oof)


def evaluate_on_b2(fs, model, pool, test):
    p, cols, info, m = fit_predict(fs, model, pool, test)
    g = (test.b2_group == 'general').values
    out = dict(overall=mape(test.y, p), general=mape(test.y[g], p[g]), new=mape(test.y[~g], p[~g]),
               bias_general=signed_bias(test.y[g], p[g]), bias_new=signed_bias(test.y[~g], p[~g]))
    return out, p, cols, info, m


def evaluate_candidate(fs, model, pool, test):
    """후보 하나의 Train(CV) / Valid(공식 Hold-out) / Test(B2) 성능과 B2 셀별 예측."""
    train_cv = oof_train_mape(pool, fs, model)
    tr, va, hold = holdout_split(pool, seed=OFFICIAL_SEED)
    pv, _, _, _ = fit_predict(fs, model, pool.iloc[tr], pool.iloc[va]); valid = mape(pool.iloc[va].y, pv)
    res, pt, cols, info, _ = evaluate_on_b2(fs, model, pool, test)
    return dict(fs=fs, model=model, train_cv=train_cv, valid=valid, res=res, pred=pt, cols=cols, info=info, hold=hold)


def run(n_repeats=N_REPEATS):
    t0 = time.time()
    ds = build_dataset(); pool = training_pool(ds, '36'); test = test_set(ds)

    # 1~2) 후보 탐색 + 선택 (B1 만 사용)
    agg = search_candidates(pool, n_repeats)
    fs, model, best = select_best(agg)                                        # 모델 1: 전체 후보 중 선택
    fs2, model2, _ = select_best(agg[agg.feat_set.isin(['A', 'B'])])          # 모델 2: Day 1 에서 먼저 정한 'ΔQ 전용(A·B)' 안에서 선택
    agg['selected'] = (agg.feat_set == fs) & (agg.model == model)
    agg['selected_dq_only'] = (agg.feat_set == fs2) & (agg.model == model2)
    agg.sort_values('holdout_mean').round(3).to_csv(R('candidates_holdout.csv'), index=False)

    # 3~4) 두 모델 평가 (B2 는 모델당 1회)
    m1 = evaluate_candidate(fs, model, pool, test); m2 = evaluate_candidate(fs2, model2, pool, test)
    p1 = performance_table(m1['train_cv'], m1['valid'], m1['res']['overall'], m1['res']['general'], m1['res']['new'])
    p2 = performance_table(m2['train_cv'], m2['valid'], m2['res']['overall'], m2['res']['general'], m2['res']['new'])
    perf = p1[['구분']].copy()
    perf[f'모델 1 (선택: {fs}+{model})'] = p1['MAPE (%)'].round(1)
    perf[f'모델 2 (ΔQ 전용: {fs2}+{model2})'] = p2['MAPE (%)'].round(1)
    perf['비고'] = p1['비고']; perf.to_csv(R('model_performance.csv'), index=False)

    stab = stable_features(pool)[1]; stab.round(3).to_csv(R('feature_stability_train.csv'), index=False)
    def _meta(m):
        return dict(feature_set=m['fs'], model=m['model'], features=m['cols'], hyperparameters=m['info'],
                    train_cv=round(m['train_cv'], 2), valid=round(m['valid'], 2), test_b2=round(m['res']['overall'], 2))
    meta = dict(model_1=_meta(m1), model_2_dq_only=_meta(m2), label_policy='36', n_train_cells=len(pool),
                n_train_protocols=int(pool.protocol.nunique()), official_holdout_seed=OFFICIAL_SEED,
                official_holdout_protocols=m1['hold'], n_repeats=n_repeats,
                selection_rule='mean repeated protocol hold-out MAPE; simplest candidate within 1 SE of the best',
                best_by_mean=dict(feat_set=best.feat_set, model=best.model, mean=round(best.holdout_mean, 2)))
    json.dump(meta, open(R('selected_model.json'), 'w'), ensure_ascii=False, indent=2)

    # 오류 분석용 셀별 예측 (+ 외삽 플래그: 학습 범위 밖 피처 수 / 핵심 피처 Var[ΔQ] 범위 이탈)
    err = test[['cell_id', 'policy', 'b2_group', 'cycle_life', 'log_dq_var', 'qd_2']].copy()
    for tag, m in (('m1', m1), ('m2', m2)):
        err[f'pred_{tag}'] = 10 ** m['pred']
        err[f'abs_pct_err_{tag}'] = np.abs(err[f'pred_{tag}'] - err.cycle_life) / err.cycle_life * 100
        err[f'signed_pct_err_{tag}'] = (err[f'pred_{tag}'] - err.cycle_life) / err.cycle_life * 100
        lo, hi = pool[m['cols']].min(), pool[m['cols']].max()
        err[f'n_feat_out_{tag}'] = ((test[m['cols']] < lo) | (test[m['cols']] > hi)).sum(axis=1).values
        err[f'extrapolation_{tag}'] = extrapolation_flag(pool[m['cols']], test[m['cols']])    # 피처 하나라도 학습 범위 밖이면 True
    err['var_out_of_range'] = ((test.log_dq_var < pool.log_dq_var.min()) | (test.log_dq_var > pool.log_dq_var.max())).values
    err.round(4).to_csv(R('error_analysis_b2.csv'), index=False)

    # (참고) 모든 후보의 B2 성능 -- 선택에 사용하지 않았다
    def _ref(fs_, m_):
        r, *_ = evaluate_on_b2(fs_, m_, pool, test)
        return dict(feat_set=fs_, model=m_, **{k: round(v, 1) for k, v in r.items() if not k.startswith('bias')})
    ref = pd.DataFrame(Parallel(n_jobs=-1)(delayed(_ref)(a, b) for a in SETS for b in MODEL_ORDER))
    ref.to_csv(R('candidates_b2_reference.csv'), index=False)

    # 라벨 처리 민감도 (두 모델 고정)
    sens = []
    for pol, name in [('36', '36셀 (기본)'), ('41', '원 라벨 41셀'), ('41_addlen', 'add_len 보정 41셀')]:
        pl = training_pool(ds, pol)
        for tag, (f_, m_) in (('모델 1', (fs, model)), ('모델 2', (fs2, model2))):
            r, *_ = evaluate_on_b2(f_, m_, pl, test)
            sens.append(dict(model=tag, train_labels=name, B2=round(r['overall'], 1), B2_general=round(r['general'], 1),
                             B2_new=round(r['new'], 1), bias_general=round(r['bias_general'], 1), bias_new=round(r['bias_new'], 1)))
    pd.DataFrame(sens).to_csv(R('sensitivity_labels.csv'), index=False)

    print(f'모델 1: {fs}+{model} ({", ".join(m1["cols"])})   모델 2: {fs2}+{model2}   | 소요 {time.time() - t0:.0f}s')
    print(perf.to_string(index=False))
    return dict(agg=agg, perf=perf, meta=meta, err=err, ref=ref, sens=pd.DataFrame(sens), m1=m1, m2=m2, pool=pool, test=test)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--repeats', type=int, default=N_REPEATS)
    run(ap.parse_args().repeats)
