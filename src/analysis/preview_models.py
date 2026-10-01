"""사전 점검: 피처 세트 x 모델별 B1 학습 -> B2 테스트 MAPE (참고용, Day 2 본 평가 아님).
학습 세트: 기본 36셀(미종료 5셀 + 하한값 5셀(0~4번) 제외, 관측 라벨만) / 민감도 41셀(미종료 5셀만 제외).
실행: python preview_models.py  -> ../results/preview_model_check.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from src.config import LOWER_BOUND, R

q3 = pd.read_csv(R('eda/q3_features.csv'))
q4 = pd.read_csv(R('eda/q4_policy_table.csv'))[['batch', 'cell_id', 'newstructure', 'switch']]
v = q3.merge(q4, on=['batch', 'cell_id']); v['y'] = np.log10(v.cycle_life)

SETS = {
    'A': ['log_dq_var'],
    'B': ['log_dq_var', 'log_dq_min', 'log_dq_abs_mean'],
    'C': ['log_dq_var', 'qd_slope_91_100', 'switch', 'tmax_mean', 'dq_kurt', 'qd_2'],
    'D': ['log_dq_var', 'log_dq_min', 'log_dq_abs_mean', 'dq_skew', 'dq_kurt', 'dq_2v', 'qd_2', 'qd_max_minus_2',
          'qd_slope_2_100', 'qd_slope_91_100', 'qd_100', 'tavg_mean', 'tmax_mean', 'chargetime_mean', 'switch'],
}
MODELS = {
    'LinReg': lambda: make_pipeline(StandardScaler(), LinearRegression()),
    # cv 는 fit 시점에 프로토콜 단위 GroupKFold 로 교체 (쌍둥이 셀이 폴드에 갈리는 누수 방지)
    'ElasticNet': lambda cv=5: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8, 1], cv=cv, max_iter=50000)),
    'Ridge': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))),
    'RandomForest': lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=0),
    'GradBoost': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=.05, subsample=.8, random_state=0),
    'LightGBM': lambda: lgb.LGBMRegressor(n_estimators=200, num_leaves=4, min_child_samples=3, learning_rate=.05, verbose=-1, random_state=0),
}
mape = lambda y, p: np.mean(np.abs(10 ** p - 10 ** y) / 10 ** y) * 100

b1_all = v[v.batch == 'batch1']; te = v[v.batch == 'batch2']; nw = te.newstructure.values
TRAIN = {'36셀(기본)': b1_all[~b1_all.cell_id.isin(LOWER_BOUND)], '41셀(민감도)': b1_all}
rows = []
for tn, tr in TRAIN.items():
    for sn, cols in SETS.items():
        med = tr[cols].median()
        Xtr, Xte = tr[cols].fillna(med), te[cols].fillna(med)
        for mn, mk in MODELS.items():
            if mn == 'ElasticNet':
                cv = list(GroupKFold(n_splits=5).split(Xtr, tr.y, groups=tr.policy))
                m = mk(cv).fit(Xtr, tr.y)
            else:
                m = mk().fit(Xtr, tr.y)
            p = m.predict(Xte)
            rows.append(dict(train=tn, feat=sn, n_feat=len(cols), model=mn, train_MAPE=mape(tr.y, m.predict(Xtr)),
                             B2=mape(te.y, p), B2_general30=mape(te.y[~nw], p[~nw]), B2_new9=mape(te.y[nw], p[nw])))
r = pd.DataFrame(rows).round(1); r.to_csv(R('eda/preview_model_check.csv'), index=False)
pd.set_option('display.width', 200)
for tn in TRAIN:
    print(f'\n##### 학습셋 {tn}'); print(r[r.train == tn].drop(columns='train').to_string(index=False))

print('\n=== 트리 하한: 예측이 학습 최솟값 미만으로 못 내려갈 때 "완벽 예측"의 MAPE ===')
for tn, tr in TRAIN.items():
    lo = tr.cycle_life.min(); g = te[~nw].cycle_life.values; a = te.cycle_life.values
    f = lambda L: np.mean(np.abs(np.maximum(L, lo) - L) / L) * 100
    print(f'{tn}: 학습 최솟값 {lo:.0f} -> B2 일반 30셀 {f(g):.1f}% / B2 전체 39셀 {f(a):.1f}%')
