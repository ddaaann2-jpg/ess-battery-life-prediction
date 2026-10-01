"""후보 모델과 하이퍼파라미터 탐색 (탐색은 항상 학습 데이터 안의 프로토콜 GroupKFold 로만 한다)."""
import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from src.split import group_kfold_splits

MODEL_ORDER = ['LinReg', 'ElasticNet', 'RandomForest', 'GradBoost', 'LightGBM']      # 단순 -> 복잡 (동률 시 단순한 쪽 선택)

GRIDS = {
    'RandomForest': {'randomforestregressor__min_samples_leaf': [1, 2, 4], 'randomforestregressor__max_depth': [3, None]},
    'GradBoost': {'gradientboostingregressor__n_estimators': [100, 200], 'gradientboostingregressor__learning_rate': [0.05, 0.1]},
    'LightGBM': {'lgbmregressor__n_estimators': [100, 200], 'lgbmregressor__num_leaves': [4, 8],
                 'lgbmregressor__min_child_samples': [3, 5]},
}


def _pipe(est):
    # 결측(IR 측정 실패 등)은 학습 데이터의 중앙값으로 대체 -> 표준화 (둘 다 학습 데이터에서만 계산)
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)


def make_estimator(name, seed=0):
    if name == 'LinReg':
        return _pipe(LinearRegression())
    if name == 'RandomForest':
        return _pipe(RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1))
    if name == 'GradBoost':
        return _pipe(GradientBoostingRegressor(max_depth=2, subsample=0.8, random_state=seed))
    if name == 'LightGBM':
        return _pipe(lgb.LGBMRegressor(learning_rate=0.05, verbose=-1, random_state=seed, n_jobs=1))
    raise ValueError(name)


def fit_model(name, X, y, groups, seed=0):
    """학습 데이터(X, y)에서 모델을 적합. 하이퍼파라미터는 groups(프로토콜) 단위 GroupKFold 로 탐색.
    반환: (적합된 모델, 선택된 하이퍼파라미터 요약 문자열)"""
    splits = group_kfold_splits(groups, 4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if name == 'LinReg':
            m = make_estimator(name).fit(X, y); return m, ''
        if name == 'ElasticNet':
            en = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0], cv=splits, max_iter=100000, random_state=seed)
            m = _pipe(en).fit(X, y)
            e = m[-1]; return m, f'alpha={e.alpha_:.3g}, l1_ratio={e.l1_ratio_:.2g}'
        gs = GridSearchCV(make_estimator(name, seed), GRIDS[name], cv=splits, scoring='neg_mean_absolute_error', n_jobs=1)
        gs.fit(X, y)
        return gs.best_estimator_, ', '.join(f"{k.split('__')[1]}={v}" for k, v in gs.best_params_.items())
