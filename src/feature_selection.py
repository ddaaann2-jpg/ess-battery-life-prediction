"""학습 데이터 내부의 '프로토콜 단위 부호 안정성'으로 피처를 선별한다 (피처 세트 E).

Day 1 의 피처 선별(세 배치 부호 일치)은 B2·B3 EDA 를 참고했다는 한계가 있었다.
여기서는 학습 데이터만 사용한다: 프로토콜의 80% 를 반복 추출(서브샘플링)해 Spearman ρ 를 계산하고
  - |평균 ρ| 가 충분히 크고  - 부호가 반복 추출에서 일관된  피처만 남긴 뒤,
  - 서로 상관이 매우 높은(|ρ| >= 0.9) 중복 피처를 제거한다.
Hold-out/CV 의 각 학습 부분에서 따로 호출하므로 검증 셀이 선별에 섞이지 않는다.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import FEATURE_SETS

POOL = FEATURE_SETS['D']      # 후보 피처 풀


def stability_table(train_df, pool=POOL, n_draws=50, frac=0.8, seed=0):
    protos = np.array(sorted(train_df.protocol.unique()))
    rng = np.random.default_rng(seed)
    k = max(2, int(round(frac * len(protos))))
    rhos = {f: [] for f in pool}
    for _ in range(n_draws):
        sub = train_df[train_df.protocol.isin(rng.choice(protos, size=k, replace=False))]
        for f in pool:
            r = spearmanr(sub[f], sub.y, nan_policy='omit')[0]
            rhos[f].append(0.0 if np.isnan(r) else r)
    rows = []
    for f in pool:
        a = np.array(rhos[f]); m = a.mean()
        rows.append(dict(feature=f, mean_rho=m, std_rho=a.std(), sign_agree=float(np.mean(np.sign(a) == np.sign(m)))))
    return pd.DataFrame(rows)


def stable_features(train_df, rho_min=0.4, agree_min=0.95, corr_max=0.9, **kw):
    """(선별된 피처 목록, 안정성 표). 하나도 못 고르면 Var[ΔQ] 하나로 대체."""
    tab = stability_table(train_df, **kw)
    ok = tab[(tab.mean_rho.abs() >= rho_min) & (tab.sign_agree >= agree_min)].copy()
    ok = ok.reindex(ok.mean_rho.abs().sort_values(ascending=False).index)
    chosen = []
    for f in ok.feature:
        if all(abs(spearmanr(train_df[f], train_df[c], nan_policy='omit')[0]) < corr_max for c in chosen):
            chosen.append(f)
    return (chosen or ['log_dq_var']), tab


def vif(df):
    """피처 간 다중공선성: 각 피처를 나머지 피처로 회귀한 R² 로 VIF = 1/(1-R²) 계산 (결측 행은 제외)."""
    X = df.dropna(); X = (X - X.mean()) / X.std(); out = {}
    for c in X.columns:
        y = X[c].values; A = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out).sort_values(ascending=False)
