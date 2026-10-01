"""B1 내부 '수명 외삽' 검증: 프로토콜 평균 수명이 가장 짧은 k개 프로토콜을 검증으로 빼고(학습에서 제외) 학습.

목적: B2 에서 보이는 큰 성능 저하가 'B1 보다 짧은 수명으로의 외삽' 때문인지 확인한다.
결과(요약): 이 검증에서는 피처 세트 C 도 오히려 가장 좋았다 -> B2 의 저하는 수명 범위 외삽이 아니라
'같은 프로토콜의 수명이 배치마다 다른' 배치 이동(프로토콜 -> 수명 관계의 변화)이라는 해석을 뒷받침한다.
실행: python -m src.analysis.shift_check -> results/shift_check_b1.csv
"""
import pandas as pd
from joblib import Parallel, delayed

from src.config import R
from src.data import build_dataset, training_pool
from src.evaluate import mape
from src.models import MODEL_ORDER
from src.train import SETS, fit_predict


def main(ks=(3, 4, 5)):
    ds = build_dataset(); pool = training_pool(ds, '36')
    pm = pool.groupby('protocol').cycle_life.mean().sort_values()

    def task(fs, m, k):
        va = pool.protocol.isin(list(pm.index[:k])).values
        p, *_ = fit_predict(fs, m, pool[~va], pool[va])
        return dict(feat_set=fs, model=m, k=k, mape=mape(pool[va].y, p))

    rows = Parallel(n_jobs=-1)(delayed(task)(fs, m, k) for fs in SETS for m in MODEL_ORDER for k in ks)
    out = pd.DataFrame(rows).groupby(['feat_set', 'model']).mape.mean().unstack().round(1)
    out.to_csv(R('shift_check_b1.csv'))
    print('가장 짧은 프로토콜:', list(pm.index[:5]))
    print(out.to_string())


if __name__ == '__main__':
    main()
