"""B1 학습 라벨 처리별 민감도: 단변수 직선(log Var[ΔQ] -> log 수명)을 B2(일반/newstructure)에 적용한 MAPE. (Batch 3는 평가하지 않는다)
 - 원 라벨 41셀 / 36셀(하한값 가능 0~4번 제외) / 논문 코드 add_len 보정 41셀
 add_len 은 논문 공식 저장소(rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)의 상수이며,
 이어진 기록 자체는 이 데이터에 없다. 이어붙는 사이클 수 = 662/981/1060/208/482 (= add_len + 1).
 출처: 공식 저장소 LoadData.m 을 직접 열어 확인했다 (19행 add_len=[661, 980, 1059, 207, 481], 24행 end+add_len(i)+1 로 이어붙임,
 71행 제외 목록 [9,11,13,14,23] = 0-index 8,10,12,13,22). Load Data.ipynb 원문은 열어 보지 않았다.
실행: python -m src.analysis.label_sensitivity  -> results/eda/label_sensitivity.csv
"""
import numpy as np, pandas as pd
from src.config import LOWER_BOUND, R

ADD_LEN = {0: 662, 1: 981, 2: 1060, 3: 208, 4: 482}
q3 = pd.read_csv(R('eda/q3_features.csv'))
q4 = pd.read_csv(R('eda/q4_policy_table.csv'))[['batch', 'cell_id', 'newstructure']]
v = q3.merge(q4, on=['batch', 'cell_id']); v['lv'] = np.log10(v.dq_var)
mape = lambda y, p: np.mean(np.abs(10 ** p - 10 ** y) / 10 ** y) * 100
bias = lambda y, p: np.mean((10 ** p - 10 ** y) / 10 ** y) * 100            # +: 과대 예측

b1 = v[v.batch == 'batch1'].copy()
sets = {'원 라벨 41셀': b1.assign(life=b1.cycle_life),
        '36셀(0~4번 제외)': b1[~b1.cell_id.isin(LOWER_BOUND)].assign(life=lambda d: d.cycle_life),
        'add_len 보정 41셀': b1.assign(life=lambda d: d.cycle_life + d.cell_id.map(ADD_LEN).fillna(0))}
b2 = v[v.batch == 'batch2']; groups = {'B2 일반 30': b2[~b2.newstructure], 'B2 newstructure 9': b2[b2.newstructure],
                                       'B2 전체 39': b2}
rows = []
for name, t in sets.items():
    k = np.polyfit(t.lv, np.log10(t.life), 1)
    for gname, g in groups.items():
        p = np.polyval(k, g.lv); y = np.log10(g.cycle_life)
        rows.append(dict(train=name, n_train=len(t), slope=k[0], group=gname, MAPE=mape(y, p), bias=bias(y, p)))
r = pd.DataFrame(rows).round(3); r.to_csv(R('eda/label_sensitivity.csv'), index=False)
print(r.pivot(index='train', columns='group', values='MAPE').round(1)); print('\n기울기:', r.groupby('train').slope.first().round(3).to_dict())
print('\n부호 있는 오차(+ = 예측이 실제보다 김, %):'); print(r.pivot(index='train', columns='group', values='bias').round(1))
print('\n보정 라벨(0~4번):', {k: int(b1[b1.cell_id == k].cycle_life.iloc[0] + a) for k, a in ADD_LEN.items()})
