"""B2 오류를 충전 프로토콜별로 분해: 프로토콜이 오차를 설명하는가? (Day 1 오류 분석 약속: 프로토콜·수명 구간·ΔQ 형태 확인)
실행: python -m src.analysis.error_by_protocol -> results/error_by_protocol.csv
"""
import pandas as pd

from src.config import R


def main():
    e = pd.read_csv(R('error_analysis_b2.csv'))
    g = e.groupby(['policy', 'b2_group']).agg(셀수=('cell_id', 'count'), 평균수명=('cycle_life', 'mean'),
                                             M1_MAPE=('abs_pct_err_m1', 'mean'), M2_MAPE=('abs_pct_err_m2', 'mean')).reset_index()
    g = g.sort_values('M1_MAPE', ascending=False).round(1); g.to_csv(R('error_by_protocol.csv'), index=False)
    print(g.to_string(index=False))
    top8 = e.sort_values('abs_pct_err_m1', ascending=False).head(8)
    print('\n모델 1 상위 8개 셀의 프로토콜 분포:', top8.policy.value_counts().to_dict())
    gen = g[g.b2_group == 'general']
    print(f'일반 셀 프로토콜별 평균 수명 범위 {gen.평균수명.min()}–{gen.평균수명.max()}, 모델 1 MAPE 범위 {gen.M1_MAPE.min()}–{gen.M1_MAPE.max()}')
    return g


if __name__ == '__main__':
    main()
