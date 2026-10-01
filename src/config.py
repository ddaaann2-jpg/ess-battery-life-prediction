"""경로·상수·플롯 설정을 한 곳에 모은 설정 모듈 (다른 모듈은 여기서만 경로를 가져온다).

원본 .mat 위치는 환경변수 ESS_RAW_DIR 로 바꿀 수 있다 (기본: <repo>/data/raw).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get('ESS_RAW_DIR', ROOT / 'data' / 'raw'))
CACHE_DIR = ROOT / 'data' / 'cache'
RESULTS_DIR = ROOT / 'results'
FIG_DIR = RESULTS_DIR / 'figures'


def R(name):
    """results/ 아래 파일 경로(str)."""
    return str(RESULTS_DIR / name)


SEED = 42

# ── 배치 구성 (과제 지정 파일) ──────────────────────────────────────────────
BATCH_FILES = {
    'batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'batch2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}
BATCHES = list(BATCH_FILES)

# ── 셀 처리 규칙 (Day 1 EDA 결론) ───────────────────────────────────────────
SPECIAL = ('VarCharge', 'SLOWCYCLE')    # 표준 프로토콜이 아닌 특수 실험 셀 (B2) -> cycle_life 없음, 제외
# Batch 1 중 기록 종료까지 EOL(QD<=0.88Ah)에 도달하지 못한 10셀 (cycle_life = 기록 길이 + 1, 즉 하한값)
UNFINISHED_DROP = [8, 10, 12, 13, 22]   # 논문 코드의 제외 목록과 동일 -> 모든 분석에서 제외 (41셀)
LOWER_BOUND = [0, 1, 2, 3, 4]           # 논문은 2017-06-30 파일의 이어진 기록으로 보정 -> 이 데이터에선 라벨이 하한값
#                                         모델 학습에서는 제외한다 (36셀 = 기본), 41셀·보정은 민감도
# 논문 공식 코드(LoadData.m / Load Data.ipynb)의 상수: 이어붙는 사이클 수 (add_len + 1)
ADD_LEN = {0: 662, 1: 981, 2: 1060, 3: 208, 4: 482}

# 분석 설계 상수
EOL_QD = 0.88            # EOL 용량 기준 (Ah)
I_EARLY_LO, I_EARLY_HI = 9, 99   # ΔQ(V) = Q[cycle 100] - Q[cycle 10]  (배열 인덱스 = cycle - 1)


def setup_matplotlib():
    """플랫폼 독립 한글 폰트 설정 (없는 폰트는 건너뛰고 다음 후보를 사용)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR', 'DejaVu Sans'],
        'axes.unicode_minus': False,
    })
    return plt
