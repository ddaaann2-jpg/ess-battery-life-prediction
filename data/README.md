# 데이터

원본 `.mat`(약 8GB)과 캐시는 용량 때문에 저장소에 포함하지 않는다 (`.gitignore`).

## 필요한 파일 (`data/raw/` 에 넣거나 환경변수 `ESS_RAW_DIR` 로 위치 지정)

| 배치 | 파일 | 역할 |
|---|---|---|
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 학습 |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 테스트 |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | EDA 비교만 |

MIT-Stanford 배터리 데이터셋(Severson et al., 2019, https://data.matr.io/1/) 기반이며, 과제에서 제공받은 파일을 사용했다.
`2018-04-03_varcharge...` 파일은 다른 논문(Attia 2020)용이라 사용하지 않는다.

## 캐시 만들기

```bash
python -m src.preprocess        # data/cache/batch{1,2,3}.pkl 생성 (약 3초, 총 190MB)
```

각 셀에서 필요한 것만 추출한다: 셀 메타(`cycle_life`, 충전 프로토콜), 사이클별 summary 스칼라,
초기 100 사이클의 `Qdlin`/`Tdlin`/`dQdV`(전압축 1,000점), 대표 사이클(2·10·100)의 원시 곡선(I, V, t, Qc).

## 셀 처리 규칙 (`src/config.py`)

| 처리 | 대상 | 이유 |
|---|---|---|
| 제외 | B2 `VarCharge`·`SLOWCYCLE` 8셀 | `cycle_life` 없음 |
| 제외 | B3 미종료 2셀 | EOL(80%) 미도달 |
| 제외 | B1 8·10·12·13·22번 | EOL 미도달 (논문 코드의 제외 목록과 동일) |
| 하한값 | B1 0~4번 | 논문은 2017-06-30 파일의 이어진 기록(`add_len`)을 더해 보정. 이 데이터의 라벨은 하한값 → 모델 학습에서 제외(36셀), 41셀·보정은 민감도 |

배열 인덱스 `i` = cycle `i+1` 이다 (Batch 1 의 cycle 1 은 비어 있어 ΔQ(V)는 cycle 10·100 = 인덱스 9·99 로 계산).
