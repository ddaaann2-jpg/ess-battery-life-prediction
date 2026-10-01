"""MIT-Stanford 배터리 .mat(v7.3/HDF5) -> 분석용 캐시(pickle) 변환.

필요한 것만 추출해 캐시한다 (원본 3GB급 -> 수십 MB):
  - 셀 메타   : cycle_life, charging_policy, barcode, channel_id
  - summary   : 사이클별 스칼라 (QD, QC, IR, Tmax/Tavg/Tmin, chargetime)
  - 초기 N 사이클 시계열 : Qdlin / Tdlin / discharge_dQdV  (n_cells, N+1, 1000)
  - 대표 사이클 원시 곡선 : I, V, t, Qc  (충전 전류 패턴 분석용)
"""
import os
import pickle
import h5py
import numpy as np

from src.config import RAW_DIR, CACHE_DIR, BATCH_FILES
N_EARLY = 100                 # 초기 사이클 수 (인덱스 0은 자리표시자라 0..100 저장)
RAW_CYCLES = (2, 10, 100)     # 원시 곡선을 저장할 사이클 (cycles 배열 인덱스)
SUMMARY_KEYS = ['cycle', 'QDischarge', 'QCharge', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']


def _arr(f, ref):
    return np.array(f[ref]).squeeze()


def _policy(f, ref):
    return ''.join(chr(int(x)) for x in np.array(f[ref]).flatten())


def load_batch(name, raw_dir=RAW_DIR):
    path = os.path.join(raw_dir, BATCH_FILES[name])
    out = {'cells': []}
    with h5py.File(path, 'r') as f:
        b = f['batch']
        n = b['summary'].shape[0]
        out['Vdlin'] = _arr(f, b['Vdlin'][0, 0])
        qdlin, tdlin, dqdv = [], [], []
        for i in range(n):
            cl = float(_arr(f, b['cycle_life'][i, 0]))
            summ = f[b['summary'][i, 0]]
            cyc = f[b['cycles'][i, 0]]
            n_cyc = cyc['Qdlin'].shape[0]
            m = min(n_cyc, N_EARLY + 1)

            def stack(key):
                arr = np.full((N_EARLY + 1, 1000), np.nan, dtype=np.float32)
                for c in range(m):
                    v = _arr(f, cyc[key][c, 0])
                    if v.shape == (1000,):          # 빈 사이클(shape (2,))은 NaN 유지
                        arr[c] = v
                return arr
            qdlin.append(stack('Qdlin'))
            tdlin.append(stack('Tdlin'))
            dqdv.append(stack('discharge_dQdV'))

            raw = {}
            for c in RAW_CYCLES:
                if c < n_cyc:
                    raw[c] = {k: _arr(f, cyc[k][c, 0]) for k in ('I', 'V', 't', 'Qc')}
            out['cells'].append({
                'cell_id': i,
                'cycle_life': cl,                       # EOL 미도달 셀은 NaN
                'policy': _policy(f, b['policy_readable'][i, 0]),
                'barcode': str(np.array(f[b['barcode'][i, 0]]).flatten()[:1]),
                'summary': {k: np.array(summ[k]).squeeze() for k in SUMMARY_KEYS},
                'n_cycles': n_cyc,
                'raw': raw,
            })
        out['Qdlin'] = np.stack(qdlin)
        out['Tdlin'] = np.stack(tdlin)
        out['dQdV'] = np.stack(dqdv)
    return out


def build_cache(names=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in names or BATCH_FILES:
        print(f'[{name}] loading ...', flush=True)
        data = load_batch(name)
        with open(os.path.join(CACHE_DIR, f'{name}.pkl'), 'wb') as fh:
            pickle.dump(data, fh, protocol=4)
        print(f'[{name}] cells={len(data["cells"])}  Qdlin={data["Qdlin"].shape}')


def load_cache(name):
    with open(os.path.join(CACHE_DIR, f'{name}.pkl'), 'rb') as fh:
        return pickle.load(fh)


if __name__ == '__main__':
    build_cache()
