"""EDA 공통 유틸: 캐시 로드, 셀 단위 DataFrame, 플롯 스타일."""
import os
import numpy as np
import pandas as pd

from src.config import (BATCHES, FIG_DIR, R, SPECIAL, UNFINISHED_DROP, LOWER_BOUND, setup_matplotlib)
from src.preprocess import load_cache
from src.data import load_all, cell_table

plt = setup_matplotlib()
BCOLOR = {'batch1': '#2b6cb0', 'batch2': '#dd6b20', 'batch3': '#2f855a'}
BLABEL = {'batch1': 'Batch 1', 'batch2': 'Batch 2', 'batch3': 'Batch 3'}

plt.rcParams.update({
    'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 13, 'axes.titlesize': 14.5,
    'axes.labelsize': 12.5, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 11,
})



def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path
