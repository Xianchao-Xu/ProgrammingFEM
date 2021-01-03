# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'sparse_cho_bac',
    'sparse_cho_fac',
]


def sparse_cho_bac(kv, loads, k_diag):
    """
    稀疏矩阵Cholesky分解的前代和回代
    :param kv:
    :param loads:
    :param k_diag:
    :return: loads，此时存储的数据为计算得到的节点位移
    """
    n = np.size(k_diag, 0)
    loads[1] = loads[1] / kv[0]
    for i in range(1, n):
        ki = k_diag[i] - i - 1
        li = k_diag[i - 1] - ki
        x = loads[i + 1]
        if li != i:
            m = i
            for j in range(li, m):
                x = x - kv[ki + j] * loads[j + 1]
        loads[i + 1] = x / kv[ki + i]
    for it in range(1, n):
        i = n - it
        ki = k_diag[i] - i - 1
        x = loads[i + 1] / kv[ki + i]
        loads[i + 1] = x
        li = k_diag[i - 1] - ki
        if li != i:
            m = i
            for k in range(li, m):
                loads[k + 1] = loads[k + 1] - x * kv[ki + k]
    loads[1] = loads[1] / kv[0]
    loads[0] = 0.0
    return loads


def sparse_cho_fac(kv, k_diag):
    """
    一维变带宽存储的稀疏矩阵Cholesky分解
    :param kv: 向量形式存储的稀疏矩阵
    :param k_diag: 对角线辅助向量
    :return 分解后的刚度矩阵（向量）
    """
    x = 0
    n = np.size(k_diag, 0)
    kv[0] = np.sqrt(kv[0])
    for i in range(1, n):
        ki = k_diag[i] - (i + 1)
        li = k_diag[i - 1] - ki + 1
        for j in range(li, i + 2):
            x = kv[ki + j - 1]
            kj = k_diag[j - 1] - j
            if j != 1:
                ll = k_diag[j - 2] - kj + 1
                ll = max(li, ll)
                if ll != j:
                    m = j
                    for k in range(ll, m):
                        x = x - kv[ki + k - 1] * kv[kj + k - 1]
            kv[ki + j - 1] = x / kv[kj + j - 1]
        kv[ki + i] = np.sqrt(x)
    return kv
