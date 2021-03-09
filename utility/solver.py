# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'stability',
    'sparse_cho_bac',
    'sparse_cho_fac',
]


def stability(kv, gv, k_diag, tolerance, limit):
    """
    使用向量迭代法求解屈曲（广义特征值）问题。
    向量迭代法分为正迭代和逆迭代，正迭代求最大特征值，逆迭代求最小特征值。
    正迭代先计算R1=Kx1，再求解Mx2=R1，得到改进的近似特征向量x2……
    逆迭代先计算R1=Mx1，再求解Kx2=R1，得到改进的近似特征向量x2……

    若直接使用得到的近似向量进行下一步的迭代，迭代向量会收敛于特征向量的倍数，
    所以，每次迭代时，需要将迭代向量进行缩放，缩放系数不唯一。
    本算法选择在迭代过程中将向量的最大值缩放为一，这样计算量会比较小。
    收敛后再将特征向量进行唯一归一。
    特征向量收敛时使用的缩放倍数，即为特征值。

    :param kv: 向量形式的整体刚度矩阵
    :param gv: 向量形式的整体几何矩阵
    :param k_diag: 对角元素辅助向量
    :param tolerance: 容差
    :param limit: 迭代次数上限
    :return: 特征值eigenvalue、特征向量eigenvector
    """
    big = 1.0
    num_equation = np.size(k_diag, 0)
    x0 = np.zeros(num_equation+1, dtype=np.float)
    x0[1] = 1.0
    x1 = np.zeros(num_equation+1, dtype=np.float)
    kv = sparse_cho_fac(kv, k_diag)  # 刚度矩阵只需要分解一次
    for iteration in range(limit):
        x1 = sparse_v_multiply(gv, x0, x1, k_diag)
        x1 = sparse_cho_bac(kv, x1, k_diag)
        big = x1[1:].max()
        if np.abs(x1[1:].max()) > big:
            big = x1[1:].min()
        x1 = x1 / big  # 将特征向量中的最大值缩放我1，缩放倍数为1.0/big
        converged = np.abs(x1[1:]-x0[1:]).max() / np.abs(x1[1:]).max() < tolerance
        x0[:] = x1[:]
        if converged:
            break
    x1[1:] = x1[1:] / np.sqrt(np.sum(x1[1:]**2))  # 位移归一
    eigenvector = x1
    eigenvalue = 1.0 / big
    return eigenvalue, eigenvector, iteration


def sparse_cho_bac(kv, loads, k_diag):
    """
    稀疏矩阵Cholesky分解的前代和回代
    :param kv: cholesky分解后的刚度矩阵
    :param loads: 载荷
    :param k_diag: 对角元素定位向量
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


def sparse_v_multiply(kv, u, f, k_diag):
    """
    实现矩阵乘法 F = KU
    因为只存储了下三角阵，而矩阵乘法需要用到整行。
    所以，考虑到对称矩阵的性状，使用对角线元素左侧和下侧的元素与右侧向量相乘。
    注意：向量形式存储时，位移向量需要多一个维度，用于存储固定自由度的位移数据0
    :param kv: 以向量形式存储的对称矩阵
    :param u: 向量u，通常为位移
    :param f: 向量f，通常为载荷
    :param k_diag: 对角线辅助向量
    :return: f
    """
    n = np.size(u, 0) - 1  # 矩阵的维数比位移向量少1
    for i in range(n):
        x = 0
        up = k_diag[i]  # 第i行（从0开始）的对角线元素
        if i == 0:
            low = up  # 首行的首元素与对角线元素相同
        else:
            low = k_diag[i-1] + 1  # 其余行的首个非零元素
        # 对角线左侧元素相乘：
        for j in range(low, up+1):
            # kv通过k_diag中的元素进行索引，而k_diag中的元素从1开始排序，所以索引时需要减1
            # u的0号位置存储0，正常元素从1开始编号，所以u的索引中i需要加1
            x += kv[j-1] * u[i+1+j-up]
        f[i+1] = x
        # 对角线下侧元素相乘
        for j in range(low, up):
            f[i+j-up+1] += kv[j-1] * u[i+1]
    return f
