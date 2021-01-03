# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'add_displacement',
    'add_force',
    'form_k_diag',
    'form_sparse_v',
    'initialize_node_dof',
    'pin_jointed',
    'rod_bee',
    'rod_ke',
    'update_node_dof',
]


def add_displacement(num_disp_node, num_node_dof, disp_node_ids, displacements,
                     node_ids, kv, k_diag, node_dof, loads, penalty):
    """
    使用置大数法处理位移约束，根据受位移约束的自由度以及对应的刚度处诚意大数
    :param num_disp_node:  受位移约束的节点数
    :param num_node_dof: 单元的自由度数
    :param disp_node_ids: 受约束的节点编号
    :param displacements: 位移
    :param node_ids: 总节点编号
    :param kv: 向量形式存储的刚度矩阵
    :param k_diag: 刚度矩阵的对角元素定位向量
    :param node_dof: 总的自由度矩阵
    :param loads: 载荷
    :param penalty: 置大数的大数
    :return: 乘以大数后的刚度矩阵（向量）和载荷
    """
    if num_disp_node != 0:
        disp_dof = np.zeros((num_disp_node, num_node_dof), dtype=np.int)
        for i in range(num_disp_node):
            disp_dof[i, :] = node_dof[node_ids.index(disp_node_ids[i]), :]
            for j in range(num_node_dof):
                if displacements[i, j] != 0.0:
                    kv[k_diag[disp_dof[i, j]-1]-1] += penalty
                    loads[disp_dof[i, j]] = kv[k_diag[disp_dof[i, j]-1]-1] * displacements[i, j]
    return kv, loads


def add_force(num_equation, node_ids, num_node_dof, node_dof, num_loaded_nodes,
              loaded_node_ids, forces):
    """
    生成载荷向量
    :param num_equation: 方程总数
    :param node_ids: 节点编号
    :param num_node_dof: 节点自由度数
    :param node_dof: 节点自由度
    :param num_loaded_nodes: 受载节点数
    :param loaded_node_ids: 受载节点编号
    :param forces: 载荷
    :return: 载荷向量
    """
    # 整体载荷向量，受约束节点处所受载荷存储于第一位，所以数组长度比自由度多1
    loads = np.zeros(num_equation + 1, dtype=np.float)
    for i in range(num_loaded_nodes):
        for j in range(num_node_dof):
            loads[node_dof[node_ids.index(loaded_node_ids[i]), j]] = forces[i, j]
    return loads


def form_k_diag(k_diag, elem_dof):
    """
    :param k_diag: 对角线辅助向量
    :param elem_dof: 单元定位向量，存储单元中各节点的自由度
    :return: k_diag, 刚度矩阵中每一行需要存储的元素个数

    一维变带宽存储时，向量kv只存储刚度矩阵中的部分数据，即：
    从每一行的第一个非零元素开始，到对角线元素结束。
    令k=elem_dof[i]，则k为系统的第k个自由度，也就是矩阵的第k行，
    而elem_dof中的其它自由度，则是矩阵中同一行的其它元素。
    一行中最大自由度编号减最小自由度编号再加一，即为该行需要存储的元素个数。
    """
    i_dof = np.size(elem_dof)

    # 遍历单元，自由度相减并加一。
    for i in range(i_dof):
        iwp1 = 1
        if elem_dof[i] != 0:
            for j in range(i_dof):
                if elem_dof[j] != 0:
                    im = elem_dof[i] - elem_dof[j] + 1
                    if im > iwp1:
                        iwp1 = im
            # 由于索引从0开始，所以对k_diag向量的索引需要减一
            k = elem_dof[i]
            if iwp1 > k_diag[k-1]:
                k_diag[k-1] = iwp1
    return k_diag


def form_sparse_v(kv, ke, elem_dof, k_diag):
    """
    组装整体刚度矩阵，以一维变带宽方式存储。
    :param kv: 向量形式的整体刚度矩阵
    :param ke: 单元刚度矩阵
    :param elem_dof: 单元定位向量（单元中各自由度的全局编号）
    :param k_diag: 对角线辅助向量

    刚度矩阵k和向量kv的关系为：k[i,j] = kv[k_diag[i]-i+j]
    由于索引从0开始，所以索引时需要减一
    """
    # elem_dof[i]即为上面公式中的i
    # elem_dof[j]即为上面公式中的j
    i_dof = np.size(elem_dof, 0)
    for i in range(i_dof):
        if elem_dof[i] != 0:
            for j in range(i_dof):
                if elem_dof[j] != 0:
                    if elem_dof[i] - elem_dof[j] >= 0:
                        i_val = k_diag[elem_dof[i]-1] - elem_dof[i] + elem_dof[j]
                        kv[i_val-1] += ke[i, j]
    return kv


def initialize_node_dof(node_ids, num_node_dof, fixed_node_ids, fixed_components):
    """
    初始化节点自由度矩阵为1，然后将被约束的自由度置零
    :param node_ids: 节点编号
    :param num_node_dof: 节点自由度数
    :param fixed_node_ids: 受约束节点编号
    :param fixed_components: 受约束的自由度
    :return: 节点自由度矩阵
    """
    num_node = len(node_ids)
    node_dof = np.ones((num_node, num_node_dof), dtype=np.int)
    num_fixed_node = len(fixed_node_ids)
    for i in range(num_fixed_node):
        node_id = fixed_node_ids[i]
        node_dof[node_ids.index(node_id), :] = fixed_components[fixed_node_ids.index(node_id), :]
    return node_dof


def pin_jointed(ke, e, a, coord):
    """
    生成杆单元的刚度矩阵
    :param ke: 刚度矩阵
    :param e: 弹性模量E
    :param a: 截面积A
    :param coord: 单元的节点坐标
    :return: ke
    """
    num_dim = np.size(coord, 1)
    if num_dim == 1:
        length = coord[1, 0] - coord[0, 0]
        ke[0, 0] = 1.0
        ke[0, 1] = -1.0
        ke[1, 0] = -1.0
        ke[1, 1] = 1.0
    elif num_dim == 2:
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        cs = (x2-x1)/length
        sn = (y2-y1)/length
        ll = cs * cs
        mm = sn * sn
        lm = cs * sn
        ke[0, 0] = ll
        ke[2, 2] = ll
        ke[0, 2] = -ll
        ke[2, 0] = -ll
        ke[1, 1] = mm
        ke[3, 3] = mm
        ke[1, 3] = -mm
        ke[3, 1] = -mm
        ke[0, 1] = lm
        ke[1, 0] = lm
        ke[2, 3] = lm
        ke[3, 2] = lm
        ke[0, 3] = -lm
        ke[3, 0] = -lm
        ke[1, 2] = -lm
        ke[2, 1] = -lm
    elif num_dim == 3:
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        z1 = coord[0, 2]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        z2 = coord[1, 2]
        length = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        l = (x2 - x1) / length
        m = (y2 - y1) / length
        n = (z2 - z1) / length
        ll = l * l
        mm = m * m
        nn = n * n
        lm = l * m
        mn = m * n
        ln = l * n
        ke[0, 0] = ll
        ke[3, 3] = ll
        ke[1, 1] = mm
        ke[4, 4] = mm
        ke[2, 2] = nn
        ke[5, 5] = nn
        ke[0, 1] = lm
        ke[1, 0] = lm
        ke[3, 4] = lm
        ke[4, 3] = lm
        ke[1, 2] = mn
        ke[2, 1] = mn
        ke[4, 5] = mn
        ke[5, 4] = mn
        ke[0, 2] = ln
        ke[2, 0] = ln
        ke[3, 5] = ln
        ke[5, 3] = ln
        ke[0, 3] = -ll
        ke[3, 0] = -ll
        ke[1, 4] = -mm
        ke[4, 1] = -mm
        ke[2, 5] = -nn
        ke[5, 2] = -nn
        ke[0, 4] = -lm
        ke[4, 0] = -lm
        ke[1, 3] = -lm
        ke[3, 1] = -lm
        ke[1, 5] = -mn
        ke[5, 1] = -mn
        ke[2, 4] = -mn
        ke[4, 2] = -mn
        ke[0, 5] = -ln
        ke[5, 0] = -ln
        ke[2, 3] = -ln
        ke[3, 2] = -ln
    else:
        print('错误的维度信息')
        return
    ke = ke * e * a / length
    return ke


def rod_bee(bee, length):
    """
    :param bee: 初始化为0的2节点杆单元的B矩阵
    :param length: 杆单元长度
    :return: B矩阵
    """
    bee[0, 0] = -1 / length
    bee[0, 1] = 1 / length
    return bee


def rod_ke(ke, e, a, length):
    """
    :param ke: 初始化的杆单元刚度矩阵
    :param e: 弹性模量
    :param a: 横截面积
    :param length: 杆单元的长度
    :return: 杆单元的单元刚度矩阵
    """
    ke[0, 0] = 1.0
    ke[1, 1] = 1.0
    ke[0, 1] = -1.0
    ke[1, 0] = -1.0
    ke = ke * e * a / length
    return ke


def update_node_dof(node_dof):
    """
    更新节点自由度矩阵
    :param node_dof: 节点自由度矩阵
    :return: 更新后的节点自由度矩阵
    """
    num_node, num_node_dof = np.shape(node_dof)
    m = 0
    for i in range(num_node):
        for j in range(num_node_dof):
            if node_dof[i, j] != 0:
                m += 1
                node_dof[i, j] = m
    return node_dof
