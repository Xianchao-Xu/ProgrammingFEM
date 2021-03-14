# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'add_displacement',
    'add_force',
    'beam_ge',
    'beam_me',
    'beam_ke',
    'form_k_diag',
    'form_plate',
    'form_sparse_v',
    'gauss_sample',
    'global_to_axial',
    'hinge_reaction',
    'initialize_node_dof',
    'pin_jointed',
    'rigid_jointed',
    'rod_bee',
    'rod_ke',
    'update_node_dof',
]


def add_displacement(num_disp_node, num_node_dof, disp_node_ids, displacements,
                     node_ids, kv, k_diag, node_dof, loads, penalty):
    """
    使用置大数法处理位移约束
    在受位移约束的自由度以及对应的刚度处乘以大数
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
                if not np.isnan(displacements[i, j]):
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


def beam_ge(ge, length):
    """
    组装梁单元的几何矩阵。
    对梁进行屈曲分析（稳定性分析时），梁受到轴向力作用。
    而梁单元无轴向自由度，方程的其中一项会形成一个只与单元长度相关的矩阵，即几何矩阵。
    :param ge: 几何矩阵
    :param length: 梁单元长度
    :return: 梁单元的几何矩阵
    """
    ge[0, 0] = 1.2 / length
    ge[0, 1] = 0.1
    ge[1, 0] = 0.1
    ge[0, 2] = -1.2 / length
    ge[2, 0] = -1.2 / length
    ge[0, 3] = 0.1
    ge[3, 0] = 0.1
    ge[1, 1] = 2.0 * length / 15.0
    ge[1, 2] = -0.1
    ge[2, 1] = -0.1
    ge[1, 3] = -length / 30.0
    ge[3, 1] = -length / 30.0
    ge[2, 2] = 1.2 / length
    ge[2, 3] = -0.1
    ge[3, 2] = -0.1
    ge[3, 3] = 2.0 * length / 15.0
    return ge


def beam_me(me, fs, length):
    """
    组装梁单元的质量矩阵
    :param me: 质量矩阵
    :param fs: 地基刚度或ρA
    :param length: 单元长度
    :return: 梁单元的质量矩阵
    """
    fac = (fs * length) / 420.0
    me[0, 0] = 156.0 * fac
    me[2, 2] = me[0, 0]
    me[0, 1] = 22.0 * length * fac
    me[1, 0] = me[0, 1]
    me[2, 3] = -me[0, 1]
    me[3, 2] = me[2, 3]
    me[0, 2] = 54.0 * fac
    me[2, 0] = me[0, 2]
    me[0, 3] = -13.0 * length * fac
    me[3, 0] = me[0, 3]
    me[1, 2] = -me[0, 3]
    me[2, 1] = me[1, 2]
    me[1, 1] = 4.0 * (length ** 2) * fac
    me[3, 3] = me[1, 1]
    me[1, 3] = -3.0 * (length ** 2) * fac
    me[3, 1] = me[1, 3]
    return me


def beam_ke(ke, e, i, length):
    """
    生成梁单元的刚度矩阵
    :param ke: 刚度矩阵
    :param e: 弹性模量
    :param i: 惯性矩
    :param length: 单元长度
    :return: 梁单元的刚度矩阵
    """
    ke[0, 0] = 12.0 * e * i / (length ** 3)
    ke[2, 2] = ke[0, 0]
    ke[0, 2] = -ke[0, 0]
    ke[2, 0] = -ke[0, 0]
    ke[0, 1] = 6.0 * e * i / (length ** 2)
    ke[1, 0] = ke[0, 1]
    ke[0, 3] = ke[0, 1]
    ke[3, 0] = ke[0, 1]
    ke[1, 2] = -ke[0, 1]
    ke[2, 1] = -ke[0, 1]
    ke[2, 3] = -ke[0, 1]
    ke[3, 2] = -ke[0, 1]
    ke[1, 1] = 4.0 * e * i / length
    ke[3, 3] = ke[1, 1]
    ke[1, 3] = 2.0 * e * i / length
    ke[3, 1] = ke[1, 3]
    return ke


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


def form_plate(d2x, d2y, d2xy, gauss_points, x_size, y_size, i):
    """
    生成矩形平面弯曲薄板单元形函数的二阶偏导项d2x, d2y, d2xy

    （平面弯曲板单元的形函数公式见《有限元方法编程（第五版）》36~38页）
    高斯积分的积分域是[-1, 1]，而矩形边长分别为a和b，
    书中公式推导时，矩形单元x、y方向的积分域分别为[0, a]和[0, b]，
    所以：x = 0.5a(ξ+1), y = 0.5b(η+1)

    :param d2x: 形函数对ξ的二阶偏导
    :param d2y: 形函数对η的二阶偏导
    :param d2xy: 形函数对ξ、η的混合偏导
    :param gauss_points: 等参元内全部高斯积分点的局部坐标
    :param x_size: 当前单元的x方向边长
    :param y_size: 当前单元的y方向边长
    :param i: 积分点索引号
    :return: d2x, d2y, d2xy
    """
    xi = gauss_points[i, 0]  # ξ
    eta = gauss_points[i, 1]  # η
    xi_plus1 = xi + 1.0  # ξ+1
    xi_plus1_2 = xi_plus1 * xi_plus1  # (ξ+1)^2
    xi_plus1_3 = xi_plus1_2 * xi_plus1  # (ξ+1)^3
    eta_plus1 = eta + 1.0  # η+1
    eta_plus1_2 = eta_plus1 * eta_plus1  # (η+1)^2
    eta_plus1_3 = eta_plus1_2 * eta_plus1  # (η+1)^3

    p1 = 1.0 - 0.75 * xi_plus1_2 + 0.25 * xi_plus1_3
    q1 = 1.0 - 0.75 * eta_plus1_2 + 0.25 * eta_plus1_3
    p2 = 0.5 * x_size * xi_plus1 * (1.0 - xi_plus1 + 0.25 * xi_plus1_2)
    q2 = 0.5 * y_size * eta_plus1 * (1.0 - eta_plus1 + 0.25 * eta_plus1_2)
    p3 = 0.25 * xi_plus1_2 * (3.0 - xi_plus1)
    q3 = 0.25 * eta_plus1_2 * (3.0 - eta_plus1)
    p4 = 0.25 * x_size * xi_plus1_2 * (0.5 * xi_plus1 - 1.0)
    q4 = 0.25 * y_size * eta_plus1_2 * (0.5 * eta_plus1 - 1.0)

    dp1 = 1.5 * xi_plus1 * (0.5 * xi_plus1 - 1.0)  # p1对ξ的一阶偏导
    dq1 = 1.5 * eta_plus1 * (0.5 * eta_plus1 - 1.0)  # q1对η的一阶偏导
    dp2 = x_size * (0.5 - xi_plus1 + 0.375 * xi_plus1_2)  # p2对ξ的一阶偏导
    dq2 = y_size * (0.5 - eta_plus1 + 0.375 * eta_plus1_2)  # q2对η的一阶偏导
    dp3 = 1.5 * xi_plus1 * (1.0 - 0.5 * xi_plus1)  # p3对ξ的一阶偏导
    dq3 = 1.5 * eta_plus1 * (1.0 - 0.5 * eta_plus1)  # q3对η的一阶偏导
    dp4 = 0.5 * x_size * xi_plus1 * (0.75 * xi_plus1 - 1.0)  # p4对ξ的一阶偏导
    dq4 = 0.5 * y_size * eta_plus1 * (0.75 * eta_plus1 - 1.0)  # q4对η的一阶偏导

    d2p1 = 1.5 * xi  # p1对ξ的二阶偏导
    d2q1 = 1.5 * eta  # q1对η的二阶偏导
    d2p2 = 0.25 * x_size * (3.0 * xi - 1.0)  # p2对ξ的二阶偏导
    d2q2 = 0.25 * y_size * (3.0 * eta - 1.0)  # q2对η的二阶偏导
    d2p3 = -1.5 * xi  # p3对ξ的二阶偏导
    d2q3 = -1.5 * eta  # q3对η的二阶偏导
    d2p4 = 0.25 * x_size * (3.0 * xi + 1.0)  # p4对ξ的二阶偏导
    d2q4 = 0.25 * y_size * (3.0 * eta + 1.0)  # q4对η的二阶偏导

    d2x[0] = d2p1 * q1  # 形函数N1对ξ的二阶偏导
    d2x[1] = d2p2 * q1  # 形函数N2对ξ的二阶偏导
    d2x[2] = d2p1 * q2  # ……
    d2x[3] = d2p2 * q2
    d2x[4] = d2p1 * q3
    d2x[5] = d2p2 * q3
    d2x[6] = d2p1 * q4
    d2x[7] = d2p2 * q4
    d2x[8] = d2p3 * q3
    d2x[9] = d2p4 * q3
    d2x[10] = d2p3 * q4
    d2x[11] = d2p4 * q4
    d2x[12] = d2p3 * q1
    d2x[13] = d2p4 * q1
    d2x[14] = d2p3 * q2
    d2x[15] = d2p4 * q2
    d2y[0] = p1 * d2q1  # 形函数N1对η的二阶偏导
    d2y[1] = p2 * d2q1  # 形函数N2对η的二阶偏导
    d2y[2] = p1 * d2q2  # ……
    d2y[3] = p2 * d2q2
    d2y[4] = p1 * d2q3
    d2y[5] = p2 * d2q3
    d2y[6] = p1 * d2q4
    d2y[7] = p2 * d2q4
    d2y[8] = p3 * d2q3
    d2y[9] = p4 * d2q3
    d2y[10] = p3 * d2q4
    d2y[11] = p4 * d2q4
    d2y[12] = p3 * d2q1
    d2y[13] = p4 * d2q1
    d2y[14] = p3 * d2q2
    d2y[15] = p4 * d2q2
    d2xy[0] = dp1 * dq1  # 形函数N1对ξ和η的混合二阶偏导
    d2xy[1] = dp2 * dq1  # 形函数N2对ξ和η的混合二阶偏导
    d2xy[2] = dp1 * dq2  # ……
    d2xy[3] = dp2 * dq2
    d2xy[4] = dp1 * dq3
    d2xy[5] = dp2 * dq3
    d2xy[6] = dp1 * dq4
    d2xy[7] = dp2 * dq4
    d2xy[8] = dp3 * dq3
    d2xy[9] = dp4 * dq3
    d2xy[10] = dp3 * dq4
    d2xy[11] = dp4 * dq4
    d2xy[12] = dp3 * dq1
    d2xy[13] = dp4 * dq1
    d2xy[14] = dp3 * dq2
    d2xy[15] = dp4 * dq2
    return d2x, d2y, d2xy


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


def gauss_sample(element, gauss_points, weights):
    """
    生成单元局部坐标系下的高斯积分点和权系数
    :param element: 单元类型
    :param gauss_points: 局部坐标系下的高斯积分点
    :param weights: 高斯积分点的权系数
    :return: points, weights
    """
    num_integral_points = np.size(gauss_points, 0)
    if 'quad' in element:
        if num_integral_points == 1:
            gauss_points[0, 0] = 0.0
            gauss_points[0, 1] = 0.0
            weights[0] = 4.0
        elif num_integral_points == 4:
            root3 = 1.0 / np.sqrt(3)
            gauss_points[0, 0] = -root3
            gauss_points[0, 1] = root3
            gauss_points[1, 0] = root3
            gauss_points[1, 1] = root3
            gauss_points[2, 0] = -root3
            gauss_points[2, 1] = -root3
            gauss_points[3, 0] = root3
            gauss_points[3, 1] = -root3
            weights[:] = 1.0
        elif num_integral_points == 9:
            root06 = np.sqrt(0.6)
            gauss_points[0:7:3, 0] = -root06
            gauss_points[1:8:3, 0] = 0.0
            gauss_points[2:9:3, 0] = root06
            gauss_points[0:3, 1] = root06
            gauss_points[3:6, 1] = 0.0
            gauss_points[6:9, 1] = -root06
            weights[0] = 25/81
            weights[1] = 40/81
            weights[2] = 25/81
            weights[3] = 40/81
            weights[4] = 64/81
            weights[5] = 40/81
            weights[6] = 25/81
            weights[7] = 40/81
            weights[8] = 25/81
        elif num_integral_points == 16:
            gauss_points[0:13:4, 0] = -0.861136311594053
            gauss_points[1:14:4, 0] = -0.339981043584856
            gauss_points[2:15:4, 0] = 0.339981043584856
            gauss_points[3:16:4, 0] = 0.861136311594053
            gauss_points[0:4, 1] = 0.861136311594053
            gauss_points[4:8, 1] = 0.339981043584856
            gauss_points[8:12, 1] = -0.339981043584856
            gauss_points[12:16, 1] = -0.861136311594053
            weights[0] = 0.121002993285602
            weights[3] = weights[0]
            weights[12] = weights[0]
            weights[15] = weights[0]
            weights[1] = 0.226851851851852
            weights[2] = weights[1]
            weights[4] = weights[1]
            weights[7] = weights[1]
            weights[8] = weights[1]
            weights[11] = weights[1]
            weights[13] = weights[1]
            weights[14] = weights[1]
            weights[5] = 0.425293303010694
            weights[6] = weights[5]
            weights[9] = weights[5]
            weights[10] = weights[5]
    return gauss_points, weights


def global_to_axial(global_action, coord):
    """
    二维和三维桁架结构中，将整体坐标系下的力分量转换为杆的轴向力
    :param global_action: 整体坐标系下的力
    :param coord: 节点坐标
    :return: 轴向载荷axial
    """
    num_dim = np.size(coord, 1)
    add = 0.0
    for i in range(num_dim):
        add += (coord[1, i] - coord[0, i]) ** 2
    length = np.sqrt(add)
    axial = 0.0
    for i in range(num_dim):
        axial += (coord[1, i] - coord[0, i]) / length * global_action[num_dim+i]
    return axial


def global_to_local(local_action, global_action, gamma, coord):
    """
    二维和三维框架问题中，将整体坐标系的响应转换到局部坐标系
    :param local_action: 局部坐标系下的响应
    :param global_action: 整体坐标系下的响应
    :param gamma: 杆单元绕局部x轴的转角
    :param coord: 杆单元的节点坐标
    :return: 杆单元局部坐标系下的响应
    """
    num_dim = np.size(coord, 1)
    if num_dim == 2:
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cg = (x2 - x1) / length
        sg = (y2 - y1) / length
        local_action[0] = cg * global_action[0] + sg * global_action[1]
        local_action[1] = cg * global_action[1] - sg * global_action[0]
        local_action[2] = global_action[2]
        local_action[3] = cg * global_action[3] + sg * global_action[4]
        local_action[4] = cg * global_action[4] - sg * global_action[3]
        local_action[5] = global_action[5]
    elif num_dim == 3:
        t = transformation_matrix(coord, gamma)
        local_action = np.dot(t, global_action)
    return local_action


def hinge_reaction(coord, conv_elem_action, action, reaction, prop_ids,
                   prop, i_elem, elem_prop_type, gamma):
    """
    将超过塑性极限的弯矩重新分配以形成自平衡修正向量
    :param coord: 单元的节点坐标
    :param conv_elem_action: 上一步收敛时的单元载荷
    :param action: 单元响应
    :param reaction: 单元自平衡修正向量
    :param prop_ids: 材料属性号
    :param prop: 材料属性
    :param i_elem: 单元索引序号，从零开始（不是单元编号）
    :param elem_prop_type: 单元的材料属性编号
    :param gamma: 杆单元绕轴的转角
    :return: 单元自平衡修正向量reaction
    """
    num_dim = np.size(coord, 1)
    num_elem_dof = np.size(action, 0)
    local_action = np.zeros(num_elem_dof, dtype=np.float)
    total_action = np.zeros(num_elem_dof, dtype=np.float)
    total_action[:] = conv_elem_action[i_elem, :]  # 恢复上一步收敛时的响应
    bm1 = 0.0
    bm2 = 0.0
    bm3 = 0.0
    bm4 = 0.0
    if num_dim == 1:
        mpy = prop[prop_ids.index(elem_prop_type[i_elem]), 2]
        length = coord[1, 0] - coord[0, 0]
        s1 = total_action[1] + action[1]  # 单元的1号节点的转矩
        s2 = total_action[3] + action[3]  # 单元的2号节点的转矩
        if np.abs(s1) > mpy:
            if s1 > 0.0:
                bm1 = mpy - s1
            else:
                bm1 = -mpy - s1
        if np.abs(s2) > mpy:
            if s2 > 0.0:
                bm2 = mpy - s2
            else:
                bm2 = -mpy - s2
        reaction[0] = (bm1 + bm2) / length  # 修正力（力等于转矩/距离）
        reaction[1] = bm1  # 修正力矩
        reaction[2] = -reaction[0]
        reaction[3] = bm2
    elif num_dim == 2:
        mpy = prop[prop_ids.index(elem_prop_type[i_elem]), 3]
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        cs = (x2 - x1) / length
        sn = (y2 - y1) / length
        s1 = total_action[2] + action[2]  # 单元的1号节点的转矩
        s2 = total_action[5] + action[5]  # 单元的2号节点的转矩
        if np.abs(s1) > mpy:
            if s1 > 0.0:
                bm1 = mpy - s1
            else:
                bm1 = -mpy - s1
        if np.abs(s2) > mpy:
            if s2 > 0.0:
                bm2 = mpy - s2
            else:
                bm2 = -mpy - s2
        reaction[0] = -(bm1 + bm2) * sn / length  # x方向的修正力，局部坐标系下的修正力（垂直于轴线）乘以正弦
        reaction[1] = (bm1 + bm2) * cs / length  # y方向的修正力，局部坐标系下的修正力（垂直于轴线）乘以余弦
        reaction[2] = bm1
        reaction[3] = -reaction[0]
        reaction[4] = -reaction[1]
        reaction[5] = bm2
    elif num_dim == 3:
        gam = gamma[i_elem]
        mpy = prop[prop_ids.index(elem_prop_type[i_elem]), 6]
        mpz = prop[prop_ids.index(elem_prop_type[i_elem]), 7]
        mpx = prop[prop_ids.index(elem_prop_type[i_elem]), 8]
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        z1 = coord[0, 2]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        z2 = coord[1, 2]
        x21 = x2 - x1
        y21 = y2 - y1
        z21 = z2 - z1
        length = np.sqrt(z21 * z21 + y21 * y21 + x21 * x21)
        global_action = total_action + action
        local_action = global_to_local(local_action, global_action, gam, coord)
        global_action = np.zeros(num_elem_dof, dtype=np.float)
        s1 = local_action[4]  # 第5个分量，局部坐标系下单元1号节点的y方向转矩
        s2 = local_action[10]  # 第11个分量，局部坐标系下单元2号节点的y方向转矩
        if np.abs(s1) > mpy:
            if s1 > 0.0:
                bm1 = mpy - s1
            else:
                bm1 = -mpy - s1
        if np.abs(s2) > mpy:
            if s2 > 0.0:
                bm2 = mpy - s2
            else:
                bm2 = -mpy - s2
        local_action[2] = -(bm1 + bm2) / length  # 局部坐标系下单元1号节点z方向的修正力（x为长轴，z方向力产生y轴的矩）
        local_action[8] = -local_action[2]
        local_action[4] = bm1  # 局部坐标系下单元1号节点y方向的修正力矩
        local_action[10] = bm2
        s3 = local_action[5]  # 局部坐标系下单元1号节点的z方向转矩
        s4 = local_action[11]  # 局部坐标系下单元2号节点的z方向转矩
        if np.abs(s3) > mpz:
            if s3 > 0.0:
                bm3 = mpz - s3
            else:
                bm3 = -mpz - s3
        if np.abs(s4) > mpz:
            if s4 > 0.0:
                bm4 = mpz - s4
            else:
                bm4 = -mpz - s4
        local_action[1] = (bm3 + bm4) / length  # 局部坐标系下单元1号节点y方向的修正力
        local_action[7] = -local_action[1]
        local_action[5] = bm3  # 局部坐标系下单元1号节点y方向的修正力矩
        local_action[11] = bm4
        s5 = local_action[3]  # 单元的1号节点的x方向转矩
        if np.abs(s5) > mpx:
            if s5 > 0.0:
                global_action[3] = mpx - s5
            else:
                global_action[3] = -mpx - s5
        local_action[9] = -local_action[3]
        reaction = local_to_global(local_action, gam, coord)
    return reaction


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


def local_to_global(local_action, gamma, coord):
    """
    三维框架问题中，将局部坐标系的响应转换到整体坐标系
    :param local_action: 局部坐标系的响应
    :param gamma: 梁单元关于局部x坐标的转角
    :param coord: 梁单位2个节点的坐标
    :return: 全局坐标系下的响应
    """
    t = transformation_matrix(coord, gamma)
    global_action = np.dot(np.transpose(t), local_action)
    return global_action


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


def rigid_jointed(ke, prop_ids, prop, gamma, elem_prop_type, i_elem, coord):
    """
    生成一般梁单元的刚度矩阵
    :param ke: 刚度矩阵
    :param prop_ids: 材料属性号
    :param prop: 材料属性
    :param gamma: 单元绕轴的转角
    :param elem_prop_type: 单元的材料属性编号
    :param i_elem: 单元号（减一）
    :param coord: 单元的节点坐标
    :return: 刚度矩阵ke
    """
    num_dim = np.size(coord, 1)
    if num_dim == 1:
        prop_e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]
        prop_i = prop[prop_ids.index(elem_prop_type[i_elem]), 1]
        ei = prop_e * prop_i
        length = coord[1, 0] - coord[0, 0]
        ke[0, 0] = 12.0 * ei / (length ** 3)
        ke[2, 2] = ke[0, 0]
        ke[0, 2] = -ke[0, 0]
        ke[2, 0] = -ke[0, 0]
        ke[1, 1] = 4.0 * ei / length
        ke[3, 3] = ke[1, 1]
        ke[0, 1] = 6.0 * ei / (length * length)
        ke[1, 0] = ke[0, 1]
        ke[0, 3] = ke[0, 1]
        ke[3, 0] = ke[0, 1]
        ke[1, 2] = -ke[0, 1]
        ke[2, 1] = -ke[0, 1]
        ke[2, 3] = -ke[0, 1]
        ke[3, 2] = -ke[0, 1]
        ke[1, 3] = 2.0 * ei / length
        ke[3, 1] = ke[1, 3]
    elif num_dim == 2:
        prop_e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]
        prop_a = prop[prop_ids.index(elem_prop_type[i_elem]), 1]
        prop_i = prop[prop_ids.index(elem_prop_type[i_elem]), 2]
        ea = prop_e * prop_a
        ei = prop_e * prop_i
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        cos_alpha = (x2 - x1) / length
        sin_alpha = (y2 - y1) / length
        e1 = ea / length
        e2 = 12.0 * ei / (length * length * length)
        e3 = ei / length
        e4 = 6.0 * ei / (length * length)
        ke[0, 0] = cos_alpha * cos_alpha * e1 + sin_alpha * sin_alpha * e2
        ke[3, 3] = ke[0, 0]
        ke[0, 1] = sin_alpha * cos_alpha * (e1 - e2)
        ke[1, 0] = ke[0, 1]
        ke[3, 4] = ke[0, 1]
        ke[4, 3] = ke[0, 1]
        ke[0, 2] = -sin_alpha * e4
        ke[2, 0] = ke[0, 2]
        ke[5, 0] = ke[0, 2]
        ke[0, 5] = ke[0, 2]
        ke[2, 3] = sin_alpha * e4
        ke[3, 2] = ke[2, 3]
        ke[3, 5] = ke[2, 3]
        ke[5, 3] = ke[2, 3]
        ke[0, 3] = -ke[0, 0]
        ke[3, 0] = ke[0, 3]
        ke[0, 4] = sin_alpha * cos_alpha * (e2 - e1)
        ke[4, 0] = ke[0, 4]
        ke[1, 3] = ke[0, 4]
        ke[3, 1] = ke[0, 4]
        ke[1, 1] = sin_alpha * sin_alpha * e1 + cos_alpha * cos_alpha * e2
        ke[4, 4] = ke[1, 1]
        ke[4, 1] = -ke[1, 1]
        ke[1, 4] = ke[4, 1]
        ke[1, 2] = cos_alpha * e4
        ke[2, 1] = ke[1, 2]
        ke[1, 5] = ke[1, 2]
        ke[5, 1] = ke[1, 2]
        ke[2, 2] = 4.0 * e3
        ke[5, 5] = ke[2, 2]
        ke[2, 4] = -cos_alpha * e4
        ke[4, 2] = ke[2, 4]
        ke[4, 5] = ke[2, 4]
        ke[5, 4] = ke[2, 4]
        ke[2, 5] = 2.0 * e3
        ke[5, 2] = ke[2, 5]
    elif num_dim == 3:
        prop_e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]
        prop_a = prop[prop_ids.index(elem_prop_type[i_elem]), 1]
        iy = prop[prop_ids.index(elem_prop_type[i_elem]), 2]
        iz = prop[prop_ids.index(elem_prop_type[i_elem]), 3]
        prop_g = prop[prop_ids.index(elem_prop_type[i_elem]), 4]
        prop_j = prop[prop_ids.index(elem_prop_type[i_elem]), 5]
        ea = prop_e * prop_a
        eiy = prop_e * iy
        eiz = prop_e * iz
        gj = prop_g * prop_j
        t = transformation_matrix(coord, gamma[i_elem])
        ke = np.zeros(ke.shape, dtype=np.float)
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        z1 = coord[0, 2]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        z2 = coord[1, 2]
        x21 = x2 - x1
        y21 = y2 - y1
        z21 = z2 - z1
        length = np.sqrt(x21 * x21 + y21 * y21 + z21 * z21)
        a1 = ea / length
        a2 = 12.0 * eiz / (length * length * length)
        a3 = 12.0 * eiy / (length * length * length)
        a4 = 6.0 * eiz / (length * length)
        a5 = 6.0 * eiy / (length * length)
        a6 = 4.0 * eiz / length
        a7 = 4.0 * eiy / length
        a8 = gj / length
        ke[0, 0] = a1
        ke[6, 6] = a1
        ke[0, 6] = -a1
        ke[6, 0] = -a1
        ke[1, 1] = a2
        ke[7, 7] = a2
        ke[1, 7] = -a2
        ke[7, 1] = -a2
        ke[2, 2] = a3
        ke[8, 8] = a3
        ke[2, 8] = -a3
        ke[8, 2] = -a3
        ke[3, 3] = a8
        ke[9, 9] = a8
        ke[3, 9] = -a8
        ke[9, 3] = -a8
        ke[4, 4] = a7
        ke[10, 10] = a7
        ke[4, 10] = 0.5 * a7
        ke[10, 4] = 0.5 * a7
        ke[5, 5] = a6
        ke[11, 11] = a6
        ke[5, 11] = 0.5 * a6
        ke[11, 5] = 0.5 * a6
        ke[1, 5] = a4
        ke[5, 1] = a4
        ke[1, 11] = a4
        ke[11, 1] = a4
        ke[5, 7] = -a4
        ke[7, 5] = -a4
        ke[7, 11] = -a4
        ke[11, 7] = -a4
        ke[4, 8] = a5
        ke[8, 4] = a5
        ke[8, 10] = a5
        ke[10, 8] = a5
        ke[2, 4] = -a5
        ke[4, 2] = -a5
        ke[2, 10] = -a5
        ke[10, 2] = -a5
        ke = np.dot(np.dot(np.transpose(t), ke), t)
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


def transformation_matrix(coord, gamma):
    """
    梁单元坐标变换时的转换矩阵。有如下关系式：
        K = transpose(t) * k * t
        M = transpose(t) * m * t
        F = transpose(t) * f

    转换矩阵的各个元素为局部坐标系各轴与整体坐标系各轴的余弦

    局部坐标系上x轴上的向量叉乘整体坐标系的y轴上的向量，可得局部坐标系z轴上的向量
    局部坐标系z轴上的向量叉乘局部坐标系x轴上的向量，可得局部坐标系y轴上的向量
    最后再绕局部坐标系的x轴旋转gamma度，得到最后的局部坐标系
    （由罗德里格旋转公式可得绕局部坐标系x轴旋转后的局部坐标系）
    局部坐标系的x方向单位向量为(x21/ell, y21/ell, z21/ell)

    :param coord: 梁单元2个节点的坐标
    :param gamma: 梁单元绕局部x轴的转角
    :return: 坐标变换矩阵t
    """
    r0 = np.zeros((3, 3), dtype=np.float)
    t = np.zeros((12, 12), dtype=np.float)
    x1 = coord[0, 0]
    y1 = coord[0, 1]
    z1 = coord[0, 2]
    x2 = coord[1, 0]
    y2 = coord[1, 1]
    z2 = coord[1, 2]
    x21 = x2 - x1
    y21 = y2 - y1
    z21 = z2 - z1
    length = np.sqrt(x21 * x21 + y21 * y21 + z21 * z21)
    pi = np.pi
    gam_rad = gamma * pi / 180.
    cg = np.cos(gam_rad)
    sg = np.sin(gam_rad)
    den = length * np.sqrt(x21 * x21 + z21 * z21)
    if den != 0.0:  # 单元不垂直于xz平面
        r0[0, 0] = x21 / length  # cos(x, X)
        r0[0, 1] = y21 / length  # cos(x, Y)
        r0[0, 2] = z21 / length  # cos(x, Z)
        r0[1, 0] = (-x21 * y21 * cg - length * z21 * sg) / den
        r0[1, 1] = den * cg / (length * length)
        r0[1, 2] = (-y21 * z21 * cg + length * x21 * sg) / den
        r0[2, 0] = (x21 * y21 * sg - length * z21 * cg) / den
        r0[2, 1] = -den * sg / (length * length)
        r0[2, 2] = (y21 * z21 * sg + length * x21 * cg) / den
    else:  # den为0说明单元垂直于xz平面
        r0[0, 0] = 0.0
        r0[0, 2] = 0.0
        r0[1, 1] = 0.0
        r0[2, 1] = 0.0
        r0[0, 1] = 1.0
        r0[1, 0] = -cg
        r0[2, 2] = cg
        r0[1, 2] = sg
        r0[2, 0] = sg
    for i in range(3):
        for j in range(3):
            x = r0[i, j]
            for k in range(0, 10, 3):
                t[i + k, j + k] = x
    return t


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
