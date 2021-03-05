# coding: utf-8
# author: xuxc
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from utility import (get_name, read_parameter, read_elem_prop_type, read_beam_direction,
                     read_load_increments, initialize_node_dof, update_node_dof,
                     get_elem_dof, form_k_diag, rigid_jointed, form_sparse_v,
                     add_force, sparse_cho_fac, sparse_cho_bac, hinge_reaction,
                     contour_plot)  # noqa: E402


def nonlinear_frame(filename=None, plot=False, field_name='Displacement', component=-1,
                    shape_name='Displacement', factor=1):
    input_file, output_file, vtk_file = get_name(filename)
    if not os.path.exists(input_file):
        print('{}不存在！'.format(input_file))
        return

    num_dim = None
    num_prop_types = None
    prop_ids = None
    prop = None
    num_node = None
    node_ids = None
    node_coord = None
    num_node_on_elem = None
    num_node_dof = None
    num_elem = None
    elem_connections = None
    elem_prop_type = None
    direction = None
    fixed_ids = None
    fixed_components = None
    num_loaded_nodes = None
    loaded_node_ids = None
    forces = None
    loads = None
    increments = None
    iteration = None
    max_iter = None
    tolerance = None

    # ********************************** 解析输入文件 ********************************** #
    fr = open(input_file, 'r', encoding='utf-8')
    line = fr.readline()
    while line:
        line = line.strip().lower()
        if line.startswith('#') or line == '':
            pass  # 跳过注释行和空行
        elif line == '* material':
            # 材料种类数、材料分量数、材料编号、材料属性
            # 一维问题的材料分量为：E、I、Mp
            # 二维问题的材料分量为：E、A、I、Mp
            # 三维问题的材料分量为：E、A、Iy、Iz、G、J、Mpy、Mpz、Mpx
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* node':
            # 节点数、问题维数、节点编号、节点坐标
            num_node, num_dim, node_ids, node_coord = read_parameter(fr, 'float')
        elif line == '* element':
            # 单元数、单元节点数、单元编号、单元连接关系
            num_elem, num_node_on_elem, elem_ids, elem_connections = read_parameter(fr, 'int')
            # 每个单元的材料属性
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
            direction = read_beam_direction(num_dim, num_elem, fr)
        elif line == '* constraint':
            # 固定节点数、节点自由度数、固定节点编号、固定的自由度分量
            num_fixed_nodes, num_node_dof, fixed_ids, fixed_components = read_parameter(fr, 'int')
        elif line == '* force':
            # 受载节点数、节点自由度数、受载节点编号、载荷
            num_loaded_nodes, num_node_dof, loaded_node_ids, forces = read_parameter(fr, 'float')
        elif line == '* max_iter':
            max_iter = int(fr.readline())
        elif line == '* tolerance':
            tolerance = float(fr.readline())
        elif line == '* increments':
            increments = read_load_increments(fr)  # 每一步的载荷增量
        else:
            print('无效的关键字：{}'.format(line))
            return
        line = fr.readline()
    fr.close()

    fw_out = open(output_file, 'w', encoding='utf-8')
    fw_vtk = open(vtk_file, 'w', encoding='utf-8')
    fw_vtk.write('# vtk DataFile Version 3.0\n')
    fw_vtk.write('Plastic Joint Frame\n')
    fw_vtk.write('ASCII\n')

    fw_vtk.write('\nDATASET UNSTRUCTURED_GRID\n')

    fw_vtk.write('\nPOINTS {} float\n'.format(num_node))
    for i in range(num_node):
        for j in range(3):
            if j < num_dim:
                fw_vtk.write('{:16.9E} '.format(node_coord[i, j]))
            else:
                fw_vtk.write('{:16.9E} '.format(0))
        fw_vtk.write('\n')

    fw_vtk.write('\nCELLS {} {}\n'.format(num_elem, num_elem * (num_node_on_elem+1)))
    for i in range(num_elem):
        fw_vtk.write('{} '.format(num_node_on_elem))
        for j in range(num_node_on_elem):
            fw_vtk.write('{} '.format(node_ids.index(elem_connections[i, j])))
        fw_vtk.write('\n')

    fw_vtk.write('\nCELL_TYPES {}\n'.format(num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('3\n')

    fw_vtk.write('\nPOINT_DATA {}\n'.format(num_node))

    # ******************************** 生成节点自由度矩阵 ******************************* #
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_ids, fixed_components)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_elem_dof = num_node_dof * num_node_on_elem
    num_equation = node_dof.max()
    k_diag = np.zeros(num_equation, dtype=np.int)  # 对角元素定位向量
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=np.int)
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
        global_elem_dof[i_elem, :] = elem_dof
        k_diag = form_k_diag(k_diag, elem_dof)
    for i in range(1, num_equation):
        k_diag[i] += k_diag[i-1]

    fw_out.write('总共有{}个方程，存储带宽为{}\n'.format(num_equation, k_diag[-1]))

    # ********************************** 组装刚度矩阵 ********************************** #
    kv = np.zeros(k_diag[-1], dtype=np.float)  # 整体刚度矩阵
    ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 单元刚度矩阵
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_node_index = [node_ids.index(i) for i in elem_conn]
        coord = node_coord[elem_node_index, :]
        ke = rigid_jointed(ke, prop_ids, prop, direction, elem_prop_type, i_elem, coord)
        elem_dof = global_elem_dof[i_elem, :]
        kv = form_sparse_v(kv, ke, elem_dof, k_diag)

    # ******************************************************************************* #
    # ****************************** 系数矩阵的Cholesky分解 **************************** #
    # 该算法中，刚度矩阵仅形成一次，非线性的引入是通过迭代不断修改内力实现的
    sparse_cho_fac(kv, k_diag)

    # ******************************************************************************* #
    # *********************************** 载荷增量循环 ******************************** #
    total_load = 0.0  # 已加载的载荷
    conv_elem_action = np.zeros((num_elem, num_elem_dof), dtype=np.float)  # 收敛时的单元相应
    total_node_dis = np.zeros(num_equation+1, dtype=np.float)  # 累积的节点位移

    steps = len(increments)
    for step in range(steps):
        total_load += increments[step]
        fw_out.write('\n载荷步：{} 加载因子：{:g}\n'.format(step+1, total_load))
        old_node_dis = np.zeros(num_equation+1, dtype=np.float)  # 上一步的节点位移
        correction_loads = np.zeros(num_equation+1, dtype=np.float)  # 内部修正载荷

        # 每个载荷步，使用的是载荷的增量，获取的也是位移的增量
        for iteration in range(1, max_iter+1):
            # # 获取当前的载荷增量：
            loads = add_force(num_equation, node_ids, num_node_dof, node_dof, num_loaded_nodes,
                              loaded_node_ids, forces*increments[step])
            loads += correction_loads  # 根据前一步结果，修正单元内部载荷
            correction_loads[:] = 0.0  # 修正载荷归零
            loads = sparse_cho_bac(kv, loads, k_diag)  # 求解后loads为位移

            # 判断是否收敛
            # 无论是否收敛，都需要对内部载荷进行修正
            converged = np.abs(loads-old_node_dis).max() / np.abs(loads).max() <= tolerance
            old_node_dis = loads

            for i_elem in range(num_elem):
                elem_conn = elem_connections[i_elem, :]
                elem_node_index = [node_ids.index(i) for i in elem_conn]
                coord = node_coord[elem_node_index, :]
                elem_dof = global_elem_dof[i_elem, :]
                elem_dis = loads[elem_dof]
                ke = rigid_jointed(ke, prop_ids, prop, direction, elem_prop_type, i_elem, coord)
                action = np.dot(ke, elem_dis)  # 单元响应，包含单元各节点的力和力矩信息
                reaction = np.zeros(num_elem_dof, dtype=np.float)  # 单元自平衡修正向量
                if max_iter != 1:  # max_iter为1时不存在两次计算结果，不能修正，不处理该情况
                    # 检查总的响应是否超过塑性弯矩，并返回自平衡修正向量
                    # 其中，修正向量的力矩为超出塑性极限的力矩，修正向量的力用于平衡修正向量的弯矩
                    # 如果未超过塑性弯矩，则修正向量为0
                    reaction = hinge_reaction(coord, conv_elem_action, action, reaction, prop_ids,
                                              prop, i_elem, elem_prop_type, direction)
                    # 通过自平衡向量形成修正载荷
                    correction_loads[elem_dof] -= reaction
                    correction_loads[0] = 0.0
                if iteration == max_iter or converged:
                    conv_elem_action[i_elem, :] += reaction[:] + action[:]

            # 收敛后打断当前载荷步内的循环
            if converged:
                break

        total_node_dis += loads

        if num_dim == 2:
            fw_out.write('节点             位移                转角\n')
        elif num_dim == 3:
            fw_out.write('节点                    位移                                  转角\n')
        for i in range(num_node):
            fw_out.write('{:4d} '.format(node_ids[i]))
            for j in range(num_node_dof):
                fw_out.write('{:12.4E} '.format(total_node_dis[node_dof[i, j]]))
            fw_out.write('\n')
        fw_out.write('迭代{:g}步后收敛\n'.format(iteration))

        fw_vtk.write('\nVECTORS Displacement_{:g} float\n'.format(total_load))
        for i in range(num_node):
            if num_dim == 1:
                fw_vtk.write('{0:12.4E} {1:12.4E} {0:12.4E}\n'.format(0, loads[node_dof[i, 0]]))
            elif num_dim == 2:
                fw_vtk.write('{:12.4E} {:12.4E} {:12.4E}\n'.format(
                    loads[node_dof[i, 0]], loads[node_dof[i, 1]], 0))
            elif num_dim == 3:
                fw_vtk.write('{:12.4E} {:12.4E} {:12.4E}\n'.format(
                    loads[node_dof[i, 0]], loads[node_dof[i, 1]], loads[node_dof[i, 2]]))

        # 如果迭代次数大于设置的上限，说明不能收敛，打断整个大循环
        # 由于每个step至少需要计算一次才能判断是否收敛，所以max_iter为1时，不能判断为超过迭代次数上限
        if iteration == max_iter and max_iter != 1:
            break

    fw_out.close()
    fw_vtk.close()

    # *********************************** 显示结果 ************************************ #
    if plot:
        contour_plot(vtk_file, field_name, component, shape_name, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nonlinear Frame Element')
    parser.add_argument('-i', '--input', action='store',
                        dest='input_file', default='4.5_test1', type=str)
    parser.add_argument('-p', '--plot', action='store',
                        dest='plot', default=False, type=bool)
    parser.add_argument('-f', '--field', action='store', dest='field',
                        default='Displacement', type=str)
    parser.add_argument('-c', '--component', action='store',
                        dest='component', default=-1, type=int)
    parser.add_argument('--shape', action='store', dest='shape_name',
                        default='Displacement', type=str)
    parser.add_argument('-s', '--scale', action='store', dest='scale',
                        default=1, type=int)
    given_args = parser.parse_args()
    inp_file = given_args.input_file
    show = given_args.plot
    field = given_args.field
    comp = given_args.component
    shape = given_args.shape_name
    scale = given_args.scale
    nonlinear_frame(inp_file, show, field, comp, shape, scale)
