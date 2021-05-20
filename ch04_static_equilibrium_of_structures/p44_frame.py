# coding: utf-8
# author: xuxc
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from utility import (get_name, read_parameter, read_elem_prop_type, read_beam_direction,
                     initialize_node_dof, update_node_dof, get_elem_dof, form_k_diag,
                     rigid_jointed, form_sparse_v, add_force, add_displacement,
                     sparse_cho_fac, sparse_cho_bac, contour_plot)  # noqa: E402


def frame(filename=None, plot=False, field_name='Displacement', component=-1,
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
    num_node_dof = None
    num_node_on_elem = None
    num_elem = None
    elem_ids = None
    elem_prop_type = None
    direction = None
    elem_connections = None
    fixed_ids = None
    fixed_components = None
    num_loaded_nodes = None
    loaded_node_ids = None
    forces = None
    num_disp_nodes = None
    disp_node_ids = None
    displacements = None

    penalty = 1.0e20

    # ********************************** 解析输入文件 ********************************** #
    fr = open(input_file, 'r', encoding='utf-8')
    line = fr.readline()
    while line:
        line = line.strip().lower()
        if line.startswith('#') or line == '':
            pass  # 跳过注释行和空行
        elif line == '* material':
            # 材料种类数、材料分量数、材料编号、材料属性
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* node':
            # 节点数、问题维数、节点编号、节点坐标
            num_node, num_dim, node_ids, node_coord = read_parameter(fr, 'float')
        elif line == '* element':
            # 单元数、单元节点数、单元编号、单元连接关系（单元节点编号）
            num_elem, num_node_on_elem, elem_ids, elem_connections = read_parameter(fr, 'int')
            # 每个单元的材料属性
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
            # 单元方向（用角度表示）
            direction = read_beam_direction(num_dim, num_elem, fr)
        elif line == '* constraint':
            # 固定节点数、节点自由度数、固定节点编号、固定的自由度分量
            num_fixed_nodes, num_node_dof, fixed_ids, fixed_components = read_parameter(fr, 'int')
        elif line == '* force':
            # 受载节点数、节点自由度数、受载节点编号、载荷
            num_loaded_nodes, num_node_dof, loaded_node_ids, forces = read_parameter(fr, 'float')
        elif line == '* displacement':
            # 位移约束节点数、节点自由度数、位移约束节点编号、位移大小
            num_disp_nodes, num_node_dof, disp_node_ids, displacements = read_parameter(fr, 'float')
        else:
            print('无效的关键字：{}'.format(line))
            return
        line = fr.readline()
    fr.close()

    # ******************************** 生成节点自由度矩阵 ******************************* #
    # 节点自由度矩阵
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_ids, fixed_components)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_elem_dof = num_node_on_elem * num_node_dof
    num_equation = node_dof.max()
    # 对角线定位向量
    k_diag = np.zeros(num_equation, dtype=int)
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=int)
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
        global_elem_dof[i_elem, :] = elem_dof
        k_diag = form_k_diag(k_diag, elem_dof)
    for i in range(1, num_equation):
        k_diag[i] += k_diag[i-1]

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

    # ****************************** 根据边界条件更新刚度矩阵 **************************** #
    loads = add_force(num_equation, node_ids, num_node_dof, node_dof, num_loaded_nodes,
                      loaded_node_ids, forces)
    kv, loads = add_displacement(num_disp_nodes, num_node_dof, disp_node_ids, displacements,
                                 node_ids, kv, k_diag, node_dof, loads, penalty)

    # ************************************ 方程求解 *********************************** #
    kv = sparse_cho_fac(kv, k_diag)
    loads = sparse_cho_bac(kv, loads, k_diag)

    # *********************************** 结果输出 ************************************ #
    fw_out = open(output_file, 'w', encoding='utf-8')
    fw_vtk = open(vtk_file, 'w', encoding='utf-8')

    fw_out.write('总共有{}个方程，存储带宽为{}\n'.format(num_equation, k_diag[-1]))

    fw_vtk.write('# vtk DataFile Version 3.0\n')
    fw_vtk.write('Beam Elements\n')
    fw_vtk.write('ASCII\n\n')

    fw_vtk.write('DATASET UNSTRUCTURED_GRID\n\n')

    fw_vtk.write('POINTS {} float\n'.format(num_node))
    for i in range(num_node):
        for j in range(3):
            if j < num_dim:
                fw_vtk.write('{:16.9E} '.format(node_coord[i, j]))
            else:
                fw_vtk.write('{:16.9E} '.format(0))
        fw_vtk.write('\n')

    fw_vtk.write('\nCELLS {} {}\n'.format(num_elem, num_elem * (num_node_on_elem + 1)))
    for i in range(num_elem):
        fw_vtk.write('{} {} {}\n'.format(
            num_node_on_elem, node_ids.index(elem_connections[i, 0]),
            node_ids.index(elem_connections[i, 1])))

    fw_vtk.write('\nCELL_TYPES {}\n'.format(num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('3\n')

    fw_out.write(' 节点{0}位移{0}{1}转角\n'.format(' '*8*num_dim, ' '*7*(num_node_dof-num_dim)))
    fw_vtk.write('\nPOINT_DATA {}\n'.format(num_node))
    fw_vtk.write('VECTORS Displacement float\n')
    for i in range(num_node):
        fw_out.write('{:4d} '.format(node_ids[i]))
        for j in range(num_node_dof):
            fw_out.write('{:16.9e} '.format(loads[node_dof[i, j]]))
        fw_out.write('\n')

        if num_dim == 1:
            fw_vtk.write('{0:16.9E} {1:16.9E} {0:16.9E}\n'.format(0, loads[node_dof[i, 0]]))
        if num_dim == 2:
            fw_vtk.write('{:16.9E} {:16.9E} {:16.9E}\n'.format(
                loads[node_dof[i, 0]], loads[node_dof[i, 1]], 0.0))
        if num_dim == 3:
            fw_vtk.write('{:16.9E} {:16.9E} {:16.9E}\n'.format(
                loads[node_dof[i, 0]], loads[node_dof[i, 1]], loads[node_dof[i, 2]]))

    fw_out.write('单元{0}力{0}{1}力矩\n'.format(' '*8*num_dim, ' '*7*(num_node_dof-num_dim)))
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_node_index = [node_ids.index(i) for i in elem_conn]
        coord = node_coord[elem_node_index, :]
        ke = rigid_jointed(ke, prop_ids, prop, direction, elem_prop_type, i_elem, coord)
        elem_dof = global_elem_dof[i_elem, :]
        elem_disp = loads[elem_dof]
        action = np.dot(ke, elem_disp)
        fw_out.write('{:4d} '.format(elem_ids[i_elem]))
        for i in range(num_elem_dof):
            fw_out.write('{:16.9e} '.format(action[i]))
            if (i+1) % num_node_dof == 0:
                if (i+1) % num_elem_dof == 0:
                    fw_out.write('\n')
                else:
                    fw_out.write('\n     ')

    fw_vtk.close()
    fw_out.close()

    # *********************************** 显示结果 ************************************ #
    if plot:
        contour_plot(vtk_file, field_name, component, shape_name, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame element')
    parser.add_argument('--filename', action='store',
                        dest='filename', default='4.4_test1', type=str)
    parser.add_argument('--plot', '-p', action='store',
                        dest='plot', default=True, type=bool)
    parser.add_argument('--field', action='store', dest='field',
                        default='Displacement', type=str)
    parser.add_argument('--component', action='store',
                        dest='component', default=-1, type=int)
    parser.add_argument('--shape_name', action='store', dest='shape_name',
                        default='Displacement', type=str)
    parser.add_argument('--factor', action='store', dest='factor',
                        default=1, type=int)
    given_args = parser.parse_args()
    file = given_args.filename
    show = given_args.plot
    field = given_args.field
    comp = given_args.component
    shape = given_args.shape_name
    scale = given_args.factor
    frame(file, show, field, comp, shape, scale)
