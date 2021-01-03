# coding: utf-8
# author: xuxc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from utility import (get_name, read_elem_prop_type, read_parameter, initialize_node_dof,
                     update_node_dof, get_elem_dof, form_k_diag, pin_jointed,
                     form_sparse_v, add_force, add_displacement, sparse_cho_fac,
                     sparse_cho_bac, contour_plot)  # noqa: E402


def bar(filename=None, plot=True):
    input_file, output_file, vtk_file = get_name(filename)
    if not os.path.exists(input_file):
        print('{}不存在！'.format(input_file))
        return

    num_prop_types = None
    prop_ids = None
    prop = None
    num_dim = None
    num_node = None
    num_node_dof = None
    node_coord = None
    num_elem = None
    elem_ids = None
    elem_connections = None
    elem_prop_type = None
    num_node_on_elem = None
    node_ids = None
    fixed_ids = None
    fixed_components = None
    num_loaded_nodes = None
    loaded_node_ids = None
    forces = None
    num_disp_node = None
    disp_node_ids = None
    displacements = None

    penalty = 1.0e20  # 处理位移约束时使用的大数

    # ********************************** 解析输入文件 ********************************** #
    fr = open(input_file, 'r', encoding='utf-8')
    line = fr.readline()
    while line:
        line = line.strip().lower()
        if line.startswith('#') or line == '':
            pass
        elif line == '* material':
            # 材料种类数、材料分量数、材料号、材料属性
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* node':
            # 节点总数、问题维数、节点编号、节点坐标
            num_node, num_dim, node_ids, node_coord = read_parameter(fr, 'float')
        elif line == '* element':
            # 单元数、单元节点数、单元编号、单元节点
            num_elem, num_node_on_elem, elem_ids, elem_connections = read_parameter(fr, 'int')
            # 每个单元的材料属性
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* constraint':
            # 约束节点数、节点自由度数、约束节点编号、约束节点自由度
            num_fixed_node, num_node_dof, fixed_ids, fixed_components = read_parameter(fr, 'int')
            assert num_node_dof == num_dim, '杆单元的节点自由度应该等于问题维数'
        elif line == '* force':
            # 受载节点数、节点自由度数、受载节点编号、载荷大小
            num_loaded_nodes, num_node_dof, loaded_node_ids, forces = read_parameter(fr, 'float')
            assert num_node_dof == num_dim, '杆单元的载荷分量应该等于问题维数'
        elif line == '* displacement':
            # 受位移约束节点数、节点自由度数、受位移约束的节点编号，位移大小
            num_disp_node, num_node_dof, disp_node_ids, displacements = read_parameter(fr, 'float')
        else:
            print('无效的关键字：{}'.format(line.strip()))
            return
        line = fr.readline()
    fr.close()

    # ******************************** 生成节点自由度矩阵 ******************************* #
    num_elem_dof = num_node_on_elem * num_node_dof  # 单元自由度数
    # 自由度矩阵
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_ids, fixed_components)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_equation = np.max(node_dof)  # 方程个数，也是最大自由度编号
    k_diag = np.zeros(num_equation, dtype=np.int)  # 对角线元素定位向量
    # 单元定位向量，用于获取单元中各节点自由度是整体的第几号自由度
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=np.int)
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
        global_elem_dof[i_elem, :] = elem_dof
        k_diag = form_k_diag(k_diag, elem_dof)
    for i in range(1, num_equation):
        k_diag[i] += k_diag[i-1]

    # ********************************** 组装刚度矩阵 ********************************** #
    kv = np.zeros(k_diag[-1], dtype=np.float)
    ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_node_index = [node_ids.index(i) for i in elem_conn]
        coord = node_coord[elem_node_index, :]
        ke = pin_jointed(ke, prop[prop_ids.index(elem_prop_type[i_elem]), 0],
                         prop[prop_ids.index(elem_prop_type[i_elem]), 1], coord)
        elem_dof = global_elem_dof[i_elem, :]
        kv = form_sparse_v(kv, ke, elem_dof, k_diag)

    # ****************************** 根据边界条件更新刚度矩阵 **************************** #
    loads = add_force(num_equation, node_ids, num_node_dof, node_dof, num_loaded_nodes,
                      loaded_node_ids, forces)
    add_displacement(num_disp_node, num_node_dof, disp_node_ids, displacements,
                     node_ids, kv, k_diag, node_dof, loads, penalty)

    # ************************************ 方程求解 *********************************** #
    kv = sparse_cho_fac(kv, k_diag)
    loads = sparse_cho_bac(kv, loads, k_diag)

    # *********************************** 结果输出 ************************************ #
    fw_out = open(output_file, 'w', encoding='utf-8')
    fw_vtk = open(vtk_file, 'w', encoding='utf-8')

    fw_out.write('总共有{}个方程，存储带宽为{}\n'.format(num_equation, k_diag[-1]))

    fw_vtk.write('# vtk DataFile Version 3.0\n')
    fw_vtk.write('Bar Elements\n')
    fw_vtk.write('ASCII\n\n')

    fw_vtk.write('DATASET UNSTRUCTURED_GRID\n\n')

    fw_vtk.write('POINTS {} float\n'.format(num_node))
    for i in range(num_node):
        for j in range(3):
            if j < num_node_dof:
                fw_vtk.write('{:16.6E} '.format(node_coord[i, j]))
            else:
                fw_vtk.write('{:16.6E} '.format(0))
        fw_vtk.write('\n')

    fw_vtk.write('\nCELLS {} {}\n'.format(num_elem, num_elem * (num_node_on_elem + 1)))
    for i in range(num_elem):
        fw_vtk.write('{} {} {}\n'.format(
            num_node_on_elem, node_ids.index(elem_connections[i, 0]),
            node_ids.index(elem_connections[i, 1])))

    fw_vtk.write('\nCELL_TYPES {}\n'.format(num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('3\n')

    fw_out.write('节点    位移\n')
    fw_vtk.write('\nPOINT_DATA {}\n'.format(num_node))
    fw_vtk.write('VECTORS Displacement float\n')
    for i in range(num_node):
        fw_out.write('{:4d} '.format(node_ids[i]))
        for j in range(num_node_dof):
            fw_out.write('{:16.6e} '.format(loads[node_dof[i, j]]))
        for j in range(3):
            if j < num_node_dof:
                fw_vtk.write('{:16.6e} '.format(loads[node_dof[i, j]]))
            else:
                fw_vtk.write('{:16.6e} '.format(0))
        fw_vtk.write('\n')
        fw_out.write('\n')
    fw_out.write('单元    载荷\n')
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_node_index = [node_ids.index(i) for i in elem_conn]
        coord = node_coord[elem_node_index, :]
        ke = pin_jointed(ke, prop[prop_ids.index(elem_prop_type[i_elem]), 0],
                         prop[prop_ids.index(elem_prop_type[i_elem]), 1], coord)
        elem_dof = global_elem_dof[i_elem, :]
        elem_disp = loads[elem_dof]
        action = np.dot(ke, elem_disp)
        fw_out.write('{:4d} '.format(elem_ids[i_elem]))
        for i in action:
            fw_out.write('{:16.6e} '.format(i))
        fw_out.write('\n')

    fw_vtk.close()
    fw_out.close()

    # *********************************** 显示结果 ************************************ #
    if plot:
        contour_plot(vtk_file)


if __name__ == '__main__':
    bar('4.2_test1', plot=True)
    bar('4.2_test2', plot=True)
    bar('4.2_test3', plot=False)
    bar('4.2_test4', plot=False)
