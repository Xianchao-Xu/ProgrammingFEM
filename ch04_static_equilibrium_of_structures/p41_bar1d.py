# coding: utf-8
# author: xuxc
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from utility import (get_name, read_parameter, read_parameter_1d, read_elem_prop_type,
                     initialize_node_dof, update_node_dof, get_elem_dof, form_k_diag,
                     rod_ke, form_sparse_v, add_force, add_displacement,
                     sparse_cho_fac, sparse_cho_bac, contour_plot)  # noqa: E402


def bar1d(filename=None, plot=True, field_name='Displacement', component=-1,
          shape_name='Displacement', factor=1):
    input_file, output_file, vtk_file = get_name(filename)
    if not os.path.exists(input_file):
        print('{}不存在！'.format(input_file))
        return

    num_prop_types = None
    prop_ids = None
    num_elem = None
    fixed_node_ids = None
    fixed_components = None
    prop = None
    elem_prop_type = None
    elem_length = None
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
            pass  # 跳过空行和注释行
        elif line == '* material':
            # 材料种类数、材料分量数、材料号、材料属性
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* length':
            num_elem = int(fr.readline())  # 单元数
            elem_length = np.zeros(num_elem, dtype=np.float)  # 单元长度
            for i in range(num_elem):
                elem_length[i] = fr.readline()
            # 每个单元的材料属性
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* constraint':
            # 约束节点数、约束节点编号、约束节点自由度
            num_fixed_node, fixed_node_ids, fixed_components = read_parameter_1d(fr, 'int')
        elif line == '* force':
            # 受载节点数、受载节点编号、载荷
            num_loaded_nodes, loaded_node_ids, forces = read_parameter_1d(fr, 'float')
        elif line == '* displacement':
            # 位移约束节点数、位移约束节点编号、位移大小
            num_disp_node, disp_node_ids, displacements = read_parameter_1d(fr, 'float')
        else:
            print('无效的关键字：{}'.format(line.strip()))
            return
        line = fr.readline()
    fr.close()

    # ******************************** 生成节点自由度矩阵 ******************************* #
    num_node_dof = 1  # 每个节点的自由度数
    num_elem_dof = 2  # 每个单元的自由度度数
    num_node = num_elem + 1  # 节点数
    node_ids = [i+1 for i in range(num_node)]  # 节点编号
    # 根据约束信息生成自由度矩阵
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_node_ids, fixed_components)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_equation = np.max(node_dof)  # 自由度总数，也是方程的总数
    k_diag = np.zeros(num_equation, dtype=int)  # 用于存储矩阵对角线元素在向量中的位置
    # 单元定位向量，存储单元中的自由度是整体的第几号自由度:
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=int)
    for i_elem in range(num_elem):
        elem_conn = np.array([i_elem+1, i_elem + 2], dtype=int)  # 单元节点编号，从1开始
        elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
        global_elem_dof[i_elem, :] = elem_dof
        k_diag = form_k_diag(k_diag, elem_dof)

    # for循环之前，k_diag矩阵存储矩阵中每一行存储几个元素
    # for循环结束后，k_diag矩阵存储对角线元素在kv矩阵中的位置
    for i in range(1, num_equation):
        k_diag[i] = k_diag[i] + k_diag[i - 1]

    # **************************** 组装刚度矩阵 **************************** #
    kv = np.zeros(k_diag[-1], dtype=np.float)  # 一维变带宽存储的单元刚度矩阵，初始化
    ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 单元刚度矩阵
    for i_elem in range(num_elem):
        ke = rod_ke(ke, prop[prop_ids.index(elem_prop_type[i_elem]), 0],
                    prop[prop_ids.index(elem_prop_type[i_elem]), 1], elem_length[i_elem])
        elem_dof = global_elem_dof[i_elem, :]
        kv = form_sparse_v(kv, ke, elem_dof, k_diag)

    # *************************** 更新刚度矩阵 ****************************** #
    # 载荷
    loads = add_force(num_equation, node_ids, num_node_dof, node_dof, num_loaded_nodes,
                      loaded_node_ids, forces)
    # 根据位移载荷更新loads矩阵和kv矩阵
    kv, loads = add_displacement(num_disp_node, num_node_dof, disp_node_ids, displacements,
                                 node_ids, kv, k_diag, node_dof, loads, penalty)

    # ****************************** 方程求解 ****************************** #
    # Cholesky分解
    sparse_cho_fac(kv, k_diag)
    sparse_cho_bac(kv, loads, k_diag)

    # ****************************** 结果输出 ****************************** #
    fw_out = open(output_file, 'w', encoding='utf-8')
    fw_vtk = open(vtk_file, 'w', encoding='utf-8')

    fw_out.write('总共有{}个方程，存储带宽为{}\n'.format(num_equation, k_diag[-1]))

    fw_vtk.write('# vtk DataFile Version 3.0\n')
    fw_vtk.write('1D Bar Elements\n')
    fw_vtk.write('ASCII\n\n')

    fw_vtk.write('DATASET UNSTRUCTURED_GRID\n')

    # 节点
    fw_vtk.write('\nPOINTS {} float\n'.format(num_node))
    x_coord = 0.0
    fw_vtk.write('{:16.6e} {:16.6e} {:16.6e}\n'.format(x_coord, 0.0, 0.0))
    for i in range(len(elem_length)):
        x_coord += elem_length[i]
        fw_vtk.write('{:16.6e} {:16.6e} {:16.6e}\n'.format(x_coord, 0.0, 0.0))

    # 单元
    fw_vtk.write('\nCELLS {} {}\n'.format(num_elem, 3 * num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('2 {} {}\n'.format(i_elem, i_elem + 1))

    # 单元类型
    fw_vtk.write('\nCELL_TYPES {}\n'.format(num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('3\n')

    fw_out.write('节点      位移\n')
    fw_vtk.write('\nPOINT_DATA {}\n'.format(num_node))
    fw_vtk.write('VECTORS Displacement float\n')
    for i in range(num_node):
        fw_out.write('{:4} {:12.4e}\n'.format(i + 1, loads[node_dof[i, 0]]))
        fw_vtk.write('{:12.4e} {:12.4e} {:12.4e}\n'.format(loads[node_dof[i, 0]], 0, 0))

    fw_out.write('单元      载荷\n')
    fw_vtk.write('\nCELL_DATA {}\n'.format(num_elem))
    fw_vtk.write('VECTORS {} float\n'.format('Force'))
    for i_elem in range(num_elem):
        ke = rod_ke(ke, prop[prop_ids.index(elem_prop_type[i_elem]), 0],
                    prop[prop_ids.index(elem_prop_type[i_elem]), 1], elem_length[i_elem])
        elem_dof = global_elem_dof[i_elem, :]
        elem_dis = loads[elem_dof]  # 单元的节点位移
        action = np.dot(ke, elem_dis)
        fw_out.write('{:4}  '.format(i_elem + 1))
        for i in action:
            fw_out.write('{:12.4e}'.format(i))
            fw_vtk.write('{:12.4e}'.format(i))
        fw_out.write('\n')
        fw_vtk.write(' 0.0\n')

    fw_vtk.close()
    fw_out.close()

    # 显示结果
    if plot:
        contour_plot(vtk_file, field_name, component, shape_name, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1D bar element')
    parser.add_argument('--filename', action='store',
                        dest='filename', default='4.1_test1', type=str)
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
    filename = given_args.filename
    plot = given_args.plot
    field = given_args.field
    component = given_args.component
    shape_name = given_args.shape_name
    factor = given_args.factor
    bar1d(filename, plot, field, component, shape_name, factor)
