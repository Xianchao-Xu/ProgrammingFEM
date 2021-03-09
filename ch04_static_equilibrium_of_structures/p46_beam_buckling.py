# coding: utf-8
# author: xuxc
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from utility import (get_name, read_parameter, read_elem_prop_type, initialize_node_dof,
                     update_node_dof, get_elem_dof, form_k_diag, beam_ke, beam_ge,
                     beam_me, form_sparse_v, stability, contour_plot)  # noqa: E402


def beam_buckling(filename=None, plot=False, field_name='Displacement', component=-1,
                  shape_name='Displacement', factor=1):
    input_file, output_file, vtk_file = get_name(filename)
    if not os.path.exists(input_file):
        print('文件不存在：{}\n'.format(input_file))
        return

    num_node_on_elem = 2

    num_prop_types = None
    num_prop_comps = None
    prop_ids = None
    prop = None
    num_elem = None
    elem_length = None
    elem_prop_type = None
    num_node_dof = None
    fixed_ids = None
    fixed_component = None
    max_iter = None
    tolerance = None

    # ********************************** 解析输入文件 ********************************** #
    fr = open(input_file, 'r', encoding='utf-8')
    line = fr.readline()
    while line:
        line = line.strip().lower()
        if line.startswith('#') or line == '':
            pass
        elif line == '* material':
            # 材料种类数、材料分量数、材料变换、材料属性
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* length':
            num_elem = int(fr.readline())  # 单元数
            elem_length = np.zeros(num_elem, dtype=np.float)  # 梁单元长度
            for i in range(num_elem):
                elem_length[i] = fr.readline()
            # 每个单元的材料属性：
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* constraint':
            _, num_node_dof, fixed_ids, fixed_component = read_parameter(fr, 'int')
        elif line == '* max_iter':
            max_iter = int(fr.readline())
        elif line == '* tolerance':
            tolerance = float(fr.readline())
        else:
            print('无效的关键字：{}'.format(line))
        line = fr.readline()
    fr.close()

    # ******************************** 生成节点自由度矩阵 ******************************* #
    num_node = num_elem + 1
    node_ids = [i+1 for i in range(num_node)]
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_ids, fixed_component)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_equation = node_dof.max()  # 自由度数，亦即方程总数
    num_elem_dof = num_node_dof * num_node_on_elem
    k_diag = np.zeros(num_equation, dtype=np.int)  # 对角元素定位向量
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=np.int)
    for i_elem in range(num_elem):
        elem_conn = np.array([i_elem+1, i_elem + 2], dtype=np.int)
        elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
        global_elem_dof[i_elem, :] = elem_dof
        k_diag = form_k_diag(k_diag, elem_dof)
    for i in range(1, num_equation):
        k_diag[i] += k_diag[i-1]

    # ********************************** 组装刚度矩阵 ********************************** #
    ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 单元刚度矩阵
    kv = np.zeros(k_diag[-1], dtype=np.float)  # 整体刚度矩阵
    ge = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 单元几何矩阵
    gv = np.zeros(k_diag[-1], dtype=np.float)  # 整体几何矩阵
    for i_elem in range(num_elem):
        ke = beam_ke(ke, prop[prop_ids.index(elem_prop_type[i_elem]), 0],
                     prop[prop_ids.index(elem_prop_type[i_elem]), 1], elem_length[i_elem])
        me = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 单元质量矩阵
        if num_prop_comps > 2:
            me = beam_me(me, prop[prop_ids.index(elem_prop_type[i_elem]), 2],
                         elem_length[i_elem])
        ge = beam_ge(ge, elem_length[i_elem])
        elem_dof = global_elem_dof[i_elem, :]
        kv = form_sparse_v(kv, ke+me, elem_dof, k_diag)
        gv = form_sparse_v(gv, ge, elem_dof, k_diag)

    # ************************************ 方程求解 *********************************** #
    eigenvalue, eigenvector, iteration = stability(kv, gv, k_diag, tolerance, max_iter)

    # ************************************ 结果处理 *********************************** #
    fw_out = open(output_file, 'w', encoding='utf-8')
    fw_out.write('总共有{:g}个方程，存储带宽为{:g}\n\n'.format(num_equation, k_diag[-1]))
    fw_out.write('屈曲载荷：{:12.6E}\n\n'.format(eigenvalue))
    fw_out.write('屈曲模态：\n')
    fw_out.write('节点     位移         转角\n')
    for i in range(num_node):
        fw_out.write('{:4d}'.format(node_ids[i]))
        for j in range(num_node_dof):
            fw_out.write('{:12.4E}'.format(eigenvector[node_dof[i, j]]))
        fw_out.write('\n')
    fw_out.close()

    fw_vtk = open(vtk_file, 'w', encoding='utf-8')
    fw_vtk.write('# vtk DataFile Version 3.0\n')
    fw_vtk.write('Stability Analysis of Beam Element\n')
    fw_vtk.write('ASCII\n\n')

    fw_vtk.write('DATASET UNSTRUCTURED_GRID\n\n')
    fw_vtk.write('POINTS {} float\n'.format(num_node))
    x_coord = 0.0
    fw_vtk.write('{0:8g} {1:8g} {1:8g}\n'.format(x_coord, 0.0))
    for i in range(num_elem):
        x_coord += elem_length[i]
        fw_vtk.write('{0:8g} {1:8g} {1:8g}\n'.format(x_coord, 0.0))
    fw_vtk.write('\nCELLS {} {}\n'.format(num_elem, (num_node_on_elem+1)*num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('2 {:8g} {:8g}\n'.format(i_elem, i_elem+1))
    fw_vtk.write('\nCELL_TYPES {}\n'.format(num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('3\n')
    fw_vtk.write('\nPOINT_DATA {}\n'.format(num_node))
    fw_vtk.write('VECTORS Displacement float\n')
    for i in range(num_node):
        fw_vtk.write('{0:8g} {1:12.4E} {0:8g}\n'.format(0.0, eigenvector[node_dof[i, 0]]))
    fw_vtk.close()

    # *********************************** 显示结果 ************************************ #
    if plot:
        contour_plot(vtk_file, field_name, component, shape_name, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buckling Analysis of Elastic Beam')
    parser.add_argument('-i', '--input_file', default='4.6_test1', type=str,
                        dest='input_file', action='store')
    parser.add_argument('-p', '--plot', dest='plot', default=False,
                        action='store', type=bool)
    parser.add_argument('-f', '--field', dest='field', default='Displacement',
                        action='store', type=str)
    parser.add_argument('-c', '--component', dest='component', default=-1,
                        action='store', type=int)
    parser.add_argument('--shape', dest='shape', default='Displacement',
                        action='store', type=str)
    parser.add_argument('-s', '--scale', dest='scale', default=1,
                        action='store', type=int)
    args = parser.parse_args()
    inp_file = args.input_file
    show = args.plot
    field = args.field
    comp = args.component
    shape = args.shape
    scale = args.scale
    beam_buckling(inp_file, show, field, comp, shape, scale)
