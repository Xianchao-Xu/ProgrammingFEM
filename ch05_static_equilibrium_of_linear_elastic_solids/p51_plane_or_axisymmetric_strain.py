# coding: utf-8
# author: xuxc
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from utility import (get_name, read_parameter, read_mesh, get_mesh_size, gen_plate,
                     read_elem_prop_type, read_element, initialize_node_dof,
                     update_node_dof, get_elem_dof, form_k_diag, gauss_sample,
                     dee_mat, shape_fun, shape_der, bee_mat, form_sparse_v,
                     add_force, add_displacement, sparse_cho_fac, sparse_cho_bac,
                     contour_plot)  # noqa: E402


def plane(filename=None, plot=False, field_name='Displacement', component=-1,
          shape_name='Displacement', factor=1):
    input_file, output_file, vtk_file = get_name(filename)
    if not os.path.exists(input_file):
        print('{}不存在！'.format(input_file))
        return

    penalty = 1.0e20

    analysis = None
    num_prop_types = None
    prop = None
    prop_ids = None
    num_node = None
    node_ids = None
    node_coord = None
    elem_type = None
    num_elem = None
    elem_ids = None
    elem_connections = None
    num_node_on_elem = None
    elem_prop_type = None
    nip = None
    n_dim = None
    num_node_dof = None
    fixed_ids = None
    fixed_components = None
    num_loaded_nodes = None
    loaded_node_ids = None
    forces = None
    num_disp_nodes = None
    disp_node_ids = None
    displacements = None

    # ********************************** 解析输入文件 ********************************** #
    fr = open(input_file, 'r', encoding='utf-8')
    line = fr.readline()
    while line:
        line = line.strip().lower()
        if line.startswith('#') or line == '':
            pass  # 跳过注释行和空行
        elif line == '* analysis_type':
            analysis = fr.readline().strip().lower()
        elif line == '* material':
            # 材料种类数、材料分量数、材料编号、材料属性
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* mesh':
            # 单元类型，积分点数，几个方向的坐标
            elem_type, direction, nip, x_coord, y_coord, z_coord = read_mesh(fr)
            num_x_elem = len(x_coord) - 1
            num_y_elem = len(y_coord) - 1
            num_z_elem = len(z_coord) if z_coord else None
            # 问题维数，单元数，节点数，单元节点数
            n_dim, num_elem, num_node, num_node_on_elem = get_mesh_size(
                elem_type, num_x_elem, num_y_elem, num_z_elem)
            node_ids = [i+1 for i in range(num_node)]
            elem_ids = [i+1 for i in range(num_elem)]
            node_coord, elem_connections = gen_plate(
                elem_type, num_node, num_elem, num_node_on_elem, x_coord, y_coord, direction)
            # 每个单元的材料属性
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* node':
            num_node, n_dim, node_ids, node_coord = read_parameter(fr, 'float')
        elif line == '* element':
            # 单元类型，单元数，单元编号，单元节点数，积分点数，单元连接关系
            elem_type, num_elem, elem_ids, num_node_on_elem, nip, elem_connections = read_element(fr)
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* constraint':
            # 固定节点数，节点自由度数，固定节点编号，固定的自由度分量
            num_fixed_nodes, num_node_dof, fixed_ids, fixed_components = read_parameter(fr, 'int')
        elif line == '* force':
            # 受载节点数，节点自由度数，受载节点编号，载荷
            num_loaded_nodes, num_node_dof, loaded_node_ids, forces = read_parameter(fr, 'float')
        elif line == '* displacement':
            # 位移约束节点数，节点自由度数，位移约束节点编号，位移
            num_disp_nodes, num_node_dof, disp_node_ids, displacements = read_parameter(fr, 'float')
        else:
            print('无效的关键字：{}'.format(line))
            return
        line = fr.readline()
    fr.close()

    num_elem_dof = num_node_on_elem * num_node_dof
    num_stress = 3
    if analysis == 'axisymmetric':
        num_stress = 4

    # ******************************** 生成节点自由度矩阵 ******************************* #
    # 节点自由度矩阵
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_ids, fixed_components)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_equation = np.max(node_dof)  # 方程数
    k_diag = np.zeros(num_equation, dtype=int)  # 对角元素辅助向量
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=int)  # 全局单元自由度矩阵
    for i_elem in range(num_elem):
        elem_conn = elem_connections[i_elem, :]
        elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
        global_elem_dof[i_elem, :] = elem_dof
        k_diag = form_k_diag(k_diag, elem_dof)
    for i in range(1, num_equation):
        k_diag[i] += k_diag[i-1]

    # ********************************** 组装刚度矩阵 ********************************** #
    gauss_points = np.zeros((nip, n_dim), dtype=np.float64)  # 局部坐标系下的高斯积分点坐标
    weights = np.zeros(nip, dtype=np.float64)  # 权系数
    gauss_points, weights = gauss_sample(elem_type, gauss_points, weights)
    kv = np.zeros(k_diag[-1], dtype=np.float64)  # 整体刚度矩阵
    global_gauss_points = np.ones(n_dim, dtype=np.float64)  # 全局坐标系下的高斯积分点坐标
    dee = np.zeros((num_stress, num_stress), dtype=np.float64)  # 弹性矩阵D
    fun = np.zeros(num_node_on_elem, dtype=np.float64)  # 形函数N
    der = np.zeros((n_dim, num_node_on_elem), dtype=np.float64)  # 局部坐标系下的形函数偏导
    bee = np.zeros((num_stress, num_elem_dof), dtype=np.float64)  # 应变矩阵B
    for i_elem in range(num_elem):
        e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]
        v = prop[prop_ids.index(elem_prop_type[i_elem]), 1]
        dee = dee_mat(dee, e, v, analysis)
        elem_conn = elem_connections[i_elem, :]
        elem_node_index = [node_ids.index(i) for i in elem_conn]
        coord = node_coord[elem_node_index, :]
        elem_dof = global_elem_dof[i_elem, :]
        ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float64)  # 单元刚度矩阵
        for i in range(nip):
            # B矩阵包含形函数对总体坐标的偏导，而形函数为在局部坐标系下导出
            # 雅可比矩阵可实现两种坐标系下导数的变换
            fun = shape_fun(fun, gauss_points, i)  # 形函数
            der = shape_der(der, gauss_points, i)  # 形函数对局部坐标的偏导
            jac = np.dot(der, coord)  # 雅可比矩阵，维度为n_dim
            inv_jac = np.linalg.inv(jac)
            derivative = np.dot(inv_jac, der)  # 形函数对总体坐标的偏导
            bee = bee_mat(bee, derivative)  # 应变矩阵B
            if analysis == 'axisymmetric':
                global_gauss_points = np.dot(fun, coord)
                # global_gauss_points[0]为径向坐标r
                bee[3, 0:num_elem_dof-1:2] = fun[:] / global_gauss_points[0]
            det = np.linalg.det(jac)  # 雅可比矩阵的行列式，积分时需要用到
            ke += np.dot(np.dot(np.transpose(bee), dee), bee)*det*weights[i]*global_gauss_points[0]
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
    # 教程上的二阶单元和vtk的二阶单元相比，节点排序不同，如果以后有时间，再调整程序，添加二阶单元
    vtk_cells = {'tria3': 5, 'quad4': 9}
    fw_vtk = None
    fw_out = open(output_file, 'w', encoding='utf-8')
    if elem_type in vtk_cells:
        fw_vtk = open(vtk_file, 'w', encoding='utf-8')
        fw_vtk.write('# vtk DataFile Version 3.0\n')
        fw_vtk.write('Plane or axisymmetric strain analysis of an elastic solid\n')
        fw_vtk.write('ASCII\n\n')
        fw_vtk.write('DATASET UNSTRUCTURED_GRID\n\n')

        fw_vtk.write('POINTS {:g} float\n'.format(num_node))
        for i in range(num_node):
            fw_vtk.write('{:16.9E} {:16.9E} {:16.9E}\n'.format(
                node_coord[i, 0], node_coord[i, 1], 0))

        fw_vtk.write('\nCELLS {:g} {:g}\n'.format(num_elem, (num_node_on_elem+1)*num_elem))
        for i_elem in range(num_elem):
            fw_vtk.write('{:4d} '.format(num_node_on_elem))
            for node_id in elem_connections[i_elem, :]:
                fw_vtk.write('{:4d} '.format(node_ids.index(node_id)))
            fw_vtk.write('\n')

        fw_vtk.write('\nCELL_TYPES {:g}\n'.format(num_elem))
        for i_elem in range(num_elem):
            fw_vtk.write('{:4d}\n'.format(vtk_cells[elem_type]))

    fw_out.write('总共有{:g}个方程，存储带宽为{:g}\n'.format(num_equation, k_diag[-1]))

    # 提取位移
    if analysis == 'axisymmetric':
        fw_out.write('\n节点        r方向位移         z方向位移\n')
    else:
        fw_out.write('\n节点        x方向位移         y方向位移\n')
    if fw_vtk:
        fw_vtk.write('\nPOINT_DATA {:g}\n'.format(num_node))
        fw_vtk.write('VECTORS Displacement float\n')
    for i in range(num_node):
        fw_out.write('{:4d} '.format(node_ids[i]))
        for j in range(num_node_dof):
            fw_out.write('{:16.6e} '.format(loads[node_dof[i, j]]))
            if fw_vtk:
                fw_vtk.write('{:16.6e} '.format(loads[node_dof[i, j]]))
        if fw_vtk:
            fw_vtk.write('{:16.6e}\n'.format(0))
        fw_out.write('\n')

    # 提取应力
    nip = 1
    gauss_points = np.zeros((nip, n_dim), dtype=np.float64)  # 高斯积分点局部坐标
    weights = np.zeros(nip, dtype=np.float64)  # 高斯积分点权系数
    gauss_points, weights = gauss_sample(elem_type, gauss_points, weights)
    fw_out.write('\n积分点（1高斯点）处的应力为：\n')
    if analysis == 'axisymmetric':
        fw_out.write('单元       r坐标         z坐标         σr         σz          τrz        σt\n')
    else:
        fw_out.write('单元       x坐标         y坐标         σx         σy          τxy\n')
    for i_elem in range(num_elem):
        e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]
        v = prop[prop_ids.index(elem_prop_type[i_elem]), 1]
        dee = dee_mat(dee, e, v, analysis)
        elem_conn = elem_connections[i_elem, :]
        elem_node_index = [node_ids.index(i) for i in elem_conn]
        coord = node_coord[elem_node_index, :]
        elem_dof = global_elem_dof[i_elem, :]
        elem_dis = loads[elem_dof]  # 单元响应（单元各节点的位移）
        for i in range(nip):
            fun = shape_fun(fun, gauss_points, i)
            der = shape_der(der, gauss_points, i)
            jac = np.dot(der, coord)
            inv_jac = np.linalg.inv(jac)
            derivative = np.dot(inv_jac, der)
            bee = bee_mat(bee, derivative)
            global_gauss_points = np.dot(fun, coord)
            if analysis == 'axisymmetric':
                bee[3, 0:num_elem_dof-1:2] = fun[:] / global_gauss_points[0]
            epsilon = np.dot(bee, elem_dis)  # 应变
            sigma = np.dot(dee, epsilon)  # 应力
            fw_out.write('{:4d} '.format(elem_ids[i_elem]))
            for dim in range(n_dim):
                fw_out.write('{:12.4e} '.format(global_gauss_points[dim]))
            for stress in range(num_stress):
                fw_out.write('{:12.4e} '.format(sigma[stress]))
            fw_out.write('\n')

    fw_out.close()
    if fw_vtk:
        fw_vtk.close()

    # *********************************** 显示结果 ************************************ #
    if plot and os.path.exists(vtk_file):
        contour_plot(vtk_file, field_name, component, shape_name, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plane or axisymmetric strain analysis')
    parser.add_argument('-i', '--input', dest='input', default='p51_tria3_test1')
    parser.add_argument('-p', '--plot', dest='plot', default=False, type=bool)
    parser.add_argument('-f', '--field', dest='field', default='Displacement')
    parser.add_argument('-c', '--component', dest='component', default=-1, type=int)
    parser.add_argument('--shape', dest='shape', default='Displacement')
    parser.add_argument('-s', '--scale', dest='scale', default=1, type=int)
    args = parser.parse_args()
    inp_file = args.input
    show = args.plot
    field = args.field
    comp = args.component
    shape = args.shape
    scale = args.scale

    plane(inp_file, show, field, comp, shape, scale)
