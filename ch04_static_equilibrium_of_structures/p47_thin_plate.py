# coding: utf-8
# author: xuxc
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa

from utility import (get_name, read_parameter, read_mesh, get_mesh_size,
                     read_element, read_elem_prop_type, initialize_node_dof,
                     update_node_dof, plate_conn, get_elem_dof, form_k_diag,
                     gauss_sample, form_plate, form_sparse_v, add_force,
                     add_displacement, sparse_cho_fac, sparse_cho_bac,
                     contour_plot)  # noqa


def plate(filename=None, plot=False, field_name='Displacement', component=-1,
          shape_name='Displacement', factor=1):
    input_file, output_file, vtk_file = get_name(filename)
    if not os.path.exists(input_file):
        print('文件不存在：{}'.format(input_file))
        return

    n_dim = None
    num_prop_types = None
    prop_ids = None
    prop = None
    num_node = None
    node_coord = None
    num_node_dof = None
    node_ids = None
    elem_type = None
    num_elem = None
    num_node_on_elem = None
    elem_ids = None
    elem_connections = None
    elem_prop_type = None
    direction = None
    num_integral_points = None
    x_coord = None
    y_coord = None
    fixed_ids = None
    fixed_components = None
    num_loaded_nodes = None
    loaded_node_ids = None
    forces = None
    num_disp_nodes = None
    disp_node_ids = None
    displacements = None

    penalty = 1e20  # 处理位移约束时使用的大数

    # ********************************** 解析输入文件 ********************************** #
    fr = open(input_file, 'r', encoding='utf-8')
    line = fr.readline()
    while line:
        line = line.strip().lower()
        if line.startswith('#') or line == '':
            pass  # 跳过注释行和空行
        elif line == '* material':
            # 材料种类数、材料分量数、材料编号、材料属性
            # 此处将厚度也放在了材料属性中，因此，prop中的参数为弹性模量、泊松比和厚度
            num_prop_types, num_prop_comps, prop_ids, prop = read_parameter(fr, 'float')
        elif line == '* mesh':  # mesh关键词出现，则不需要node、element关键词
            # 单元类型，积分点数，几个方向的坐标
            elem_type, direction, num_integral_points, x_coord, y_coord, z_coord = read_mesh(fr)
            num_x_elem = len(x_coord) - 1
            num_y_elem = len(x_coord) - 1
            num_z_elem = len(x_coord) - 1 if z_coord else None
            # 问题维数，单元数，节点数，单元节点数
            n_dim, num_elem, num_node, num_node_on_elem = get_mesh_size(
                elem_type, num_x_elem, num_y_elem, num_z_elem)
            # 每个单元的材料属性：
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* node':
            # 节点数，问题维数，节点编号，节点坐标
            num_node, n_dim, node_ids, node_coord = read_parameter(fr, 'float')
        elif line == '* element':
            # 单元类型，单元数，单元编号，单元节点数，积分点数，单元连接关系
            elem_type, num_elem, elem_ids, num_node_on_elem, num_integral_points, elem_connections = read_element(fr)
            elem_prop_type = read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr)
        elif line == '* constraint':
            # 固定节点数，节点自由度数，固定节点编号，固定的自由度分量
            # 板单元的节点自由度为w、θx(即∂w/∂x)、θy(即∂w/∂y)、θxy(即∂2w/∂x∂y）
            num_fixed_nodes, num_node_dof, fixed_ids, fixed_components = read_parameter(fr, 'int')
        elif line == '* force':
            # 受载节点数、节点自由度数、受载节点编号、载荷
            num_loaded_nodes, num_node_dof, loaded_node_ids, forces = read_parameter(fr, 'float')
        elif line == '* displacement':
            # 位移约束数、节点自由度数、位移约束节点编号、位移大小
            num_disp_nodes, num_node_dof, disp_node_ids, displacements = read_parameter(fr, 'float')
        else:
            print('无效的关键字：{}'.format(line))
            return
        line = fr.readline()
    fr.close()

    # ******************************** 生成节点自由度矩阵 ******************************* #
    if node_ids is None:
        node_ids = [i+1 for i in range(num_node)]
    if elem_ids is None:
        elem_ids = [i+1 for i in range(num_elem)]
    # 节点自由度矩阵
    node_dof = initialize_node_dof(node_ids, num_node_dof, fixed_ids, fixed_components)
    node_dof = update_node_dof(node_dof)

    # ********************************** 获取存储带宽 ********************************** #
    num_elem_dof = num_node_on_elem * num_node_dof
    num_equation = node_dof.max()  # 方程总数
    k_diag = np.zeros(num_equation, dtype=int)  # 对角元素定位向量
    global_elem_dof = np.zeros((num_elem, num_elem_dof), dtype=int)  # 全局单元自由度矩阵
    if elem_connections is None:
        elem_connections = np.zeros((num_elem, num_node_on_elem), dtype=int)
        node_coord = np.zeros((num_node, n_dim), dtype=np.float)  # 节点坐标矩阵
        for i_elem in range(num_elem):
            elem_conn, elem_coord = plate_conn(elem_type, i_elem, x_coord, y_coord, direction)
            elem_connections[i_elem, :] = elem_conn
            node_coord[elem_conn-1, :] = elem_coord
            elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
            global_elem_dof[i_elem, :] = elem_dof
            k_diag = form_k_diag(k_diag, elem_dof)
    else:
        for i_elem in range(num_elem):
            elem_conn = elem_connections[i_elem, :]
            elem_dof = get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof)
            global_elem_dof[i_elem, :] = elem_dof
            k_diag = form_k_diag(k_diag, elem_dof)
    for i in range(1, num_equation):
        k_diag[i] += k_diag[i-1]

    # ********************************** 组装刚度矩阵 ********************************** #
    gauss_points = np.zeros((num_integral_points, n_dim), dtype=np.float)  # 高斯积分点
    weights = np.zeros(num_integral_points, dtype=np.float)  # 高斯积分点的权系数
    gauss_points, weights = gauss_sample(elem_type, gauss_points, weights)
    kv = np.zeros(k_diag[-1], dtype=np.float)  # 整体刚度矩阵
    d2x = np.zeros(num_elem_dof, dtype=np.float)  # 形函数对ξ的二阶偏导
    d2y = np.zeros(num_elem_dof, dtype=np.float)  # 形函数对η的二阶偏导
    d2xy = np.zeros(num_elem_dof, dtype=np.float)  # 形函数对ξ、η的混合偏导
    dtd = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 中间变量，积分点对刚度的贡献量
    for i_elem in range(num_elem):
        e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]  # 弹性模量
        v = prop[prop_ids.index(elem_prop_type[i_elem]), 1]  # 泊松比
        h = prop[prop_ids.index(elem_prop_type[i_elem]), 2]  # 厚度
        d = e * h**3 / (12.0 * (1 - v*v))  # 薄板的抗弯刚度
        elem_dof = global_elem_dof[i_elem, :]
        ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)  # 单元刚度矩阵
        elem_conn = elem_connections[i_elem, :]
        x_size = abs(node_coord[node_ids.index(elem_conn[3]), 0] -
                     node_coord[node_ids.index(elem_conn[0]), 0])  # 单元x方向的边长
        y_size = abs(node_coord[node_ids.index(elem_conn[1]), 1] -
                     node_coord[node_ids.index(elem_conn[0]), 1])  # 单元y方向的边长
        # 计算每一个高斯积分点对刚度的贡献：
        for k in range(num_integral_points):
            d2x, d2y, d2xy = form_plate(d2x, d2y, d2xy, gauss_points, x_size, y_size, k)
            # 《有限元方法编程（第五版）》公式2.101
            # 因为x = 0.5a(ξ+1)，所以dx=0.5a*dξ
            # 对下标j的循环通过冒号索引完成
            for i in range(num_elem_dof):
                common_factor = 0.25 * x_size * y_size * weights[k]
                term1 = 16 * d2x[i] * d2x[:] / (x_size**4)
                term2 = 16 * v * d2x[i] * d2y[:] / (x_size**2 * y_size**2)
                term3 = 16 * v * d2x[:] * d2y[i] / (x_size**2 * y_size**2)
                term4 = 16 * d2y[i] * d2y[:] / (y_size**4)
                term5 = 16 * 2 * (1-v) * d2xy[i] * d2xy[:] / (x_size**2 * y_size**2)
                dtd[i, :] = common_factor * d * (term1 + term2 + term3 + term4 + term5)
            ke += dtd
        form_sparse_v(kv, ke, elem_dof, k_diag)

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

    fw_out.write('总共有{:g}个方程，存储带宽为{:g}\n'.format(num_equation, k_diag[-1]))
    fw_out.write(' 节点      位移               θx              θy                θxy\n')
    for i in range(num_node):
        fw_out.write('{:4d} '.format(node_ids[i]))
        for j in range(num_node_dof):
            fw_out.write('{:16.9E} '.format(loads[node_dof[i, j]]))
        fw_out.write('\n')

    num_integral_points = 1
    gauss_points = np.zeros((num_integral_points, n_dim), dtype=np.float)
    weights = np.zeros(num_integral_points, dtype=np.float)
    gauss_points, weights = gauss_sample(elem_type, gauss_points, weights)
    fw_out.write('单元        Mx               My               Mxy\n')
    for i_elem in range(num_elem):
        elem_dof = global_elem_dof[i_elem, :]
        e = prop[prop_ids.index(elem_prop_type[i_elem]), 0]
        v = prop[prop_ids.index(elem_prop_type[i_elem]), 1]
        h = prop[prop_ids.index(elem_prop_type[i_elem]), 2]
        d = e * h**3 / (12.0 * (1 - v**2))
        elem_conn = elem_connections[i_elem, :]
        x_size = abs(node_coord[node_ids.index(elem_conn[3]), 0] -
                     node_coord[node_ids.index(elem_conn[0]), 0])
        y_size = abs(node_coord[node_ids.index(elem_conn[1]), 1] -
                     node_coord[node_ids.index(elem_conn[0]), 1])
        for k in range(num_integral_points):
            d2x, d2y, d2xy = form_plate(d2x, d2y, d2xy, gauss_points, x_size, y_size, k)
            moment = np.zeros(3, dtype=np.float)
            # 《有限元方法编程（第五版）》公式2.97
            # 其中，w = [N]{w}
            for i in range(num_elem_dof):
                moment[0] += 4*d*(d2x[i]/(x_size**2)+v*d2y[i]/(y_size**2))*loads[elem_dof[i]]
                moment[1] += 4*d*(v*d2x[i]/(x_size**2)+d2y[i]/(y_size**2))*loads[elem_dof[i]]
                moment[2] += 4*(1-v)*(d2xy[i]/(x_size*y_size))*loads[elem_dof[i]]
            fw_out.write('{:4d} {:16.9E} {:16.9E} {:16.9E}\n'.format(
                elem_ids[i_elem], moment[0], moment[1], moment[2]))
    fw_out.close()

    fw_vtk = open(vtk_file, 'w', encoding='utf-8')
    fw_vtk.write('# vtk DataFile Version 3.0\n')
    fw_vtk.write('Bending Rectangular Plate Elements\n')
    fw_vtk.write('ASCII\n\n')
    fw_vtk.write('DATASET UNSTRUCTURED_GRID\n\n')

    fw_vtk.write('POINTS {:g} float\n'.format(num_node))
    for i in range(num_node):
        fw_vtk.write('{:16.9E} {:16.9E} {:16.9E}\n'.format(node_coord[i, 0], node_coord[i, 1], 0))

    fw_vtk.write('\nCELLS {:g} {:g}\n'.format(num_elem, (num_node_on_elem+1)*num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('{:4d} '.format(num_node_on_elem))
        for node_id in elem_connections[i_elem, :]:
            fw_vtk.write('{:4d} '.format(node_ids.index(node_id)))
        fw_vtk.write('\n')

    fw_vtk.write('\nCELL_TYPES {:g}\n'.format(num_elem))
    for i_elem in range(num_elem):
        fw_vtk.write('9\n')

    fw_vtk.write('\nPOINT_DATA {:g}\n'.format(num_node))
    fw_vtk.write('VECTORS Displacement float\n')
    for i in range(num_node):
        fw_vtk.write('{0:16.9E} {0:16.9E} {1:16.9E}\n'.format(0, loads[node_dof[i, 0]]))
    fw_vtk.close()

    # *********************************** 显示结果 ************************************ #
    if plot:
        contour_plot(vtk_file, field_name, component, shape_name, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buckling Analysis of Elastic Beam')
    parser.add_argument('-i', '--input_file', default='4.7_test1', type=str,
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
    plate(inp_file, show, field, comp, shape, scale)
