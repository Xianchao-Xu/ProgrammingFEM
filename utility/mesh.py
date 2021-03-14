# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'get_mesh_size',
    'plate_conn',
]


def get_mesh_size(elem_type, num_x_elem, num_y_elem, num_z_elem=None):
    """
    获取单元总数和节点总数
    :param elem_type: 单元类型
    :param num_x_elem: x方向的单元数
    :param num_y_elem: y方向的单元数
    :param num_z_elem: z方向的单元数
    :return: 问题维数n_dim，单元数num_elem，节点数num_node，单元节点数
    """
    if elem_type == 'quad4':
        num_elem = num_x_elem * num_y_elem
        num_node = (num_x_elem+1) * (num_y_elem+1)
        num_node_on_elem = 4
        n_dim = 2
    else:
        print('错误的单元类型')
        return
    return n_dim, num_elem, num_node, num_node_on_elem


def plate_conn(element, i_elem, x_coord, y_coord, direction):
    """
    获取三角形或者四边形板单元的节点坐标和单元连接关系
    :param element: 单元类型
    :param i_elem: 单元索引号（不是单元号）
    :param x_coord: x方向坐标列表
    :param y_coord: y方向坐标列表
    :param direction: 节点编号方向
    :return: 单元连接关系elem_conn，单元的节点坐标elem_coord
    """
    n_dim = 2
    num_x_elem = len(x_coord) - 1
    if 'quad' in element:
        num_y_elem = len(y_coord) - 1
        if direction == 'x' or direction == 'r':
            iq = i_elem // num_x_elem + 1  # 当前单元是y方向的第几个单元
            ip = i_elem + 1 - (iq - 1) * num_x_elem  # 当前单元是x方向的第几个单元
        else:
            ip = i_elem // num_y_elem + 1
            iq = i_elem + 1 - (ip - 1) * num_y_elem
        if element == 'quad4':
            num_node_on_elem = 4
            elem_conn = np.zeros(num_node_on_elem, dtype=np.int)
            elem_coord = np.zeros((num_node_on_elem, n_dim), dtype=np.float)
            if direction == 'x' or direction == 'r':
                elem_conn[0] = iq * (num_x_elem + 1) + ip
                elem_conn[1] = (iq - 1) * (num_x_elem + 1) + ip
                elem_conn[2] = elem_conn[1] + 1
                elem_conn[3] = elem_conn[0] + 1
            else:
                elem_conn[0] = (ip - 1) * (num_y_elem + 1) + iq + 1
                elem_conn[1] = elem_conn[0] - 1
                elem_conn[2] = ip * (num_y_elem + 1) + iq
                elem_conn[3] = elem_conn[2] + 1
            elem_coord[0, 0] = x_coord[ip - 1]
            elem_coord[1, 0] = x_coord[ip - 1]
            elem_coord[2, 0] = x_coord[ip]
            elem_coord[3, 0] = x_coord[ip]
            elem_coord[0, 1] = y_coord[iq]
            elem_coord[1, 1] = y_coord[iq - 1]
            elem_coord[2, 1] = y_coord[iq - 1]
            elem_coord[3, 1] = y_coord[iq]
        else:
            print('错误的单元类型：矩形单元节点数错误')
            return
    else:
        print('错误的单元类型：单元形状错误')
        return
    return elem_conn, elem_coord
