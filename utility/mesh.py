# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'gen_plate',
    'get_mesh_size',
    'plate_conn',
]


def gen_plate(element, num_node, num_elem, num_node_on_elem, x_coord, y_coord, direction):
    """
    生成平面模型的节点和单元信息
    :param element: 单元类型
    :param num_node: 节点数
    :param num_elem: 单元数
    :param num_node_on_elem: 单元节点数
    :param x_coord: x方向坐标列表
    :param y_coord: y方向坐标列表
    :param direction: 节点编号方向
    :return: 节点坐标node_coord, 单元连接关系elem_connections
    """
    elem_connections = np.zeros((num_elem, num_node_on_elem), dtype=np.int)
    node_coord = np.zeros((num_node, 2), dtype=np.float)
    for i_elem in range(num_elem):
        elem_conn, elem_coord = plate_conn(element, i_elem, x_coord, y_coord, direction)
        elem_connections[i_elem, :] = elem_conn
        node_coord[elem_conn-1, :] = elem_coord
    return node_coord, elem_connections


def get_mesh_size(elem_type, num_x_elem, num_y_elem, num_z_elem=None):
    """
    获取单元总数和节点总数
    :param elem_type: 单元类型
    :param num_x_elem: x方向的单元数
    :param num_y_elem: y方向的单元数
    :param num_z_elem: z方向的单元数
    :return: 问题维数n_dim，单元数num_elem，节点数num_node，单元节点数
    """
    if elem_type[:4] == 'tria':
        n_dim = 2
        num_elem = num_x_elem * num_y_elem * 2
        if elem_type == 'tria3':
            num_node = (num_x_elem + 1) * (num_y_elem + 1)
            num_node_on_elem = 3
        elif elem_type == 'tria6':
            num_node = (2 * num_x_elem + 1) * (2 * num_y_elem + 1)
            num_node_on_elem = 6
        elif elem_type == 'tria10':
            num_node = (3 * num_x_elem + 1) * (3 * num_y_elem + 1)
            num_node_on_elem = 10
        elif elem_type == 'tria15':
            num_node = (4 * num_x_elem + 1) * (4 * num_y_elem + 1)
            num_node_on_elem = 15
        else:
            print('错误的单元类型，节点数错误：{}'.format(elem_type))
            return
    elif elem_type[:4] == 'quad':
        n_dim = 2
        num_elem = num_x_elem * num_y_elem
        if elem_type == 'quad4':
            num_node = (num_x_elem+1) * (num_y_elem+1)
            num_node_on_elem = 4
        elif elem_type == 'quad8':
            num_node = (2*num_x_elem+1)*(num_y_elem+1) + (num_x_elem+1) * num_y_elem
            num_node_on_elem = 8
        else:
            print('错误的单元类型，节点数错误：{}'.format(elem_type))
            return
    else:
        print('错误的单元类型，单元形状错误：{}'.format(elem_type))
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
    if 'tria' in element:
        num_y_elem = (np.size(y_coord, 0) - 1) * 2
        if direction == 'x' or direction == 'r':
            j_elem = 2 * num_x_elem * (i_elem // (2 * num_x_elem))  # 该排单元的计数起点
            ip = (i_elem - j_elem) // 2 + 1  # x方向的第几个单元
            iq = 2 * (i_elem // (2 * num_x_elem) + 1) - 1 + (((i_elem + 1) // 2) * 2) // (i_elem + 1)
        else:
            j_elem = i_elem // num_y_elem
            ip = j_elem + 1
            iq = i_elem + 1 - num_y_elem * j_elem
        if element == 'tria3':
            num_node_on_elem = 3
            elem_conn = np.zeros(num_node_on_elem, dtype=np.int)
            elem_coord = np.zeros((num_node_on_elem, n_dim), dtype=np.float)
            if np.mod(iq, 2) != 0:
                if direction == 'x' or direction == 'r':
                    elem_conn[0] = (num_x_elem + 1) * (iq - 1) // 2 + ip
                    elem_conn[1] = elem_conn[0] + 1
                    elem_conn[2] = (num_x_elem + 1) * (iq + 1) // 2 + ip
                else:
                    elem_conn[0] = (ip - 1) * (num_y_elem + 2) // 2 + (iq + 1) // 2
                    elem_conn[1] = elem_conn[0] + (num_y_elem + 2) // 2
                    elem_conn[2] = elem_conn[0] + 1
                elem_coord[0, 0] = x_coord[ip - 1]
                elem_coord[0, 1] = y_coord[(iq + 1) // 2 - 1]
                elem_coord[1, 0] = x_coord[ip]
                elem_coord[1, 1] = y_coord[(iq + 1) // 2 - 1]
                elem_coord[2, 0] = x_coord[ip - 1]
                elem_coord[2, 1] = y_coord[(iq + 3) // 2 - 1]
            else:
                if direction == 'x' or direction == 'r':
                    elem_conn[0] = (num_x_elem + 1) * iq // 2 + ip + 1
                    elem_conn[1] = elem_conn[0] - 1
                    elem_conn[2] = (num_x_elem + 1) * (iq - 2) // 2 + ip + 1
                else:
                    elem_conn[0] = ip * (num_y_elem + 2) // 2 + (iq + 2) / 2
                    elem_conn[1] = (ip - 1) * (num_y_elem + 2) // 2 + (iq + 1) // 2 + 1
                    elem_conn[2] = elem_conn[0] - 1
                elem_coord[0, 0] = x_coord[ip]
                elem_coord[0, 1] = y_coord[iq // 2]
                elem_coord[1, 0] = x_coord[ip - 1]
                elem_coord[1, 1] = y_coord[iq // 2]
                elem_coord[2, 0] = x_coord[ip]
                elem_coord[2, 1] = y_coord[iq // 2 - 1]
        elif element == 'tria6':
            num_node_on_elem = 6
            elem_conn = np.zeros(num_node_on_elem, dtype=np.int)
            elem_coord = np.zeros((num_node_on_elem, n_dim), dtype=np.float)
            if np.mod(iq, 2) != 0:
                if direction == 'x' or direction == 'r':
                    elem_conn[0] = (iq - 1) * (2 * num_x_elem + 1) + 2 * ip - 1
                    elem_conn[1] = elem_conn[0] + 1
                    elem_conn[2] = elem_conn[0] + 2
                    elem_conn[3] = (iq - 1) * (2 * num_x_elem + 1) + 2 * num_x_elem + 2 * ip + 1
                    elem_conn[4] = (iq + 1) * (2 * num_x_elem + 1) + 2 * ip - 1
                    elem_conn[5] = elem_conn[3] - 1
                else:
                    elem_conn[0] = 2 * (num_y_elem + 1) * (ip - 1) + iq
                    elem_conn[1] = 2 * (num_y_elem + 1) * (ip - 1) + num_y_elem + 1 + iq
                    elem_conn[2] = 2 * (num_y_elem + 1) * ip + iq
                    elem_conn[3] = elem_conn[1] + 1
                    elem_conn[4] = elem_conn[0] + 2
                    elem_conn[5] = elem_conn[0] + 1
                elem_coord[0, 0] = x_coord[ip - 1]
                elem_coord[0, 1] = y_coord[iq // 2]
                elem_coord[2, 0] = x_coord[ip]
                elem_coord[2, 1] = y_coord[iq // 2]
                elem_coord[4, 0] = x_coord[ip - 1]
                elem_coord[4, 1] = y_coord[(iq + 1) // 2]
            else:
                if direction == 'x' or direction == 'r':
                    elem_conn[0] = iq * (2 * num_x_elem + 1) + 2 * ip + 1
                    elem_conn[1] = elem_conn[0] - 1
                    elem_conn[2] = elem_conn[0] - 2
                    elem_conn[3] = (iq - 2) * (2 * num_x_elem + 1) + 2 * num_x_elem + 2 * ip + 1
                    elem_conn[4] = (iq - 2) * (2 * num_x_elem + 1) + 2 * ip + 1
                    elem_conn[5] = elem_conn[3] + 1
                else:
                    elem_conn[0] = 2 * (num_y_elem + 1) * ip + iq + 1
                    elem_conn[1] = 2 * (num_y_elem + 1) * (ip - 1) + num_y_elem + iq + 2
                    elem_conn[2] = 2 * (num_y_elem + 1) * (ip - 1) + iq + 1
                    elem_conn[3] = elem_conn[1] - 1
                    elem_conn[4] = elem_conn[0] - 2
                    elem_conn[5] = elem_conn[0] - 1
                elem_coord[0, 0] = x_coord[ip]
                elem_coord[0, 1] = y_coord[(iq + 1) // 2]
                elem_coord[2, 0] = x_coord[ip - 1]
                elem_coord[2, 1] = y_coord[(iq + 1) // 2]
                elem_coord[4, 0] = x_coord[ip]
                elem_coord[4, 1] = y_coord[(iq - 1) // 2]
            elem_coord[1, :] = 0.5 * (elem_coord[0, :] + elem_coord[2, :])
            elem_coord[3, :] = 0.5 * (elem_coord[2, :] + elem_coord[4, :])
            elem_coord[5, :] = 0.5 * (elem_coord[4, :] + elem_coord[0, :])
        elif element == 'tria10':
            num_node_on_elem = 10
            elem_conn = np.zeros(num_node_on_elem, dtype=np.int)
            elem_coord = np.zeros((num_node_on_elem, n_dim), dtype=np.float)
            if np.mod(iq, 2) != 0:
                if direction == 'x' or direction == 'r':
                    elem_conn[0] = (iq - 1) // 2 * (3 * num_x_elem + 1) * 3 + 3 * ip - 2
                    elem_conn[1] = elem_conn[0] + 1
                    elem_conn[2] = elem_conn[0] + 2
                    elem_conn[3] = elem_conn[0] + 3
                    elem_conn[4] = (iq-1)//2*(3*num_x_elem+1)*3+3*num_x_elem+1+3*ip
                    elem_conn[5] = (iq-1)//2*(3*num_x_elem+1)*3+6*num_x_elem+2+3*ip-1
                    elem_conn[6] = (iq-1)//2*(3*num_x_elem+1)*3+9*num_x_elem+3+3*ip-2
                    elem_conn[7] = elem_conn[5] - 1
                    elem_conn[8] = elem_conn[4] - 2
                    elem_conn[9] = elem_conn[8] + 1
                else:
                    elem_conn[0] = (9*(num_y_elem-2)/2+12)*(ip-1)+3*(iq-1)/2+1
                    elem_conn[1] = (9*(num_y_elem-2)/2+12)*(ip-1)+3*(num_y_elem-2)/2+4+3*(iq-1)/2+1
                    elem_conn[2] = (9*(num_y_elem-2)/2+12)*(ip-1)+3*(num_y_elem-2)+8+3*(iq-1)/2+1
                    elem_conn[3] = (9*(num_y_elem-2)/2+12)*(ip-1)+9*(num_y_elem-2)/2+12+3*(iq-1)/2+1
                    elem_conn[4] = elem_conn[2] + 1
                    elem_conn[5] = elem_conn[1] + 2
                    elem_conn[6] = elem_conn[0] + 3
                    elem_conn[7] = elem_conn[0] + 2
                    elem_conn[8] = elem_conn[0] + 1
                    elem_conn[9] = elem_conn[1] + 1
                elem_coord[0, 0] = x_coord[ip-1]
                elem_coord[1, 0] = x_coord[ip-1]+(x_coord[ip]-x_coord[ip-1])/3.
                elem_coord[2, 0] = x_coord[ip-1]+2.*(x_coord[ip]-x_coord[ip-1])/3.
                elem_coord[3, 0] = x_coord[ip]
                elem_coord[3, 1] = y_coord[iq//2]
                elem_coord[4, 1] = y_coord[iq//2]+(y_coord[(iq+2)//2]-y_coord[iq//2])/3.
                elem_coord[5, 1] = y_coord[iq//2]+2.*(y_coord[(iq+2)//2]-y_coord[iq//2])/3.
                elem_coord[6, 1] = y_coord[(iq+2)//2]
            else:
                if direction == 'x' or direction == 'r':
                    elem_conn[0] = (iq-2)/2*(3*num_x_elem+1)*3+9*num_x_elem+3+3*ip+1
                    elem_conn[1] = elem_conn[0] - 1
                    elem_conn[2] = elem_conn[0] - 2
                    elem_conn[3] = elem_conn[0] - 3
                    elem_conn[4] = (iq-2)//2*(3*num_x_elem+1)*3+6*num_x_elem+2+3*ip-1
                    elem_conn[5] = (iq-2)//2*(3*num_x_elem+1)*3+3*num_x_elem+1+3*ip
                    elem_conn[6] = (iq-2)//2*(3*num_x_elem+1)*3+3*ip+1
                    elem_conn[7] = elem_conn[5] + 1
                    elem_conn[8] = elem_conn[4] + 2
                    elem_conn[9] = elem_conn[8] - 1
                else:
                    elem_conn[0] = (9*(num_y_elem-2)//2+12)*(ip-1)+9*(num_y_elem-2)//2+12+3*iq//2+1
                    elem_conn[1] = (9*(num_y_elem-2)//2+12)*(ip-1)+3*(num_y_elem-2)+8+3*iq//2+1
                    elem_conn[2] = (9*(num_y_elem-2)//2+12)*(ip-1)+3*(num_y_elem-2)//2+4+3*iq//2+1
                    elem_conn[3] = (9*(num_y_elem-2)//2+12)*(ip-1)+3*iq/2+1
                    elem_conn[4] = elem_conn[2] - 1
                    elem_conn[5] = elem_conn[1] - 2
                    elem_conn[6] = elem_conn[0] - 3
                    elem_conn[7] = elem_conn[0] - 2
                    elem_conn[8] = elem_conn[0] - 1
                    elem_conn[9] = elem_conn[1] - 1
                elem_coord[0, 0] = x_coord[ip]
                elem_coord[1, 0] = x_coord[ip]-(x_coord[ip]-x_coord[ip-1])/3.
                elem_coord[2, 0] = x_coord[ip]-2.0*(x_coord[ip]-x_coord[ip-1])/3.
                elem_coord[3, 0] = x_coord[ip-1]
                elem_coord[3, 1] = y_coord[(iq+1)//2]
                elem_coord[4, 1] = y_coord[(iq+1)//2]-(y_coord[(iq+1)//2]-y_coord[(iq-1)//2])/3.
                elem_coord[5, 1] = y_coord[(iq+1)//2]-2.*(y_coord[(iq+1)//2]-y_coord[(iq-1)//2])/3.
                elem_coord[6, 1] = y_coord[(iq-1)//2]
            elem_coord[4, 0] = elem_coord[2, 0]
            elem_coord[5, 0] = elem_coord[1, 0]
            elem_coord[6, 0] = elem_coord[0, 0]
            elem_coord[7, 0] = elem_coord[0, 0]
            elem_coord[8, 0] = elem_coord[0, 0]
            elem_coord[9, 0] = elem_coord[1, 0]
            elem_coord[0, 1] = elem_coord[3, 1]
            elem_coord[1, 1] = elem_coord[3, 1]
            elem_coord[2, 1] = elem_coord[3, 1]
            elem_coord[7, 1] = elem_coord[5, 1]
            elem_coord[8, 1] = elem_coord[4, 1]
            elem_coord[9, 1] = elem_coord[4, 1]
        elif element == 'tria15':
            num_node_on_elem = 15
            elem_conn = np.zeros(num_node_on_elem, dtype=np.int)
            elem_coord = np.zeros((num_node_on_elem, n_dim), dtype=np.float)
            if np.mod(iq, 2) != 0:
                if direction == 'x' or direction == 'r':
                    fac1 = 4 * (4 * num_x_elem + 1) * (iq - 1) // 2
                    elem_conn[0] = fac1 + 4 * ip - 3
                    elem_conn[1] = elem_conn[0] + 1
                    elem_conn[2] = elem_conn[0] + 2
                    elem_conn[3] = elem_conn[0] + 3
                    elem_conn[4] = elem_conn[0] + 4
                    elem_conn[5] = fac1 + 4 * num_x_elem + 1 + 4 * ip
                    elem_conn[6] = fac1 + 8 * num_x_elem + 1 + 4 * ip
                    elem_conn[7] = fac1 + 12 * num_x_elem + 1 + 4 * ip
                    elem_conn[8] = fac1 + 16 * num_x_elem + 1 + 4 * ip
                    elem_conn[9] = elem_conn[7] - 1
                    elem_conn[10] = elem_conn[6] - 2
                    elem_conn[11] = elem_conn[5] - 3
                    elem_conn[12] = elem_conn[11] + 1
                    elem_conn[13] = elem_conn[11] + 2
                    elem_conn[14] = elem_conn[10] + 1
                else:
                    fac1 = 4 * (2 * num_y_elem + 1) * (ip - 1) + 2 * iq - 1
                    elem_conn[0] = fac1
                    elem_conn[1] = fac1 + 2 * num_y_elem + 1
                    elem_conn[2] = fac1 + 4 * num_y_elem + 2
                    elem_conn[3] = fac1 + 6 * num_y_elem + 3
                    elem_conn[4] = fac1 + 8 * num_y_elem + 4
                    elem_conn[5] = fac1 + 6 * num_y_elem + 4
                    elem_conn[6] = fac1 + 4 * num_y_elem + 4
                    elem_conn[7] = fac1 + 2 * num_y_elem + 4
                    elem_conn[8] = fac1 + 4
                    elem_conn[9] = fac1 + 3
                    elem_conn[10] = fac1 + 2
                    elem_conn[11] = fac1 + 1
                    elem_conn[12] = fac1 + 2 * num_y_elem + 2
                    elem_conn[13] = fac1 + 4 * num_y_elem + 3
                    elem_conn[14] = fac1 + 2 * num_y_elem + 3
                elem_coord[0, 0] = x_coord[ip - 1]
                elem_coord[0, 1] = y_coord[iq // 2]
                elem_coord[4, 0] = x_coord[ip]
                elem_coord[4, 1] = y_coord[iq // 2]
                elem_coord[8, 0] = x_coord[ip - 1]
                elem_coord[8, 1] = y_coord[(iq + 2) // 2]
            else:
                if direction == 'x' or direction == 'r':
                    fac1 = 4 * (4 * num_x_elem + 1) * (iq - 2) // 2
                    elem_conn[0] = fac1 + 16 * num_x_elem + 5 + 4 * ip
                    elem_conn[1] = elem_conn[0] - 1
                    elem_conn[2] = elem_conn[0] - 2
                    elem_conn[3] = elem_conn[0] - 3
                    elem_conn[4] = elem_conn[0] - 4
                    elem_conn[5] = fac1 + 12 * num_x_elem + 1 + 4 * ip
                    elem_conn[6] = fac1 + 8 * num_x_elem + 1 + 4 * ip
                    elem_conn[7] = fac1 + 4 * num_x_elem + 1 + 4 * ip
                    elem_conn[8] = fac1 + 4 * ip + 1
                    elem_conn[9] = elem_conn[7] + 1
                    elem_conn[10] = elem_conn[6] + 2
                    elem_conn[11] = elem_conn[5] + 3
                    elem_conn[12] = elem_conn[11] - 1
                    elem_conn[13] = elem_conn[11] - 2
                    elem_conn[14] = elem_conn[10] - 1
                else:
                    fac1 = 4 * (2 * num_y_elem + 1) * (ip - 1) + 2 * iq + 8 * num_y_elem + 5
                    elem_conn[0] = fac1
                    elem_conn[1] = fac1 - 2 * num_y_elem - 1
                    elem_conn[2] = fac1 - 4 * num_y_elem - 2
                    elem_conn[3] = fac1 - 6 * num_y_elem - 3
                    elem_conn[4] = fac1 - 8 * num_y_elem - 4
                    elem_conn[5] = fac1 - 6 * num_y_elem - 4
                    elem_conn[6] = fac1 - 4 * num_y_elem - 4
                    elem_conn[7] = fac1 - 2 * num_y_elem - 4
                    elem_conn[8] = fac1 - 4
                    elem_conn[9] = fac1 - 3
                    elem_conn[10] = fac1 - 2
                    elem_conn[11] = fac1 - 1
                    elem_conn[12] = fac1 - 2 * num_y_elem - 2
                    elem_conn[13] = fac1 - 4 * num_y_elem - 3
                    elem_conn[14] = fac1 - 2 * num_y_elem - 3
                elem_coord[0, 0] = x_coord[ip]
                elem_coord[0, 1] = y_coord[(iq + 1) // 2]
                elem_coord[4, 0] = x_coord[ip - 1]
                elem_coord[4, 1] = y_coord[(iq + 1) // 2]
                elem_coord[8, 0] = x_coord[ip]
                elem_coord[8, 1] = y_coord[(iq - 1) // 2]
            elem_coord[2, :] = 0.5 * (elem_coord[0, :] + elem_coord[4, :])
            elem_coord[6, :] = 0.5 * (elem_coord[4, :] + elem_coord[8, :])
            elem_coord[10, :] = 0.5 * (elem_coord[8, :] + elem_coord[0, :])
            elem_coord[1, :] = 0.5 * (elem_coord[0, :] + elem_coord[2, :])
            elem_coord[3, :] = 0.5 * (elem_coord[2, :] + elem_coord[4, :])
            elem_coord[5, :] = 0.5 * (elem_coord[4, :] + elem_coord[6, :])
            elem_coord[7, :] = 0.5 * (elem_coord[6, :] + elem_coord[8, :])
            elem_coord[9, :] = 0.5 * (elem_coord[8, :] + elem_coord[10, :])
            elem_coord[11, :] = 0.5 * (elem_coord[10, :] + elem_coord[0, :])
            elem_coord[14, :] = 0.5 * (elem_coord[6, :] + elem_coord[10, :])
            elem_coord[13, :] = 0.5 * (elem_coord[2, :] + elem_coord[6, :])
            elem_coord[12, :] = 0.5 * (elem_coord[1, :] + elem_coord[14, :])
        else:
            print('错误的单元类型，三角形节点数错误')
            return
    elif 'quad' in element:
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
        elif element == 'quad8':
            num_node_on_elem = 8
            elem_conn = np.zeros(num_node_on_elem, dtype=np.int)
            elem_coord = np.zeros((num_node_on_elem, n_dim), dtype=np.float)
            if direction == 'x' or direction == 'r':
                elem_conn[0] = iq * (3 * num_x_elem + 2) + 2 * ip - 1
                elem_conn[1] = iq * (3 * num_x_elem + 2) + ip - num_x_elem - 1
                elem_conn[2] = (iq - 1) * (3 * num_x_elem + 2) + 2 * ip - 1
                elem_conn[3] = elem_conn[2] + 1
                elem_conn[4] = elem_conn[3] + 1
                elem_conn[5] = elem_conn[1] + 1
                elem_conn[6] = elem_conn[0] + 2
                elem_conn[7] = elem_conn[0] + 1
            else:
                elem_conn[0] = (ip - 1) * (3 * num_y_elem + 2) + 2 * iq + 1
                elem_conn[1] = elem_conn[0] - 1
                elem_conn[2] = elem_conn[0] - 2
                elem_conn[3] = (ip - 1) * (3 * num_y_elem + 2) + 2 * num_y_elem + iq + 1
                elem_conn[4] = ip * (3 * num_y_elem + 2) + 2 * iq - 1
                elem_conn[5] = elem_conn[4] + 1
                elem_conn[6] = elem_conn[4] + 2
                elem_conn[7] = elem_conn[3] + 1
            elem_coord[0: 3, 0] = x_coord[ip - 1]
            elem_coord[4: 7, 0] = x_coord[ip]
            elem_coord[3, 0] = 0.5 * (elem_coord[2, 0] + elem_coord[4, 0])
            elem_coord[7, 0] = 0.5 * (elem_coord[6, 0] + elem_coord[0, 0])
            elem_coord[0, 1] = y_coord[iq]
            elem_coord[6: 8, 1] = y_coord[iq]
            elem_coord[2: 5, 1] = y_coord[iq - 1]
            elem_coord[1, 1] = 0.5 * (elem_coord[0, 1] + elem_coord[2, 1])
            elem_coord[5, 1] = 0.5 * (elem_coord[4, 1] + elem_coord[6, 1])
        else:
            print('错误的单元类型：矩形单元节点数错误')
            return
    else:
        print('错误的单元类型：单元形状错误')
        return
    return elem_conn, elem_coord
