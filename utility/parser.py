# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'get_name',
    'read_beam_direction',
    'read_element',
    'read_elem_prop_type',
    'read_load_increments',
    'read_mesh',
    'read_parameter_1d',
    'read_parameter',
]


def get_name(filename=None):
    """
    生成输入文件名和输出文件名
    :param filename: 参数文件的文件名，可以不带后缀，需放置于input_files文件夹下
    :return: inp文件名、out文件名和vtk文件名
    """
    import sys
    if filename is None:
        if len(sys.argv) < 2:
            filename = input('请输入inp文件名：\n')
        else:
            filename = sys.argv[1]
    if filename.split('.')[-1] == 'inp':
        filename = filename[:-4]
    input_file = 'input_files/{}.inp'.format(filename)
    output_file = 'output_files/{}.out'.format(filename)
    vtk_file = 'vtk_files/{}.vtk'.format(filename)
    return input_file, output_file, vtk_file


def read_beam_direction(num_dim, num_elem, fr):
    """
    读取梁单元的转动角度
    :param num_dim: 问题维数，三维问题才需要梁单元方向
    :param num_elem: 单元数
    :param fr: 输入文件句柄
    :return: 每个梁单元的转动角度
    """
    direction = np.zeros(num_elem, dtype=np.float)
    if num_dim == 3:
        for i in range(num_elem):
            direction[i] = fr.readline()
    return direction


def read_element(fr):
    """
    读取单元类型和单元连接关系
    :param fr: 输入文件句柄
    :return: 单元类型elem_type、单元数num_elem、单元编号elem_ids、
             单元节点数num_node_on_elem、积分点数num_integral_points、
             单元连接关系elem_connections
    """
    line = fr.readline().strip().split()
    elem_type = line[0]
    if elem_type == 'tria3':
        num_node_on_elem = 3
    elif elem_type == 'tria15':
        num_node_on_elem = 15
    elif elem_type == 'quad4':
        num_node_on_elem = 4
    elif elem_type == 'quad8':
        num_node_on_elem = 8
    else:
        print('错误的单元类型：{}'.format(elem_type))
        return
    num_elem = int(line[1])
    num_integral_points = int(line[2])
    elem_ids = list()
    elem_connections = np.zeros((num_elem, num_node_on_elem), dtype=np.int)
    for i in range(num_elem):
        line = fr.readline().strip().split()
        elem_ids.append(int(line[0]))
        for j in range(num_node_on_elem):
            elem_connections[i, j] = line[j+1]
    return (elem_type, num_elem, elem_ids, num_node_on_elem,
            num_integral_points, elem_connections)


def read_elem_prop_type(num_elem, num_prop_types, prop_ids, fr):
    """
    读取每个单元的材料号
    :param num_elem: 单元数
    :param num_prop_types: 材料总类数
    :param prop_ids: 材料号
    :param fr: 输入文件的句柄
    :return: elem_prop_types，每个单元的材料属性号
    """
    elem_prop_type = np.ones(num_elem, dtype=np.int)
    if num_prop_types == 1:
        elem_prop_type *= prop_ids[0]
    else:
        for i in range(num_elem):
            elem_prop_type[i] = int(fr.readline())
    return elem_prop_type


def read_load_increments(fr):
    """
    读取并返回每一步的载荷增量
    :param fr: 输入文件句柄
    :return: increments，每一步的载荷增量
    """
    increments = np.array([i for i in fr.readline().split()], dtype=np.float)
    return increments


def read_mesh(fr):
    """
    读取单元相关数据
    :param fr: 输入文件句柄
    :return: 单元类型elem_type，积分点数，几个方向的坐标
    """
    elem_type = fr.readline().strip().lower()
    direction = fr.readline().strip().lower()
    num_integral_points = int(fr.readline())
    x_coord = np.array(fr.readline().split(), dtype=np.float)
    y_coord = np.array(fr.readline().split(), dtype=np.float)
    z_coord = None
    if elem_type[:4] in ['hexa']:
        z_coord = np.array(fr.readline().split(), dtype=np.float)
    return elem_type, direction, num_integral_points, x_coord, y_coord, z_coord


def read_parameter_1d(fr, para_type='float'):
    """
    :param fr: 输入文件句柄
    :param para_type: 数组的数据类型，float或者int
    :return: 行数、编号和值
    """
    row = int(fr.readline())
    ids = [0] * row
    if para_type == 'float':
        values = np.zeros((row, 1), dtype=np.float)
    elif para_type == 'int':
        values = np.zeros((row, 1), dtype=np.int)
    else:
        print('不能识别的参数类型：{}'.format(para_type))
        return
    for i in range(row):
        line = fr.readline().split()
        ids[i] = int(line[0])
        values[i, 0] = line[1]
    return row, ids, values


def read_parameter(fr, para_type='float'):
    """
    :param fr: 输入文件句柄
    :param para_type: 数组的数据类型，float或者int
    :return: 行数、列数、编号和值
    """
    row, column = [int(i) for i in fr.readline().split()]
    ids = [0] * row
    if para_type == 'float':
        values = np.zeros((row, column), dtype=np.float)
    elif para_type == 'int':
        values = np.zeros((row, column), dtype=np.int)
    else:
        print('不能识别的参数类型：{}'.format(para_type))
        return
    for i in range(row):
        line = fr.readline().split()
        ids[i] = int(line[0])
        for j in range(column):
            try:
                values[i, j] = line[j+1]
            except ValueError:
                values[i, j] = np.nan
    return row, column, ids, values
