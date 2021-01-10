# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'get_name',
    'read_elem_prop_type',
    'read_parameter_1d',
    'read_parameter',
]


def get_name(filename=None):
    """
    生成输入文件名和输出文件名
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
