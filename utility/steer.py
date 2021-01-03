# coding: utf-8
# author: xuxc
import numpy as np

__all__ = [
    'get_elem_dof',
]


def get_elem_dof(elem_conn, node_ids, node_dof, num_elem_dof):
    """
    获取单元各节点的自由度是全局的第几号自由度
    :param elem_conn: 单元的节点编号
    :param node_ids: 节点编号
    :param node_dof: 整体的节点自由度编号
    :param num_elem_dof: 单元的自由度数
    :return: elem_dof，单元的自由度编号
    """
    num_node_in_elem = np.shape(elem_conn)[0]
    num_node_dof = np.shape(node_dof)[1]
    elem_dof = np.zeros(num_elem_dof, dtype=np.int)  # 单元定位向量，确定单元各节点的自由度是整体的第几号自由度
    for i in range(num_node_in_elem):
        k = i * num_node_dof
        elem_dof[k: k + num_node_dof] = node_dof[node_ids.index(elem_conn[i]), :]
    return elem_dof
