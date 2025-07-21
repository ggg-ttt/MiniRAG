"""
NetworkX 存储模块
=======================

本模块提供了使用 NetworkX 的图存储接口。NetworkX 是一个流行的Python库，用于创建、操作和研究复杂网络的结构、动态和功能。

`NetworkXStorage` 类继承自 `BaseGraphStorage` 类，提供了使用 NetworkX 加载、保存、操作和查询图的方法。

作者: lightrag team
创建时间: 2024-01-25
许可证: MIT

特此授权，任何人可免费获得本软件及其相关文档文件（“本软件”）的副本，可以不受限制地处理本软件，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售本软件的副本，并允许获得本软件的人在以下条件下这样做：

上述版权声明和本许可声明应包含在本软件的所有副本或主要部分中。

本软件按“原样”提供，不作任何明示或暗示的保证，包括但不限于适销性、特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任承担责任，无论是在合同、侵权行为或其他诉讼中，还是因本软件或本软件的使用或其他交易而引起的。

版本: 1.0.0

依赖:
    - NetworkX
    - NumPy
    - LightRAG
    - graspologic

特性:
    - 以多种格式（如 GEXF、GraphML、JSON）加载和保存图
    - 查询图的节点和边
    - 计算节点和边的度
    - 使用多种算法（如 Node2Vec）对节点进行嵌入
    - 从图中移除节点和边

用法:
    from minirags.kg.networkx_impl import NetworkXStorage

"""
import asyncio
import html
import os
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
import copy

from minirag.utils import (
    logger,
)

from minirag.base import (
    BaseGraphStorage,
)

from minirag.utils import merge_tuples

@dataclass
class NetworkXStorage(BaseGraphStorage):
    """
    使用 NetworkX 库实现的图存储，作为 BaseGraphStorage 的一个具体实现。
    它在内存中管理图数据，并通过 GraphML 文件进行持久化。
    """
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        """从 GraphML 文件加载 NetworkX 图对象。"""
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        """将 NetworkX 图对象写入 GraphML 文件。"""
        logger.info(
            f"正在写入图，包含 {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """
        返回图中最大的连通分量，并以稳定的方式对节点和边进行排序。
        最大连通分量（LCC） 代表了知识库中最核心、最关联密集的部分。
        因此，很多图分析算法（如社区发现、关键节点识别等）会选择只在 LCC 上运行
        此方法参考自：https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        # 提取最大连通分量
        graph = cast(nx.Graph, largest_connected_component(graph))
        # 对节点标签进行标准化处理
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """
        确保具有相同关系的无向图总是以相同的方式被读取和处理，以保证结果的稳定性。
        此方法参考自：https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        # 对节点进行排序
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        fixed_graph.add_nodes_from(sorted_nodes)

        edges = list(graph.edges(data=True))

        # 如果是无向图，则对源节点和目标节点进行排序，确保(A,B)和(B,A)的一致性
        if not graph.is_directed():
            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    source, target = target, source
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        # 对边进行排序
        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"
        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """
        初始化方法，在对象创建后自动调用。
        它会尝试从文件加载图，如果文件不存在，则创建一个新的空图。
        """
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"从 {self._graphml_xml_file} 加载图，包含 {preloaded_graph.number_of_nodes()} 个节点, {preloaded_graph.number_of_edges()} 条边"
            )
        self._graph = preloaded_graph or nx.Graph()
        # 定义支持的节点嵌入算法
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        """索引完成后的回调，将图写入文件进行持久化。"""
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)
        
    async def get_types(self) -> tuple[list[str], list[str]]:
        """获取图中所有节点的所有实体类型。"""
        types = set()
        types_with_case = set()

        for _, data in self._graph.nodes(data=True):
            # 兼容旧的 "type" 字段和新的 "entity_type" 字段
            node_type_key = "entity_type" if "entity_type" in data else "type"
            if node_type_key in data:
                types.add(data[node_type_key].lower()) 
                types_with_case.add(data[node_type_key])
        return list(types), list(types_with_case)

    async def get_node_from_types(self, type_list: list[str]) -> Union[list[dict], None]:
        """根据给定的类型列表，获取所有匹配的节点。"""
        node_list = []
        for name, attr in self._graph.nodes(data=True):
            node_type = attr.get('entity_type', '').strip('"')
            if node_type in type_list:
                node_list.append(name)
        
        node_datas = await asyncio.gather(
            *[self.get_node(name) for name in node_list]
        )
        node_datas = [
            {**n, "entity_name": k}
            for k, n in zip(node_list, node_datas)
            if n is not None
        ]
        return node_datas

    async def get_neighbors_within_k_hops(self, source_node_id: str, k: int) -> list:
        """
        获取源节点在 k 跳范围内的所有邻居路径。
        这是一个类似广度优先搜索（BFS）的实现。
        """
        count = 0
        if not await self.has_node(source_node_id):
            logger.warning(f"节点不存在: {source_node_id}")
            return []
        
        # 初始路径为1跳邻居
        source_edge = list(self._graph.edges(source_node_id))
        count += 1
        
        # 循环扩展路径直到 k 跳
        while count < k:
            count += 1
            sc_edge = copy.deepcopy(source_edge)
            source_edge = []
            for pair in sc_edge:
                # 获取路径末端节点的新邻居
                append_edge = list(self._graph.edges(pair[-1]))
                # 合并路径
                for tuples in merge_tuples([pair], append_edge):
                    source_edge.append(tuples)
        return source_edge

    async def has_node(self, node_id: str) -> bool:
        """检查图中是否存在指定ID的节点。"""
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """检查图中是否存在指定的边。"""
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """获取指定ID节点的数据。"""
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        """获取节点的度（连接的边数）。"""
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """获取一条边的度（定义为两个端点度的和）。"""
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """获取指定边的数据。"""
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str) -> Union[list, None]:
        """获取一个节点所有的边。"""
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        """插入或更新一个节点及其数据。"""
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        """插入或更新一条边及其数据。"""
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """根据指定的 node_id 从图中删除一个节点。"""
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"节点 {node_id} 已从图中删除。")
        else:
            logger.warning(f"图中未找到要删除的节点 {node_id}。")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """使用指定算法对图节点进行嵌入。"""
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"不支持的节点嵌入算法: {algorithm}")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: 此功能当前未使用
    async def _node2vec_embed(self) -> tuple[np.ndarray, list[str]]:
        """使用 graspologic 库的 node2vec 实现节点嵌入。"""
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    def remove_nodes(self, nodes: list[str]):
        """
        批量删除多个节点。
        Args:
            nodes: 待删除的节点ID列表。
        """
        for node in nodes:
            if self._graph.has_node(node):
                self._graph.remove_node(node)

    def remove_edges(self, edges: list[tuple[str, str]]):
        """
        批量删除多条边。
        Args:
            edges: 待删除的边列表，每条边是一个 (source, target) 元组。
        """
        for source, target in edges:
            if self._graph.has_edge(source, target):
                self._graph.remove_edge(source, target)
