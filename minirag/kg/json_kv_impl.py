"""
JsonDocStatus Storage Module
=======================

本模块提供了一个基于NetworkX的图存储接口，NetworkX是一个流行的Python库，用于创建、操作和研究复杂网络的结构、动态和功能。

`NetworkXStorage`类继承自LightRAG库的`BaseGraphStorage`类，提供了使用NetworkX加载、保存、操作和查询图的方法。

作者: lightrag team
创建时间: 2024-01-25
许可证: MIT

本模块允许任何人免费使用、复制、修改、合并、发布、分发、再授权和/或销售本软件及其副本，并允许被提供本软件的人在满足以下条件的情况下这样做：

上述版权声明和本许可声明应包含在本软件的所有副本或重要部分中。

本软件按“原样”提供，不附带任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性和非侵权性的担保。在任何情况下，作者或版权持有人均不对因本软件或本软件的使用或其他交易而产生的任何索赔、损害或其他责任负责，无论是在合同诉讼、侵权或其他方面。

版本: 1.0.0

依赖:
    - NetworkX
    - NumPy
    - LightRAG
    - graspologic

特性:
    - 以多种格式（如GEXF、GraphML、JSON）加载和保存图
    - 查询图的节点和边
    - 计算节点和边的度
    - 使用多种算法（如Node2Vec）对节点进行嵌入
    - 从图中移除节点和边

用法:
    from minirag.storage.networkx_storage import NetworkXStorage

"""

import asyncio
import os
from dataclasses import dataclass

from minirag.utils import (
    logger,      # 日志工具
    load_json,   # 加载JSON文件的工具函数
    write_json,  # 写入JSON文件的工具函数
)

from minirag.base import (
    BaseKVStorage,  # 基础KV存储抽象类
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    """
    基于JSON文件的KV存储实现，继承自BaseKVStorage。
    支持异步操作，适用于小规模数据的本地持久化。
    """
    def __post_init__(self):
        # 初始化时设置工作目录和存储文件名，并加载已有数据
        working_dir = self.global_config["working_dir"]  # 从全局配置获取工作目录
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")  # 拼接存储文件名
        self._data = load_json(self._file_name) or {}  # 加载已有数据，若无则为空字典
        self._lock = asyncio.Lock()  # 异步锁，保证并发安全
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")  # 记录加载信息

    async def all_keys(self) -> list[str]:
        """
        获取所有已存储的key列表。
        Returns:
            list[str]: 所有key的列表
        """
        return list(self._data.keys())

    async def index_done_callback(self):
        """
        索引操作完成后的回调，将当前数据写入JSON文件持久化。
        """
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        """
        根据id获取对应的数据。
        Args:
            id: 数据的唯一标识符
        Returns:
            dict或None: 对应的数据或不存在时返回None
        """
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        """
        批量根据id获取数据，可选只返回部分字段。
        Args:
            ids: id列表
            fields: 需要返回的字段列表（可选）
        Returns:
            list: 每个id对应的数据（或None），如指定fields则只返回这些字段
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        过滤出data中尚未存储的key。
        Args:
            data: 待检查的key列表
        Returns:
            set[str]: 不在存储中的key集合
        """
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        """
        插入或更新数据，仅插入尚未存在的key。
        Args:
            data: 待插入的{key: value}字典
        Returns:
            dict: 实际插入的新数据（已存在的不返回）
        """
        left_data = {k: v for k, v in data.items() if k not in self._data}  # 仅保留新key
        self._data.update(left_data)  # 更新存储
        return left_data

    async def drop(self):
        """
        清空所有存储数据。
        """
        self._data = {}

    async def filter(self, filter_func):
        """
        根据过滤函数筛选符合条件的key-value对。
        Args:
            filter_func: 过滤函数，接收value参数，返回bool
        Returns:
            dict: 满足条件的key-value对
        """
        result = {}
        async with self._lock:  # 保证并发安全
            for key, value in self._data.items():
                if filter_func(value):
                    result[key] = value
        return result

    async def delete(self, ids: list[str]):
        """
        删除指定id的数据。
        Args:
            ids: 待删除的id列表
        """
        async with self._lock:  # 保证并发安全
            for id in ids:
                if id in self._data:
                    del self._data[id]
            await self.index_done_callback()  # 删除后持久化
            logger.info(f"Successfully deleted {len(ids)} items from {self.namespace}")
