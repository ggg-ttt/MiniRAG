import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Any
from dotenv import load_dotenv

# 导入各类操作函数和工具函数
from .operate import (
    chunking_by_token_size,
    extract_entities,
    hybrid_query,
    minirag_query,
    naive_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    clean_text,
    get_content_summary,
    set_logger,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    DocStatus,
)

# 存储类型与实现模块的映射
STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.jsondocstatus_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorge": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "WeaviateVectorStorage": ".kg.weaviate_impl",
    "WeaviateKVStorage": ".kg.weaviate_impl",
    "WeaviateGraphStorage": ".kg.weaviate_impl",
    "run_sync": ".kg.weaviate_impl",
}

# future KG integrations
# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )

# 加载环境变量
load_dotenv(dotenv_path=".env", override=False)


def lazy_external_import(module_name: str, class_name: str):
    """
    懒加载外部模块中的类。
    :param module_name: 模块名
    :param class_name: 类名
    :return: 类的构造器
    """
    import inspect
    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib
        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    保证总能获取到一个可用的事件循环。
    :return: 当前或新建的事件循环
    """
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class MiniRAG:
    """
    MiniRAG主类，负责RAG系统的初始化、文档插入、查询、删除等核心流程。
    支持多种存储后端、分块、实体抽取、异步处理等。
    """
    working_dir: str = field(
        default_factory=lambda: f"./minirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)
    # 文本分块参数
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"
    # 实体抽取参数
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    # 节点嵌入参数
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    embedding_func: EmbeddingFunc = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    # LLM相关参数
    llm_model_func: callable = None
    llm_model_name: str = (
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)
    # 存储相关参数
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True
    # 扩展参数
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    # 文档状态存储类型
    doc_status_storage: str = field(default="JsonDocStatusStorage")
    # 自定义分块函数
    chunking_func: callable = chunking_by_token_size
    chunking_func_kwargs: dict = field(default_factory=dict)
    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))

    def __post_init__(self):
        """
        初始化方法，设置日志、工作目录、各类存储、缓存、嵌入函数等。
        """
        log_file = os.path.join(self.working_dir, "minirag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        # 打印全局配置
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"MiniRAG init with param:\n  {_print_config}\n")
        # 初始化各类存储类
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )
        self.key_string_value_json_storage_cls = partial(
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(
            self.graph_storage_cls, global_config=global_config
        )
        self.json_doc_status_storage = self.key_string_value_json_storage_cls(
            namespace="json_doc_status_storage",
            embedding_func=None,
        )
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        # 限制嵌入函数的并发数
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        ####
        # 文档、分块、图等存储实例
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        global_config = asdict(self)
        self.entity_name_vdb = self.vector_db_storage_cls(
            namespace="entities_name",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        # LLM模型函数并发限制
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )
        # 文档状态存储
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)
        self.doc_status = self.doc_status_storage_cls(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=None,
        )

    def _get_storage_class(self, storage_name: str) -> dict:
        """
        根据存储名获取对应的存储类。
        :param storage_name: 存储类型名
        :return: 存储类
        """
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def set_storage_client(self, db_client):
        """
        设置底层数据库客户端（如Oracle等）。
        :param db_client: 数据库客户端实例
        """
        for storage in [
            self.vector_db_storage_cls,
            self.graph_storage_cls,
            self.doc_status,
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.key_string_value_json_storage_cls,
            self.chunks_vdb,
            self.relationships_vdb,
            self.entities_vdb,
            self.graph_storage_cls,
            self.chunk_entity_relation_graph,
            self.llm_response_cache,
        ]:
            # 统一设置db属性
            storage.db = db_client

    def insert(self, string_or_strings):
        """
        同步插入文档（自动调度异步）。
        :param string_or_strings: 单个或多个文档字符串
        :return: 插入结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """
        异步插入文档，支持分块、实体抽取等。
        :param input: 文本或文本列表
        :param split_by_character: 按字符分割（可选）
        :param split_by_character_only: 是否仅按字符分割
        :param ids: 文档ID或ID列表
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        await self.apipeline_enqueue_documents(input, ids)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )
        # 对新处理的分块做实体抽取
        inserting_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": doc_id,
            }
            for doc_id, status_doc in (
                await self.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            ).items()
            for dp in self.chunking_func(
                status_doc.content,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }
        if inserting_chunks:
            logger.info("Performing entity extraction on newly processed chunks")
            await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                entity_name_vdb=self.entity_name_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
        await self._insert_done()

    async def apipeline_enqueue_documents(
        self, input: str | list[str], ids: list[str] | None = None
    ) -> None:
        """
        文档入队处理流程：去重、生成ID、过滤已处理文档、入队。
        :param input: 文本或文本列表
        :param ids: ID列表
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")
            contents = {id_: doc for id_, doc in zip(ids, input)}
        else:
            input = list(set(clean_text(doc) for doc in input))
            contents = {compute_mdhash_id(doc, prefix="doc-"): doc for doc in input}
        unique_contents = {
            id_: content
            for content, id_ in {
                content: id_ for id_, content in contents.items()
            }.items()
        }
        new_docs: dict[str, Any] = {
            id_: {
                "content": content,
                "content_summary": get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for id_, content in unique_contents.items()
        }
        all_new_doc_ids = set(new_docs.keys())
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }
        if not new_docs:
            logger.info("No new unique documents were found.")
            return
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        处理待处理文档：分块、实体关系抽取、状态更新。
        :param split_by_character: 按字符分割（可选）
        :param split_by_character_only: 是否仅按字符分割
        """
        processing_docs, failed_docs, pending_docs = await asyncio.gather(
            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self.doc_status.get_docs_by_status(DocStatus.FAILED),
            self.doc_status.get_docs_by_status(DocStatus.PENDING),
        )
        to_process_docs: dict[str, Any] = {
            **processing_docs,
            **failed_docs,
            **pending_docs,
        }
        if not to_process_docs:
            logger.info("No documents to process")
            return
        # 分批处理，支持并发
        docs_batches = [
            list(to_process_docs.items())[i : i + self.max_parallel_insert]
            for i in range(0, len(to_process_docs), self.max_parallel_insert)
        ]
        logger.info(f"Number of batches to process: {len(docs_batches)}")
        for batch_idx, docs_batch in enumerate(docs_batches):
            for doc_id, status_doc in docs_batch:
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_id,
                    }
                    for dp in self.chunking_func(
                        status_doc.content,
                        self.chunk_overlap_token_size,
                        self.chunk_token_size,
                        self.tiktoken_model_name,
                    )
                }
                await asyncio.gather(
                    self.chunks_vdb.upsert(chunks),
                    self.full_docs.upsert({doc_id: {"content": status_doc.content}}),
                    self.text_chunks.upsert(chunks),
                )
                await self.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSED,
                            "chunks_count": len(chunks),
                            "content": status_doc.content,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now().isoformat(),
                        }
                    }
                )
        logger.info("Document processing pipeline completed")

    async def _insert_done(self):
        """
        插入流程结束后的回调，通知各存储完成索引。
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.entity_name_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        同步查询接口。
        :param query: 查询字符串
        :param param: 查询参数
        :return: 查询结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步查询接口，支持多种模式（light/mini/naive）。
        :param query: 查询字符串
        :param param: 查询参数
        :return: 查询结果
        """
        if param.mode == "light":
            response = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "mini":
            response = await minirag_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.entity_name_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                self.embedding_func,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        """
        查询流程结束后的回调，通知缓存完成索引。
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        """
        同步删除指定实体及其关系。
        :param entity_name: 实体名
        :return: 删除结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        """
        异步删除指定实体及其关系。
        :param entity_name: 实体名
        """
        entity_name = f'"{entity_name.upper()}"'
        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)
            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        """
        删除实体流程结束后的回调，通知相关存储完成索引。
        """
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)