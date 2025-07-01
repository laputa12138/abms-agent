# RAG 多智能体报告生成系统

## 项目概述

本项目是一个基于本地文档集（支持PDF, DOCX, TXT格式）的多智能体报告生成系统。它利用本地部署的嵌入（Embedding）模型和大型语言模型（LLM），通过一系列智能体（Agent）的协作，根据用户指定的主题，自动生成结构化报告。系统采用**父子分块 (Parent-Child Chunking)**策略处理长文档，并通过**混合检索 (Hybrid Search)**（向量检索 + BM25关键词检索）技术提高召回内容的相关性和上下文丰富度。所有模型调用均通过Xinference API进行，确保了本地化处理和数据私密性。

该系统现在还支持**FAISS索引的本地持久化与加载**，允许用户在多次运行时重用已处理和向量化的文档数据，避免重复计算。同时，增强了日志记录功能，支持输出到控制台和文件，并可通过命令行参数控制日志级别。

系统的核心功能包括：
1.  **多格式文档处理与父子分块**：
    *   解析用户指定文件夹内的 `.pdf`, `.docx`, `.txt` 文档。
    *   对提取的文本内容进行父子分块：父块较大，提供丰富上下文；子块较小，用于精确向量检索。
2.  **向量化、存储与持久化**：
    *   使用本地嵌入模型将 **子块** 转换为向量。
    *   存储在本地FAISS向量数据库中，同时保存子块与父块的关联信息。
    *   支持将FAISS索引和相关元数据保存到磁盘，并在后续运行时加载，避免重复处理相同文档集。
3.  **动态任务驱动的多智能体协作 (通过WorkflowState和Orchestrator)**：
    *   **WorkflowState**: 作为中央“工作记忆”，管理整个报告生成流程的动态状态，包括任务队列、大纲、章节内容、评估结果等。
    *   **Orchestrator**: 驱动工作流，从`WorkflowState`获取任务，并分发给相应的Agent执行。Agent执行后更新`WorkflowState`并可能添加新任务。
    *   **各Agent职责** (已适配WorkflowState交互模式):
        *   `TopicAnalyzerAgent`: 解析主题，生成中英文关键词和泛化主题。
        *   `OutlineGeneratorAgent`: 生成Markdown报告大纲，并为每个章节创建处理任务。
        *   `ContentRetrieverAgent` (依赖`RetrievalService`): 为章节执行混合检索（向量+BM25），获取父块上下文。
        *   `ChapterWriterAgent`: 基于检索到的父块撰写章节初稿。
        *   `EvaluatorAgent`: 评估章节质量，根据评分决定是否需要精炼。
        *   `RefinerAgent`: 根据评估反馈修改和完善章节内容。
        *   `ReportCompilerAgent`: 在所有章节完成后，整合内容生成最终报告。
4.  **本地化模型部署**：所有核心AI能力均依赖通过Xinference部署的本地模型API。

## 功能特性

*   **本地化处理**：所有文档处理和模型调用均在本地完成。
*   **多文档类型支持**：自动扫描指定文件夹，处理 `.pdf`, `.docx`, `.txt` 文件。
*   **父子分块**：优化长文档处理，通过子块精确检索，父块补充上下文。
*   **混合检索**：结合向量检索和关键词检索，提高检索质量。
*   **FAISS索引持久化**: 允许保存和加载处理好的文档向量及元数据，提高重复运行效率。
*   **动态任务驱动流程**: 基于`WorkflowState`和`Orchestrator`，实现更灵活的迭代式报告生成。
*   **增强的日志系统**: 支持控制台和文件日志，级别可配置，便于追踪和调试。
*   **中英文关键词**：主题分析阶段生成中英文关键词。
*   **高度可配置**：模型、分块、检索、日志、索引等均可通过配置文件或命令行调整。
*   **中文报告生成**：专注于生成中文报告。

## 文件结构说明

```
.
├── agents/                     # 智能体实现 (已适配WorkflowState)
│   └── ...
├── config/
│   └── settings.py             # 项目配置 (API, 模型名, 默认参数等)
├── core/                       # 核心逻辑模块
│   ├── document_processor.py   # 多文档处理, 父子分块
│   ├── embedding_service.py    # Embedding模型接口
│   ├── llm_service.py          # LLM接口
│   ├── orchestrator.py         # 新增：工作流编排器
│   ├── reranker_service.py     # Reranker模型接口
│   ├── retrieval_service.py    # 新增：核心检索服务 (混合检索)
│   ├── vector_store.py         # FAISS向量存储 (支持父子块, 持久化)
│   └── workflow_state.py       # 新增：工作流状态管理器
├── data/                       # (Git忽略) 存放用户上传的源文档
│   └── .gitkeep
├── logs/                       # (Git忽略) 存放日志文件
│   └── .gitkeep
├── output/                     # (Git忽略) 存放生成的报告
│   └── .gitkeep
├── vector_stores/              # (Git忽略, 可配置) 存放持久化的FAISS索引和元数据
│   └── .gitkeep
├── pipelines/
│   └── report_generation_pipeline.py # 已重构为使用Orchestrator和WorkflowState, 支持索引加载/保存
├── .gitignore                  # 已更新，包含logs/和vector_stores/
├── main.py                     # 主入口 (命令行接口, 已更新日志和索引参数)
├── README.md                   # 项目说明（本文档）
└── requirements.txt            # Python依赖 (已更新)
```

## 环境配置

1.  **Python环境**: Python 3.8+。建议使用虚拟环境。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```
2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    (NLTK `punkt` 模型会在首次使用时尝试自动下载。若失败，请手动执行 `import nltk; nltk.download('punkt')`)
3.  **Xinference服务**: 确保Xinference运行正常并已部署所需模型。

## 使用方法

通过 `main.py` 脚本和命令行参数运行。

**基本命令格式**:
```bash
python main.py --topic "您的报告主题" --data_path "存放源文档的文件夹路径" [其他可选参数]
```

**主要参数说明**:

*   `--topic "主题字符串"` (必需): 报告主题。
*   `--data_path "文件夹路径"` (可选, 默认: `./data/`): 源文档文件夹路径。
*   `--output_path "输出文件.md"` (可选, 默认: `output/report_时间戳.md`): 报告保存路径。
*   `--report_title "自定义标题"` (可选): 报告的自定义标题。

**Xinference与模型配置**:
*   `--xinference_url`, `--llm_model`, `--embedding_model`, `--reranker_model`.

**文档处理 - 分块参数**:
*   `--parent_chunk_size`, `--parent_chunk_overlap`, `--child_chunk_size`, `--child_chunk_overlap`.

**检索参数**:
*   `--vector_top_k`, `--keyword_top_k`, `--hybrid_search_alpha`, `--final_top_n_retrieval`.

**流程执行与索引参数**:
*   `--max_refinement_iterations <整数>`: 章节内容最大精炼次数。
*   `--max_workflow_iterations <整数>`: 工作流主循环最大迭代次数 (防死循环)。
*   `--vector_store_path "路径"` (可选, 默认: `./vector_stores/`): FAISS索引和元数据文件的保存/加载目录。
*   `--index_name "名称"` (可选): FAISS索引文件的特定名称（不含扩展名）。若提供且文件存在（且未指定`--force_reindex`），则会尝试加载。若不提供，则会基于`--data_path`的目录名自动生成。
*   `--force_reindex` (可选标志): 若设置，则强制重新处理文档并创建新索引，即使存在同名旧索引也会覆盖。

**日志参数**:
*   `--log_level <级别>` (可选, 默认: `INFO`): 控制台日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)。
*   `--debug` (可选标志): 若设置，则全局日志级别（控制台和文件）设为DEBUG。
*   `--log_path "路径"` (可选, 默认: `./logs/`): 日志文件保存目录。

**示例**:
```bash
python main.py \
    --topic "碳中和路径下的新能源技术发展趋势" \
    --data_path "./新能源研究报告/" \
    --output_path "output/新能源技术报告_v3.md" \
    --index_name "new_energy_research_2024" \
    --vector_store_path "./my_vector_indexes/" \
    --log_level DEBUG \
    --force_reindex
```
此命令将：
1.  以“碳中和路径下的新能源技术发展趋势”为主题。
2.  从 `./新能源研究报告/` 文件夹读取文档。
3.  **强制重新索引**这些文档，并将新的FAISS索引及元数据保存到 `./my_vector_indexes/` 目录下，文件名为 `new_energy_research_2024.faiss` 和 `new_energy_research_2024.meta.json`。如果已有同名文件，将被覆盖。
4.  生成的报告保存为 `output/新能源技术报告_v3.md`。
5.  控制台和文件日志都将以DEBUG级别记录。

**加载现有索引示例**:
```bash
python main.py \
    --topic "基于先前研究的深度分析" \
    --data_path "./新能源研究报告/" \
    --output_path "output/深度分析报告.md" \
    --index_name "new_energy_research_2024" \
    --vector_store_path "./my_vector_indexes/"
```
此命令将：
1.  尝试从 `./my_vector_indexes/new_energy_research_2024.faiss` 和 `.meta.json` 加载已存在的索引。
2.  如果加载成功，则**跳过** `--data_path` 中文档的重新处理和向量化步骤。
3.  如果未找到指定索引，或加载失败，则会按常规流程处理 `--data_path` 中的文档，并尝试以 `new_energy_research_2024` 为名保存新索引。

## 高级配置与定制
(与之前版本类似，可修改 `config/settings.py`，调整Agent Prompts，修改核心模块逻辑等。)

## 注意事项
(与之前版本类似，增加了对BM25中文分词的提示，并强调索引持久化功能。)
*   ...
*   **BM25分词**: 当前BM25的关键词检索使用简单的空格分词。对于中文等需要专业分词的语言，效果可能有限。未来可集成如 `jieba` 等中文分词库来提升关键词检索的准确性。
*   **索引文件**: 持久化的FAISS索引文件 (`.faiss`) 和元数据文件 (`.meta.json`) 需要一起管理。

## 未来可能的改进方向
(与之前版本类似，可补充关于Agent动态交互和更复杂工作流的展望。)
*   ...
*   增强Agent的自主决策能力，例如根据上下文动态生成检索查询、提议修改大纲、或选择不同的处理策略。
*   实现更复杂的循环和条件分支逻辑在Orchestrator中。
```
