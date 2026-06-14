# AgentBase

仿 DeepSeek 风格的本地 AI Agent 平台。支持多模型对话、知识库 RAG、MCP/HTTP 工具调用、联网搜索、深度思考推理、SSE流式输出与多轮会话持久化。自带账号体系、智能体市场和发布审核流程，开箱即用。

---

## 项目概述

AgentBase 是一个全栈智能体平台：
- **后端**：FastAPI + PostgreSQL + SQLAlchemy + Milvus + Redis
- **前端**：Vite + React (JSX) + ReactMarkdown + KaTeX
- 支持用户创建、配置和发布自定义 AI 智能体，在聊天页面中进行流式多轮对话
- 核心闭环：**注册 → 配置模型 → 创建智能体（知识库 + 工具 + 技能 + 提示词）→ 发布审核 → 市场复制 → 流式聊天（深度思考 + RAG + 联网搜索 + 附件）**

---

## 核心功能

### 账号与权限
- 本地注册/登录，JWT 鉴权
- 个人资料管理（名称、头像）
- `admin` / `user` 双角色：管理员审核智能体、管理系统模型、查看成员列表；普通用户管理自己的智能体
- 邀请制加入工作空间（`INVITE_API_ENABLED` 开关控制）
- API Key 加密存储（Fernet 对称加密，`API_KEY_ENCRYPTION_KEY` 必填）

### 模型配置
- **系统模型**：管理员预设共享模型配置（provider / model_name / 能力标注）
- **用户私有模型**：每个用户可配置自己的模型供应商（base_url + api_key + 模型名），支持任意 OpenAI 兼容接口
- 模型能力标注：`supports_text` / `supports_image` / `supports_document` / `supports_reasoning`
- 推理类型：`native`（DeepSeek 原生 thinking）/ `prompt`（提示词增强）/ `none`
- 内置兼容：DashScope（通义千问）、DeepSeek、任意 OpenAI Compatible API
- 用户可设置默认模型、在线测试模型连通性和多模态能力

### 智能体管理
- 创建/编辑/删除智能体，配置名称、头像、开场白、系统提示词、温度
- 选择系统模型或用户私有模型
- 绑定知识库（多对多）
- 绑定工具（多对多）
- 绑定技能（Skills，多对多，支持优先级排序）
- 自定义变量（string / number / boolean），聊天时动态填入
- 推荐问题列表
- 记忆画像配置（Memory Profile：开启/关闭、策略、最大消息数）
- RAG 配置（top_k / dense_top_k / bm25_top_k / rrf_k / rerank / cache / 拒答）
- 工具策略（auto 模式 + 工具白名单）
- **草稿模式**：创建者可在 Builder 页面调试聊天，使用当前草稿配置
- **发布 → 审核 → 内部市场**：admin 审核通过后进入市场，其他用户可一键复制
- **版本快照**：每次发布保存完整配置快照，支持版本历史查看，已发布聊天使用快照配置保证一致性

### 技能系统 (Skills)
- 创建/编辑/删除技能，每个技能可配置独立名称、描述、图标、系统提示词
- 技能可绑定工具和知识库
- 智能体可绑定多个技能，按优先级排序
- 运行时自动合并技能的系统提示词、工具、知识库到智能体配置中

### 提示词模板
- 创建/编辑/删除自定义模板（标题、内容、分类、标签）
- 内置模板库，支持一键复制到个人模板
- 聊天 Builder 页面可插入模板到系统提示词

### 工作流引擎
- 基于节点的可视化工作流：`Start → Knowledge → Tool → LLM → Answer`
- 每个智能体可自定义节点顺序
- 运行时每个节点产出 `run_step` 事件，并持久化到 `run_events` 供重连回放
- 支持草稿模式和发布模式的工作流分别存储

---

## 聊天交互

### 流式对话
- SSE（Server-Sent Events）流式输出，实时渲染
- 事件类型：`token` / `reasoning_token` / `tool_call_start` / `tool_call_result` / `search_status` / `rag_status` / `memory_used` / `thinking_status` / `sources` / `done` / `cancelled` / `error`
- 支持 Markdown 渲染（Github Flavored Markdown）、数学公式（KaTeX）、代码块（语法高亮 + 一键复制）
- 消息操作：复制回复内容、好评/差评反馈

### 深度思考 (Reasoning)
- 按模型能力可选开启，UI 显示思考状态和耗时
- 支持 `native` 推理（DeepSeek thinking）和 `prompt` 推理（提示词增强）两种模式
- 推理过程以可折叠时间线展示，包含思考内容 + 工具调用步骤

### 联网搜索
- 按需开启，DuckDuckGo HTML / Tavily / SerpAPI 三种后端
- 搜索结果以来源卡片展示
- 工具调用时间线中展示搜索摘要

### 多模态附件
- 支持图片上传/粘贴（base64 存储，data URL 传输）
- 支持文档上传（TXT / MD / CSV / PDF / DOCX），自动文本提取
- 模型能力检测：不支持图片/文档的模型自动禁用对应上传

### 会话管理
- 新建/切换/重命名/删除历史会话
- 新建会话自动生成标题：后台根据首条消息内容，使用当前运行时模型配置（解析代理模型和私有模型优先级），静默生成简洁会话标题
- 会话列表显示消息数量和时间
- 会话记忆：自动压缩历史对话为摘要，超长上下文注入
- 记忆画像（Memory Profile）：按用户 × 智能体维度持久化偏好、事实和摘要

---

## RAG 检索增强生成

### 检索管线
完整的混合检索 + 精排管线：

1. **Parent-Child Chunk 分割**：父块保留完整上下文（1600 字符），子块用于精确检索（560 字符，30% 重叠）
2. **Dense Retrieval**：向量相似度检索（Embedding → Milvus / 内存回退）
3. **中文 BM25**：jieba 分词 + rank-bm25 关键词稀疏检索
4. **RRF 融合**（Reciprocal Rank Fusion）：合并 Dense + BM25 排序结果
5. **可选 Rerank**：通过 OpenAI 兼容 rerank API（默认 `qwen3-rerank`）精排
6. **Redis 缓存**：相同 query 在 TTL 内直接返回缓存结果（`RAG_CACHE_TTL_SECONDS`）

### 知识库管理
- 创建/编辑/删除知识库（名称 + 描述）
- 文本直接录入 + 文件上传（TXT / MD / CSV / PDF / DOCX）
- 文档入库 → 文本提取 → 分段存储 → 写入向量库 + PostgreSQL
- 批量文档上传（最多 50 个/次）
- 文档列表、文档删除（同步清理 PostgreSQL chunks + 向量数据）
- 全量同步索引（reindex），异步作业状态查询（通过 Redis）
- **可配置分段策略**：
  - `auto`：默认 Parent-Child Chunk
  - `hierarchy`：按 Markdown 标题层级分段
  - `custom`：自定义父块大小、子块大小、重叠比例
- 分段预览：切换策略后实时预览分段结果
- 结构化引用来源展示（标题、页码、章节、检索通道、相关度分数）

### 证据不足拒答
- 当所有检索结果相关度低于阈值时，LLM 拒绝猜测，输出拒答提示
- 通过 `RAG_REFUSE_WHEN_NO_EVIDENCE` 控制

---

## 工具集成

### 内置工具
- `current_time`：获取当前时间（支持时区）
- `calculator`：安全数学表达式求值（AST 解析，白名单函数和运算符）
- `web_search`：联网搜索（需 `search_enabled=true`）
- `read_file` / `write_file`：本地文件读写（沙箱路径限制）
- `run_powershell`：执行 PowerShell 命令（超时控制 + 输出截断 + DNS pinning 防 SSRF）

### HTTP 工具
- 完整的 CRUD 管理，支持 GET / POST / PUT / PATCH / DELETE
- 请求头、查询参数、请求体 Schema 配置
- 多种认证方式：None / Bearer Token / API Key（Header / Query）
- API Key 加密存储
- 响应 JSONPath 提取（`$.data.items` 等）
- 超时设置（1-30 秒）
- SSRF 防护：禁止内网地址和云元数据地址，MCP HTTPS 白名单
- 在线测试：填入参数即时验证工具连通性和响应

### MCP 工具（Model Context Protocol）
- **Streamable HTTP 传输**：连接远程 MCP Server，自动发现工具列表
- **STDIO 传输**：本地进程通信（如 Playwright MCP 浏览器自动化）
  - 会话池化：同一聊天 session 复用 MCP 进程和浏览器状态
  - 自动生命周期管理：空闲 TTL 回收、异常自愈
  - 独立隔离：不同 session_key 获得完全独立的浏览器实例
- MCP 工具可配置输入 Schema、超时和认证头
- 一键发现（Discover）：输入 MCP Server URL，自动列出可用工具

### 工具策略
- `auto` 模式：LLM 自主决定是否调用工具
- `allowed_tool_names` 白名单：限制工具调用范围
- 工具调用限制：单次运行最多 200 次工具调用、50 轮工具交互、1800 秒总时长
- **并发执行**：同一轮次的多个工具调用通过 `ThreadPoolExecutor` 线程池并发执行（上限 8 并发），完成后按原始顺序处理结果，提升多工具场景下的响应速度
- 运行追踪：每次工具调用写入 `run_events`，包含输入、输出、耗时、状态

---

## 生成控制

### 停止生成
- 前端显示红色停止按钮（生成中可见），点击后：
  1. 调用 `POST /api/runs/{run_id}/cancel` 通知后端
  2. 后端设置取消事件 + 关闭 LLM HTTP 连接
  3. 工作流检测取消 → 标记 `run.status = "cancelled"` → 保存部分回复
  4. 前端收到 `cancelled` SSE 事件 → 更新 UI
- 所有流式/非流式执行路径均支持取消检测（LLM 节点、Tool 节点、draft 分块输出）

### 会话恢复与重连
- 后台线程执行工作流，与 SSE 连接解耦
- 页面刷新/关闭 → SSE 连接断开 → 后台线程继续执行
- 重新打开会话时：
  1. 从数据库加载历史消息
  2. 检测 `active_run`（session 下正在运行的后台 run）
  3. 若存在，通过 `GET /api/runs/{run_id}/events` 重新订阅 SSE 流
  4. 从数据库回放 `run_events`，继续接收后续 token
- 事件日志持久化到 `run_events`，刷新页面或 SSE 断开后可恢复 timeline
- 僵尸 Run 清理：启动时标记 30 分钟以上的 `running` Run 为 `failed`

### DSML 工具调用
- 原生工具调用之外的补充方案：通过 `<||DSML||tool_calls>` / `<||DSML||invoke>` / `<||DSML||parameter>` XML 标记
- 流式输出中自动检测和拦截 DSML 标记，防止泄露到前端
- 完整的 DSML 解析器：提取工具名称和参数，转换为标准工具调用格式

---

## 数据库模型

| 表 | 说明 |
|---|------|
| `users` | 用户（email / name / password_hash / avatar_url） |
| `workspaces` | 工作空间（name / slug） |
| `workspace_members` | 成员关系（user_id / workspace_id / role） |
| `workspace_invites` | 邀请（token / email / role / accepted_at） |
| `agents` | 智能体（name / system_prompt / model / temperature / status） |
| `agent_versions` | 发布版本快照（完整 JSON snapshot） |
| `agent_settings` | 智能体设置（推荐问题 / 变量 / 记忆 / RAG / 工具策略） |
| `agent_tools` | 智能体-工具绑定 |
| `agent_knowledge_base` | 智能体-知识库绑定 |
| `agent_skills` | 智能体-技能绑定（优先级排序） |
| `agent_memory_profiles` | 记忆画像 |
| `skills` | 技能（name / system_prompt / icon / category） |
| `skill_tools` | 技能-工具绑定 |
| `skill_knowledge_base` | 技能-知识库绑定 |
| `tools` | 工具（type / method / url / auth / mcp_config / schema） |
| `model_configs` | 系统模型配置 |
| `user_model_configs` | 用户私有模型配置 |
| `prompt_templates` | 提示词模板 |
| `knowledge_bases` | 知识库 |
| `knowledge_documents` | 文档（status / text / chunk_count / segment_config） |
| `knowledge_chunks` | 文档分段（parent_id / chunk_id / embedding_model / metadata） |
| `sessions` | 会话（agent_id / user_id / title） |
| `messages` | 消息（role / content / reasoning / tool_calls / sources） |
| `runs` | 运行记录（status / started_at / completed_at） |
| `run_events` | 运行事件（sequence / event / payload / sse） |
| `feedback` | 消息反馈（rating / comment） |
| `uploads` | 文件上传 |
| `workflow_definitions` | 工作流定义 |

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI (Python 3.11) | 60+ API 端点，Uvicorn 服务器 |
| 前端 | Vite + React (JSX) | 固定端口 `127.0.0.1:5174`，组件化 SPA |
| 数据库 | PostgreSQL | SQLAlchemy 2.0 ORM，完整关系模型 |
| 向量存储 | Milvus / 内存回退 | `AGENTBASE_VECTOR_BACKEND` 切换 |
| 缓存 | Redis | RAG 缓存 + 索引作业状态 |
| LLM 网关 | OpenAI 兼容 HTTP API | DashScope / DeepSeek / 自定义 |
| Embedding | OpenAI 兼容 API | `text-embedding-v4` |
| Rerank | OpenAI 兼容 API | `qwen3-rerank` |
| 中文分词 | jieba | BM25 检索分词 |
| 关键词检索 | rank-bm25 | 稀疏检索 |
| MCP 客户端 | mcp + asyncio | STDIO 会话池化 + HTTP SSE |
| 文档解析 | PyPDF / zipfile + xml | PDF + DOCX 文本提取 |
| 网络搜索 | DuckDuckGo / Tavily / SerpAPI | 可切换后端 |
| Markdown | react-markdown + remark-gfm + rehype-katex | GFM + 数学公式 |
| 图标 | lucide-react | 统一图标库 |
| 测试 | PyTest | 单元测试 + 集成测试 + RAG 评测 |
| 容器化 | Docker Compose | PostgreSQL + Redis + Milvus + API + 前端 |

---

## 项目结构

```
├── api/                            # FastAPI 应用层
│   ├── main.py                     #   60+ 端点 + SSE 流式聊天 + 取消 + 重连
│   ├── deps.py                     #   依赖注入（当前用户/成员/工作空间）
│   └── schemas.py                  #   Pydantic 请求/响应模型
├── core/                           # 核心业务层
│   ├── config.py                   #   Pydantic Settings 配置（40+ 环境变量）
│   ├── db/
│   │   ├── models.py               #   25 个 SQLAlchemy ORM 模型
│   │   ├── base.py                 #   Declarative Base
│   │   └── session.py              #   会话工厂 + init_db + 兼容迁移
│   ├── integrations/
│   │   ├── llm.py                  #   OpenAI 兼容网关（chat / stream / embed / rerank）
│   │   ├── mcp_client.py           #   MCP 客户端（STDIO 池化 + HTTP SSE）
│   │   └── vector_store.py         #   向量存储抽象（Milvus + 内存回退）
│   ├── runtime/
│   │   ├── workflow.py             #   WorkflowRunner 节点式执行引擎
│   │   └── cancel.py               #   线程安全取消注册表
│   ├── security/
│   │   ├── auth.py                 #   JWT 创建/验证 + 密码哈希
│   │   ├── api_keys.py             #   Fernet API Key 加密存储
│   │   └── permissions.py          #   角色鉴权
│   └── services/
│       ├── agents.py               #   智能体 CRUD + 发布/审核/市场复制
│       ├── bootstrap.py            #   首次启动初始化
│       ├── knowledge.py            #   知识库 + 文档 + 索引 + 分段
│       ├── memory.py               #   会话摘要 + Memory Profile
│       ├── models.py               #   系统模型管理
│       ├── prompt_templates.py     #   提示词模板管理
│       ├── rag.py                  #   RAG 检索管线（Dense+BM25+RRF+Rerank+Cache）
│       ├── rag_cache.py            #   Redis RAG 缓存
│       ├── skills.py               #   技能 CRUD + 关联管理
│       ├── tools.py                #   工具 CRUD + 执行 + MCP 发现 + HTTP 测试
│       ├── uploads.py              #   文件上传管理
│       ├── user_models.py          #   用户私有模型管理
│       └── web_search.py           #   联网搜索（DuckDuckGo / Tavily / SerpAPI）
├── frontend/src/
│   ├── main.jsx                    #   App 根组件（状态管理 + SSE + 路由）
│   ├── utils.js                    #   工具函数 + API 封装
│   ├── styles.css                  #   全局样式
│   ├── views/
│   │   ├── ChatView.jsx            #   聊天页（ChatComposer + MessageList + 开关）
│   │   └── BuilderView.jsx         #   智能体配置页（多 Tab 构建器）
│   └── components/
│       ├── MessageList.jsx         #   消息列表（Markdown + 推理时间线 + 引用来源）
│       ├── AgentAvatar.jsx         #   智能体头像
│       ├── KnowledgeBaseDialog.jsx #   知识库管理弹窗
│       ├── KnowledgeDocumentList.jsx # 文档列表
│       ├── PromptTemplateDialog.jsx #  提示词模板弹窗
│       ├── SkillDialog.jsx         #   技能管理弹窗
│       └── ResegmentModal.jsx      #   分段策略配置弹窗
├── tests/                          # 测试
├── eval/                           # RAG 评测
├── scripts/                        # 工具脚本
├── docker-compose.yml              # 一键部署（PostgreSQL + Redis + Milvus + API + 前端）
├── Dockerfile.api                  # API 容器镜像
├── requirements.txt                # Python 依赖
└── .env.example                    # 环境变量模板
```

---

## 快速启动

### 环境准备

```powershell
conda create -n agentbase python=3.11 -y
conda activate agentbase
node --version   # 需要 >= 18
```

### 基础设施

```powershell
# 启动 PostgreSQL + Redis + Milvus（仅数据库，API 和前端手动运行）
docker compose up -d postgres redis milvus
```

开发阶段也可不装 Docker，使用内存向量模式 + 无 Redis：

```env
AGENTBASE_VECTOR_BACKEND=memory
# 不设置 REDIS_URL
```

### 后端

```powershell
cp .env.example .env                # 编辑 .env，填写 LLM API Key
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

首次启动自动创建数据库表、默认工作空间、内置工具和系统模型。

### 前端

```powershell
cd frontend
npm install
npm run dev                         # http://127.0.0.1:5174
```

### 端口

| 服务 | 地址 |
|------|------|
| Backend API | `http://127.0.0.1:8000` |
| Frontend Dev | `http://127.0.0.1:5174` |

---

## 关键环境变量

```env
# ── 安全 ──
JWT_SECRET=change-me-in-production          # JWT 签名密钥（生产环境必须修改）
API_KEY_ENCRYPTION_KEY=                     # Fernet 加密密钥（用于存储用户 API Key）

# ── 数据库 ──
DATABASE_URL=postgresql+psycopg://agentbase:agentbase@localhost:5433/agentbase

# ── LLM ──
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=                           # 通义千问
OPENAI_MODEL=qwen-plus
OPENAI_EMBEDDING_MODEL=text-embedding-v4
DEEPSEEK_API_KEY=                            # DeepSeek（可选）
DEEPSEEK_MODEL=deepseek-chat

# ── 向量检索 ──
AGENTBASE_VECTOR_BACKEND=memory              # memory | milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=agentbase_chunks

# ── RAG ──
RAG_TOP_K=4                                  # 最终返回文档数
RAG_DENSE_TOP_K=12                           # Dense 检索候选数
RAG_BM25_TOP_K=12                            # BM25 检索候选数
RAG_RRF_K=60                                 # RRF 融合参数
RAG_RERANK_ENABLED=true                      # 启用 Rerank
RAG_RERANK_MODEL=qwen3-rerank
RAG_REFUSE_WHEN_NO_EVIDENCE=true             # 证据不足时拒答

# ── 联网搜索 ──
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo_html          # duckduckgo_html | tavily | serpapi

# ── 其他 ──
AGENTBASE_MOCK_LLM=false                     # Mock 模式（无 API Key 时调试前端）
INVITE_API_ENABLED=false
```

---

## 主要 API

| 模块 | 端点示例 | 说明 |
|------|---------|------|
| Auth | `POST /api/auth/register` `POST /api/auth/login` `GET /api/auth/me` | 注册/登录/个人资料 |
| Workspace | `GET /api/workspaces/current` `GET /api/workspaces/members` | 工作空间与成员 |
| Models | `GET /api/models` `POST/PATCH/DELETE /api/admin/models` | 系统模型 |
| User Models | `GET/POST/PATCH/DELETE /api/user-models` `POST /api/user-models/test` | 私有模型 |
| Agents | `GET/POST /api/agents` `GET/PATCH/DELETE /api/agents/{id}` | 智能体 CRUD |
| Publish | `POST /api/agents/{id}/publish` | 发布（需审核） |
| Review | `GET /api/admin/agent-reviews` `POST .../approve` `POST .../reject` | 审核 |
| Market | `GET /api/market/agents` `POST /api/market/agents/{id}/copy` | 内部市场 |
| Skills | `GET/POST /api/skills` `PATCH/DELETE /api/skills/{id}` | 技能管理 |
| Memory | `GET/PATCH/DELETE /api/agents/{id}/memory-profile` | 记忆画像 |
| Prompt | `GET/POST /api/prompt-templates` `POST .../copy-builtin` | 提示词模板 |
| Workflow | `GET/PATCH /api/agents/{id}/workflow` | 工作流配置 |
| Chat | `POST /api/agents/{id}/chat/stream` | SSE 流式聊天 |
| Sessions | `GET /api/agents/{id}/sessions` `PATCH/DELETE /api/sessions/{id}` | 会话管理 |
| Runs | `GET /api/runs/{id}` `POST .../cancel` `GET .../events` | 运行记录/取消/重连 |
| Feedback | `POST /api/messages/{id}/feedback` | 消息反馈 |
| Knowledge | `GET/POST/DELETE /api/knowledge-bases` `POST .../documents` `POST .../index` `POST .../preview` | 知识库管理 |
| Tools | `GET/POST /api/tools` `POST .../mcp/discover` `POST .../test` | 工具管理 |
| Uploads | `POST /api/uploads` | 文件上传 |
| Search | `GET /api/search/test` | 联网搜索测试 |
| Health | `GET /api/health` | 健康检查 |

---

## 架构

```
浏览器 (React SPA)
    │  HTTP REST + SSE
    ▼
FastAPI ── JWT 鉴权 ──→ CRUD API (Agents / Knowledge / Tools / Models / Skills)
    │
    └──→ POST /chat/stream
           │
           ├─→ 创建 session + user_message（请求级 DB）
           ├─→ 启动后台线程（独立 DB Session）
           │      │
           │      └─→ WorkflowRunner.run_events()
           │             │
           │             ├─ [Start]     接收输入 + 附件 + 记忆 + 变量
           │             ├─ [Knowledge]  RAG 检索 (Dense + BM25 + RRF + Rerank + Cache)
           │             ├─ [Tool]       LLM 决策 → 执行工具 (HTTP / MCP / Builtin)
           │             ├─ [LLM]        流式调用 LLM 生成候选回答
           │             └─ [Answer]     输出最终回答 + 引用来源
           │             │
           │             ├─→ PostgreSQL (消息/Run/run_events/会话记忆)
           │             ├─→ Milvus / 内存 (向量检索)
           │             ├─→ Redis (RAG 缓存)
           │             └─→ LLM Provider (DashScope / DeepSeek / 自定义)
           │
           ├─→ Event Queue ──→ SSE Generator ──→ 客户端
           └─→ Event Log Buffer（重连回放）
```

---

## 许可证

MIT
