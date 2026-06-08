-- Lingshu Agent complete PostgreSQL schema.
-- Source of truth: docs/storage-design.md, docs/api-design.md,
-- docs/product-requirements.md, and docs/agent-capability-plan.md.
--
-- Target runtime: PostgreSQL 16 from docker-compose.yml.

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(120) NOT NULL,
    avatar_url TEXT NOT NULL DEFAULT '',
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

CREATE TABLE IF NOT EXISTS workspaces (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(160) NOT NULL,
    slug VARCHAR(160) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_workspaces_slug ON workspaces (slug);

CREATE TABLE IF NOT EXISTS workspace_members (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_workspace_member UNIQUE (workspace_id, user_id)
);
CREATE INDEX IF NOT EXISTS ix_workspace_members_workspace_id ON workspace_members (workspace_id);
CREATE INDEX IF NOT EXISTS ix_workspace_members_user_id ON workspace_members (user_id);

CREATE TABLE IF NOT EXISTS workspace_invites (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'user')),
    token VARCHAR(80) NOT NULL UNIQUE,
    accepted_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_workspace_invites_workspace_id ON workspace_invites (workspace_id);
CREATE INDEX IF NOT EXISTS ix_workspace_invites_token ON workspace_invites (token);

CREATE TABLE IF NOT EXISTS model_configs (
    id BIGSERIAL PRIMARY KEY,
    provider VARCHAR(80) NOT NULL DEFAULT 'openai-compatible' CHECK (provider = 'openai-compatible'),
    model_name VARCHAR(160) NOT NULL UNIQUE,
    display_name VARCHAR(160) NOT NULL,
    supports_text BOOLEAN NOT NULL DEFAULT TRUE,
    supports_image BOOLEAN NOT NULL DEFAULT FALSE,
    supports_document BOOLEAN NOT NULL DEFAULT TRUE,
    supports_reasoning BOOLEAN NOT NULL DEFAULT FALSE,
    reasoning_type VARCHAR(20) NOT NULL DEFAULT 'none' CHECK (reasoning_type IN ('native', 'prompt', 'none')),
    reasoning_label VARCHAR(80) NOT NULL DEFAULT '不支持',
    max_context INTEGER NOT NULL DEFAULT 8192 CHECK (max_context > 0),
    default_temperature DOUBLE PRECISION NOT NULL DEFAULT 0.4 CHECK (default_temperature >= 0 AND default_temperature <= 2),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_model_configs_model_name ON model_configs (model_name);
CREATE INDEX IF NOT EXISTS ix_model_configs_enabled ON model_configs (enabled);

CREATE TABLE IF NOT EXISTS user_model_configs (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    display_name VARCHAR(160) NOT NULL,
    provider VARCHAR(80) NOT NULL DEFAULT 'openai-compatible' CHECK (provider = 'openai-compatible'),
    base_url VARCHAR(500) NOT NULL CHECK (length(trim(base_url)) > 0),
    encrypted_api_key TEXT NOT NULL CHECK (length(trim(encrypted_api_key)) > 0),
    chat_model VARCHAR(160) NOT NULL CHECK (length(trim(chat_model)) > 0),
    supports_image BOOLEAN NOT NULL DEFAULT FALSE,
    supports_document BOOLEAN NOT NULL DEFAULT TRUE,
    supports_reasoning BOOLEAN NOT NULL DEFAULT FALSE,
    reasoning_type VARCHAR(20) NOT NULL DEFAULT 'none' CHECK (reasoning_type IN ('native', 'prompt', 'none')),
    reasoning_label VARCHAR(80) NOT NULL DEFAULT '不支持',
    max_context INTEGER NOT NULL DEFAULT 131072 CHECK (max_context > 0),
    default_temperature DOUBLE PRECISION NOT NULL DEFAULT 0.4 CHECK (default_temperature >= 0 AND default_temperature <= 2),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_user_model_configs_user_id ON user_model_configs (user_id);
CREATE INDEX IF NOT EXISTS ix_user_model_configs_enabled ON user_model_configs (user_id, enabled);
CREATE UNIQUE INDEX IF NOT EXISTS ix_user_model_configs_one_default_per_user
    ON user_model_configs (user_id)
    WHERE is_default = TRUE;

CREATE TABLE IF NOT EXISTS agents (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    model_id BIGINT NULL REFERENCES model_configs(id) ON DELETE SET NULL,
    user_model_config_id BIGINT NULL REFERENCES user_model_configs(id) ON DELETE SET NULL,
    name VARCHAR(160) NOT NULL,
    avatar VARCHAR(40) NOT NULL DEFAULT 'SA',
    description TEXT NOT NULL DEFAULT '',
    opening_message TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    model VARCHAR(120) NOT NULL DEFAULT 'qwen-plus',
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.4 CHECK (temperature >= 0 AND temperature <= 2),
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'pending_review', 'published', 'rejected')),
    published_version_id BIGINT NULL,
    is_template BOOLEAN NOT NULL DEFAULT FALSE,
    created_by BIGINT NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_agents_workspace_id ON agents (workspace_id);
CREATE INDEX IF NOT EXISTS ix_agents_model_id ON agents (model_id);
CREATE INDEX IF NOT EXISTS ix_agents_user_model_config_id ON agents (user_model_config_id);
CREATE INDEX IF NOT EXISTS ix_agents_created_by ON agents (created_by);
CREATE INDEX IF NOT EXISTS ix_agents_status ON agents (workspace_id, status);
CREATE INDEX IF NOT EXISTS ix_agents_published_version_id ON agents (published_version_id);

CREATE TABLE IF NOT EXISTS agent_versions (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    snapshot JSONB NOT NULL,
    created_by BIGINT NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_agent_versions_version UNIQUE (agent_id, version)
);
CREATE INDEX IF NOT EXISTS ix_agent_versions_agent_id ON agent_versions (agent_id);
CREATE INDEX IF NOT EXISTS ix_agent_versions_created_by ON agent_versions (created_by);
CREATE INDEX IF NOT EXISTS ix_agent_versions_snapshot_model_id ON agent_versions ((snapshot->>'model_id'));
CREATE INDEX IF NOT EXISTS ix_agent_versions_snapshot_user_model_config_id ON agent_versions ((snapshot->>'user_model_config_id'));

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_agents_published_version'
    ) THEN
        ALTER TABLE agents
            ADD CONSTRAINT fk_agents_published_version
            FOREIGN KEY (published_version_id) REFERENCES agent_versions(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS agent_settings (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL UNIQUE REFERENCES agents(id) ON DELETE CASCADE,
    suggested_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
    variables JSONB NOT NULL DEFAULT '[]'::jsonb,
    memory JSONB NOT NULL DEFAULT '{}'::jsonb,
    rag JSONB NOT NULL DEFAULT '{}'::jsonb,
    tool_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    skill_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_agent_settings_agent_id ON agent_settings (agent_id);

CREATE TABLE IF NOT EXISTS tools (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(40) NOT NULL DEFAULT 'builtin' CHECK (type IN ('builtin', 'builtin_search', 'http', 'mcp')),
    name VARCHAR(120) NOT NULL CHECK (name ~ '^[a-z][a-z0-9_]*$'),
    label VARCHAR(160) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    server_label VARCHAR(160) NOT NULL DEFAULT '',
    schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    method VARCHAR(10) NULL CHECK (method IS NULL OR method IN ('GET', 'POST', 'PUT', 'PATCH', 'DELETE')),
    url TEXT NULL,
    headers_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    query_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    body_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    auth_type VARCHAR(40) NOT NULL DEFAULT 'none' CHECK (auth_type IN ('none', 'bearer', 'header', 'query')),
    auth_header_name VARCHAR(120) NULL,
    auth_query_name VARCHAR(120) NULL,
    encrypted_secret TEXT NULL,
    response_path VARCHAR(255) NOT NULL DEFAULT '$',
    timeout_seconds INTEGER NOT NULL DEFAULT 10 CHECK (timeout_seconds >= 1 AND timeout_seconds <= 30),
    search_options JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (type <> 'http' OR url LIKE 'https://%')
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_tools_global_name
    ON tools (name)
    WHERE workspace_id IS NULL AND user_id IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_tools_owner_workspace_name
    ON tools (workspace_id, user_id, name)
    WHERE workspace_id IS NOT NULL AND user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_tools_workspace_id ON tools (workspace_id);
CREATE INDEX IF NOT EXISTS ix_tools_user_id ON tools (user_id);
CREATE INDEX IF NOT EXISTS ix_tools_type ON tools (type);
CREATE INDEX IF NOT EXISTS ix_tools_enabled ON tools (enabled);

CREATE TABLE IF NOT EXISTS agent_tools (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tool_id BIGINT NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_agent_tool UNIQUE (agent_id, tool_id)
);
CREATE INDEX IF NOT EXISTS ix_agent_tools_agent_id ON agent_tools (agent_id);
CREATE INDEX IF NOT EXISTS ix_agent_tools_tool_id ON agent_tools (tool_id);
CREATE INDEX IF NOT EXISTS ix_agent_tools_enabled ON agent_tools (enabled);

CREATE TABLE IF NOT EXISTS skills (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id),
    name VARCHAR(160) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    icon VARCHAR(40) NOT NULL DEFAULT 'SK',
    category VARCHAR(80) NOT NULL DEFAULT 'general',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    rag_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    memory_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_skills_workspace_id ON skills (workspace_id);
CREATE INDEX IF NOT EXISTS ix_skills_user_id ON skills (user_id);
CREATE INDEX IF NOT EXISTS ix_skills_category ON skills (category);
CREATE INDEX IF NOT EXISTS ix_skills_enabled ON skills (enabled);

CREATE TABLE IF NOT EXISTS skill_tools (
    id BIGSERIAL PRIMARY KEY,
    skill_id BIGINT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    tool_id BIGINT NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    CONSTRAINT uq_skill_tool UNIQUE (skill_id, tool_id)
);
CREATE INDEX IF NOT EXISTS ix_skill_tools_skill_id ON skill_tools (skill_id);
CREATE INDEX IF NOT EXISTS ix_skill_tools_tool_id ON skill_tools (tool_id);

CREATE TABLE IF NOT EXISTS skill_knowledge_bases (
    id BIGSERIAL PRIMARY KEY,
    skill_id BIGINT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    knowledge_base_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    CONSTRAINT uq_skill_knowledge_base UNIQUE (skill_id, knowledge_base_id)
);
CREATE INDEX IF NOT EXISTS ix_skill_knowledge_bases_skill_id ON skill_knowledge_bases (skill_id);
CREATE INDEX IF NOT EXISTS ix_skill_knowledge_bases_knowledge_base_id ON skill_knowledge_bases (knowledge_base_id);

CREATE TABLE IF NOT EXISTS agent_skills (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    skill_id BIGINT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    priority INTEGER NOT NULL DEFAULT 0,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT uq_agent_skill UNIQUE (agent_id, skill_id)
);
CREATE INDEX IF NOT EXISTS ix_agent_skills_agent_id ON agent_skills (agent_id);
CREATE INDEX IF NOT EXISTS ix_agent_skills_skill_id ON agent_skills (skill_id);
CREATE INDEX IF NOT EXISTS ix_agent_skills_enabled ON agent_skills (enabled);

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name VARCHAR(160) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_by BIGINT NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_knowledge_bases_workspace_id ON knowledge_bases (workspace_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_bases_created_by ON knowledge_bases (created_by);

CREATE TABLE IF NOT EXISTS agent_knowledge_bases (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    knowledge_base_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    CONSTRAINT uq_agent_kb UNIQUE (agent_id, knowledge_base_id)
);
CREATE INDEX IF NOT EXISTS ix_agent_knowledge_bases_agent_id ON agent_knowledge_bases (agent_id);
CREATE INDEX IF NOT EXISTS ix_agent_knowledge_bases_knowledge_base_id ON agent_knowledge_bases (knowledge_base_id);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id BIGSERIAL PRIMARY KEY,
    knowledge_base_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(80) NOT NULL,
    source_type VARCHAR(40) NOT NULL DEFAULT 'text' CHECK (source_type IN ('text', 'file')),
    text TEXT NOT NULL,
    text_preview TEXT NOT NULL DEFAULT '',
    status VARCHAR(20) NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'indexing', 'indexed', 'failed')),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_knowledge_documents_knowledge_base_id ON knowledge_documents (knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_documents_status ON knowledge_documents (status);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    knowledge_base_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL,
    vector_id VARCHAR(120) NOT NULL UNIQUE,
    parent_id VARCHAR(120) NOT NULL DEFAULT '',
    chunk_id VARCHAR(120) NOT NULL DEFAULT '',
    title VARCHAR(255) NOT NULL DEFAULT '',
    page INTEGER,
    section VARCHAR(255) NOT NULL DEFAULT '',
    content_hash VARCHAR(80) NOT NULL DEFAULT '',
    embedding_model VARCHAR(160) NOT NULL DEFAULT '',
    embedding_dimension INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_workspace_id ON knowledge_chunks (workspace_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_knowledge_base_id ON knowledge_chunks (knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_document_id ON knowledge_chunks (document_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_vector_id ON knowledge_chunks (vector_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_parent_id ON knowledge_chunks (parent_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_chunk_id ON knowledge_chunks (chunk_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_content_hash ON knowledge_chunks (content_hash);

CREATE TABLE IF NOT EXISTS workflow_definitions (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL UNIQUE REFERENCES agents(id) ON DELETE CASCADE,
    nodes JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_workflow_definitions_agent_id ON workflow_definitions (agent_id);

CREATE TABLE IF NOT EXISTS uploads (
    id VARCHAR(80) PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id),
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(120) NOT NULL,
    kind VARCHAR(30) NOT NULL CHECK (kind IN ('image', 'document')),
    data_url TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0 CHECK (size >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_uploads_workspace_id ON uploads (workspace_id);
CREATE INDEX IF NOT EXISTS ix_uploads_user_id ON uploads (user_id);
CREATE INDEX IF NOT EXISTS ix_uploads_kind ON uploads (kind);

CREATE TABLE IF NOT EXISTS sessions (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id),
    title VARCHAR(200) NOT NULL DEFAULT 'New conversation',
    is_debug BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_sessions_workspace_id ON sessions (workspace_id);
CREATE INDEX IF NOT EXISTS ix_sessions_agent_id ON sessions (agent_id);
CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id);
CREATE INDEX IF NOT EXISTS ix_sessions_agent_user_updated ON sessions (agent_id, user_id, updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    reasoning TEXT NOT NULL DEFAULT '',
    reasoning_duration_ms INTEGER NULL,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_messages_session_id ON messages (session_id);
CREATE INDEX IF NOT EXISTS ix_messages_role ON messages (role);
CREATE INDEX IF NOT EXISTS ix_messages_created_at ON messages (created_at);

CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    session_id BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ NULL
);
CREATE INDEX IF NOT EXISTS ix_runs_workspace_id ON runs (workspace_id);
CREATE INDEX IF NOT EXISTS ix_runs_agent_id ON runs (agent_id);
CREATE INDEX IF NOT EXISTS ix_runs_session_id ON runs (session_id);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs (status);

CREATE TABLE IF NOT EXISTS run_steps (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    node_id VARCHAR(80) NOT NULL,
    node_type VARCHAR(40) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('started', 'running', 'succeeded', 'failed', 'blocked')),
    input JSONB NOT NULL DEFAULT '{}'::jsonb,
    output JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_run_steps_run_id ON run_steps (run_id);
CREATE INDEX IF NOT EXISTS ix_run_steps_status ON run_steps (status);
CREATE INDEX IF NOT EXISTS ix_run_steps_node_type ON run_steps (node_type);

CREATE TABLE IF NOT EXISTS session_memory (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0 CHECK (message_count >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_session_memory_session_id ON session_memory (session_id);

CREATE TABLE IF NOT EXISTS agent_memory_profiles (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    summary TEXT NOT NULL DEFAULT '' CHECK (length(summary) <= 4000),
    facts JSONB NOT NULL DEFAULT '[]'::jsonb,
    preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_agent_memory_profile_scope UNIQUE (workspace_id, user_id, agent_id)
);
CREATE INDEX IF NOT EXISTS ix_agent_memory_profiles_workspace_id ON agent_memory_profiles (workspace_id);
CREATE INDEX IF NOT EXISTS ix_agent_memory_profiles_user_id ON agent_memory_profiles (user_id);
CREATE INDEX IF NOT EXISTS ix_agent_memory_profiles_agent_id ON agent_memory_profiles (agent_id);
CREATE INDEX IF NOT EXISTS ix_agent_memory_profiles_enabled ON agent_memory_profiles (enabled);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(160) NOT NULL CHECK (length(trim(title)) > 0),
    description TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    category VARCHAR(80) NOT NULL DEFAULT 'general',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_prompt_templates_owner_title UNIQUE (workspace_id, user_id, title)
);
CREATE INDEX IF NOT EXISTS ix_prompt_templates_workspace_id ON prompt_templates (workspace_id);
CREATE INDEX IF NOT EXISTS ix_prompt_templates_user_id ON prompt_templates (user_id);
CREATE INDEX IF NOT EXISTS ix_prompt_templates_category ON prompt_templates (category);
CREATE INDEX IF NOT EXISTS ix_prompt_templates_enabled ON prompt_templates (enabled);

CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id),
    rating VARCHAR(20) NOT NULL CHECK (rating IN ('positive', 'negative', 'none')),
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_feedback_message_user UNIQUE (message_id, user_id)
);
CREATE INDEX IF NOT EXISTS ix_feedback_message_id ON feedback (message_id);
CREATE INDEX IF NOT EXISTS ix_feedback_user_id ON feedback (user_id);

-- Column comments.
COMMENT ON COLUMN users.id IS '用户主键';
COMMENT ON COLUMN users.email IS '用户邮箱，唯一登录标识';
COMMENT ON COLUMN users.name IS '用户显示名称';
COMMENT ON COLUMN users.avatar_url IS '用户头像地址';
COMMENT ON COLUMN users.password_hash IS '用户密码哈希值';
COMMENT ON COLUMN users.is_active IS '用户是否启用';
COMMENT ON COLUMN users.created_at IS '用户创建时间';

COMMENT ON COLUMN workspaces.id IS '工作区主键';
COMMENT ON COLUMN workspaces.name IS '工作区名称';
COMMENT ON COLUMN workspaces.slug IS '工作区唯一标识 slug';
COMMENT ON COLUMN workspaces.created_at IS '工作区创建时间';

COMMENT ON COLUMN workspace_members.id IS '工作区成员关系主键';
COMMENT ON COLUMN workspace_members.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN workspace_members.user_id IS '成员用户 ID';
COMMENT ON COLUMN workspace_members.role IS '成员角色';
COMMENT ON COLUMN workspace_members.created_at IS '加入工作区时间';

COMMENT ON COLUMN workspace_invites.id IS '工作区邀请主键';
COMMENT ON COLUMN workspace_invites.workspace_id IS '目标工作区 ID';
COMMENT ON COLUMN workspace_invites.email IS '被邀请邮箱';
COMMENT ON COLUMN workspace_invites.role IS '邀请后授予的角色';
COMMENT ON COLUMN workspace_invites.token IS '邀请令牌';
COMMENT ON COLUMN workspace_invites.accepted_at IS '邀请接受时间';
COMMENT ON COLUMN workspace_invites.created_at IS '邀请创建时间';

COMMENT ON COLUMN model_configs.id IS '模型配置主键';
COMMENT ON COLUMN model_configs.provider IS '模型提供方类型';
COMMENT ON COLUMN model_configs.model_name IS '模型唯一标识';
COMMENT ON COLUMN model_configs.display_name IS '模型展示名称';
COMMENT ON COLUMN model_configs.supports_text IS '是否支持文本输入输出';
COMMENT ON COLUMN model_configs.supports_image IS '是否支持图像能力';
COMMENT ON COLUMN model_configs.supports_document IS '是否支持文档能力';
COMMENT ON COLUMN model_configs.supports_reasoning IS '是否支持推理能力';
COMMENT ON COLUMN model_configs.reasoning_type IS '推理能力接入方式';
COMMENT ON COLUMN model_configs.reasoning_label IS '推理能力展示标签';
COMMENT ON COLUMN model_configs.max_context IS '最大上下文长度';
COMMENT ON COLUMN model_configs.default_temperature IS '默认生成温度';
COMMENT ON COLUMN model_configs.enabled IS '模型是否启用';
COMMENT ON COLUMN model_configs.created_at IS '模型配置创建时间';

COMMENT ON COLUMN user_model_configs.id IS '用户模型配置主键';
COMMENT ON COLUMN user_model_configs.user_id IS '所属用户 ID';
COMMENT ON COLUMN user_model_configs.display_name IS '用户自定义模型名称';
COMMENT ON COLUMN user_model_configs.provider IS '接入协议类型';
COMMENT ON COLUMN user_model_configs.base_url IS '模型服务基础地址';
COMMENT ON COLUMN user_model_configs.encrypted_api_key IS '加密后的 API Key';
COMMENT ON COLUMN user_model_configs.chat_model IS '对话模型名称';
COMMENT ON COLUMN user_model_configs.supports_image IS '是否支持图像能力';
COMMENT ON COLUMN user_model_configs.supports_document IS '是否支持文档能力';
COMMENT ON COLUMN user_model_configs.supports_reasoning IS '是否支持推理能力';
COMMENT ON COLUMN user_model_configs.reasoning_type IS '推理能力接入方式';
COMMENT ON COLUMN user_model_configs.reasoning_label IS '推理能力展示标签';
COMMENT ON COLUMN user_model_configs.max_context IS '最大上下文长度';
COMMENT ON COLUMN user_model_configs.default_temperature IS '默认生成温度';
COMMENT ON COLUMN user_model_configs.enabled IS '配置是否启用';
COMMENT ON COLUMN user_model_configs.is_default IS '是否为用户默认模型';
COMMENT ON COLUMN user_model_configs.created_at IS '用户模型配置创建时间';
COMMENT ON COLUMN user_model_configs.updated_at IS '用户模型配置更新时间';

COMMENT ON COLUMN agents.id IS '智能体主键';
COMMENT ON COLUMN agents.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN agents.model_id IS '关联的平台预置模型配置 ID';
COMMENT ON COLUMN agents.user_model_config_id IS '关联的用户模型配置 ID';
COMMENT ON COLUMN agents.name IS '智能体名称';
COMMENT ON COLUMN agents.avatar IS '智能体头像简称';
COMMENT ON COLUMN agents.description IS '智能体简介';
COMMENT ON COLUMN agents.opening_message IS '会话开场白';
COMMENT ON COLUMN agents.system_prompt IS '系统提示词';
COMMENT ON COLUMN agents.model IS '运行时模型名称快照';
COMMENT ON COLUMN agents.temperature IS '生成温度';
COMMENT ON COLUMN agents.status IS '智能体状态';
COMMENT ON COLUMN agents.published_version_id IS '当前已发布版本 ID';
COMMENT ON COLUMN agents.is_template IS '是否为模板智能体';
COMMENT ON COLUMN agents.created_by IS '创建人用户 ID';
COMMENT ON COLUMN agents.created_at IS '智能体创建时间';
COMMENT ON COLUMN agents.updated_at IS '智能体更新时间';

COMMENT ON COLUMN agent_versions.id IS '智能体版本主键';
COMMENT ON COLUMN agent_versions.agent_id IS '所属智能体 ID';
COMMENT ON COLUMN agent_versions.version IS '版本号';
COMMENT ON COLUMN agent_versions.snapshot IS '版本快照数据';
COMMENT ON COLUMN agent_versions.created_by IS '版本创建人用户 ID';
COMMENT ON COLUMN agent_versions.created_at IS '版本创建时间';

COMMENT ON COLUMN agent_settings.id IS '智能体设置主键';
COMMENT ON COLUMN agent_settings.agent_id IS '所属智能体 ID';
COMMENT ON COLUMN agent_settings.suggested_questions IS '建议提问列表';
COMMENT ON COLUMN agent_settings.variables IS '提示词变量定义列表';
COMMENT ON COLUMN agent_settings.memory IS '记忆配置';
COMMENT ON COLUMN agent_settings.rag IS 'RAG 配置';
COMMENT ON COLUMN agent_settings.tool_policy IS '工具调用策略配置';
COMMENT ON COLUMN agent_settings.updated_at IS '智能体设置更新时间';

COMMENT ON COLUMN tools.id IS '工具主键';
COMMENT ON COLUMN tools.workspace_id IS '所属工作区 ID，全局工具为空';
COMMENT ON COLUMN tools.user_id IS '所属用户 ID，全局工具为空';
COMMENT ON COLUMN tools.type IS '工具类型';
COMMENT ON COLUMN tools.name IS '工具唯一代码名';
COMMENT ON COLUMN tools.label IS '工具展示名称';
COMMENT ON COLUMN tools.description IS '工具描述';
COMMENT ON COLUMN tools.server_label IS 'MCP 服务展示名称或分组标签';
COMMENT ON COLUMN tools.schema IS '工具输入参数 Schema';
COMMENT ON COLUMN tools.enabled IS '工具是否启用';
COMMENT ON COLUMN tools.method IS 'HTTP 请求方法';
COMMENT ON COLUMN tools.url IS 'HTTP 工具请求地址';
COMMENT ON COLUMN tools.headers_schema IS '请求头参数 Schema';
COMMENT ON COLUMN tools.query_schema IS '查询参数 Schema';
COMMENT ON COLUMN tools.body_schema IS '请求体参数 Schema';
COMMENT ON COLUMN tools.auth_type IS '认证方式';
COMMENT ON COLUMN tools.auth_header_name IS '认证请求头名称';
COMMENT ON COLUMN tools.auth_query_name IS '认证查询参数名称';
COMMENT ON COLUMN tools.encrypted_secret IS '加密后的工具密钥';
COMMENT ON COLUMN tools.response_path IS '响应结果提取路径';
COMMENT ON COLUMN tools.timeout_seconds IS '请求超时时间（秒）';
COMMENT ON COLUMN tools.search_options IS '搜索类工具附加配置';
COMMENT ON COLUMN tools.created_at IS '工具创建时间';
COMMENT ON COLUMN tools.updated_at IS '工具更新时间';

COMMENT ON COLUMN agent_tools.id IS '智能体工具绑定主键';
COMMENT ON COLUMN agent_tools.agent_id IS '智能体 ID';
COMMENT ON COLUMN agent_tools.tool_id IS '工具 ID';
COMMENT ON COLUMN agent_tools.enabled IS '是否对该智能体启用工具';
COMMENT ON COLUMN agent_tools.config IS '智能体维度的工具配置';
COMMENT ON COLUMN agent_tools.created_at IS '绑定创建时间';

COMMENT ON TABLE skills IS 'Skill 可复用能力包';
COMMENT ON COLUMN skills.id IS 'Skill 主键';
COMMENT ON COLUMN skills.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN skills.user_id IS '创建者用户 ID';
COMMENT ON COLUMN skills.name IS 'Skill 名称';
COMMENT ON COLUMN skills.description IS 'Skill 描述';
COMMENT ON COLUMN skills.system_prompt IS 'Skill 专属系统提示词片段';
COMMENT ON COLUMN skills.icon IS 'Skill 图标';
COMMENT ON COLUMN skills.category IS 'Skill 分类';
COMMENT ON COLUMN skills.tags IS 'Skill 标签列表';
COMMENT ON COLUMN skills.rag_config IS 'Skill RAG 配置';
COMMENT ON COLUMN skills.memory_config IS 'Skill 记忆配置';
COMMENT ON COLUMN skills.enabled IS 'Skill 是否启用';
COMMENT ON COLUMN skills.created_at IS 'Skill 创建时间';
COMMENT ON COLUMN skills.updated_at IS 'Skill 更新时间';

COMMENT ON TABLE skill_tools IS 'Skill 与 Tool 关联表';
COMMENT ON COLUMN skill_tools.id IS '关联主键';
COMMENT ON COLUMN skill_tools.skill_id IS 'Skill ID';
COMMENT ON COLUMN skill_tools.tool_id IS 'Tool ID';

COMMENT ON TABLE skill_knowledge_bases IS 'Skill 与 KnowledgeBase 关联表';
COMMENT ON COLUMN skill_knowledge_bases.id IS '关联主键';
COMMENT ON COLUMN skill_knowledge_bases.skill_id IS 'Skill ID';
COMMENT ON COLUMN skill_knowledge_bases.knowledge_base_id IS 'KnowledgeBase ID';

COMMENT ON TABLE agent_skills IS 'Agent 与 Skill 挂载关联表';
COMMENT ON COLUMN agent_skills.id IS '关联主键';
COMMENT ON COLUMN agent_skills.agent_id IS 'Agent ID';
COMMENT ON COLUMN agent_skills.skill_id IS 'Skill ID';
COMMENT ON COLUMN agent_skills.priority IS 'Skill 优先级，数值越大越靠前';
COMMENT ON COLUMN agent_skills.enabled IS '是否对该 Agent 启用此 Skill';

COMMENT ON COLUMN agent_settings.skill_policy IS 'Skill 调用策略配置';

COMMENT ON COLUMN knowledge_bases.id IS '知识库主键';
COMMENT ON COLUMN knowledge_bases.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN knowledge_bases.name IS '知识库名称';
COMMENT ON COLUMN knowledge_bases.description IS '知识库描述';
COMMENT ON COLUMN knowledge_bases.created_by IS '创建人用户 ID';
COMMENT ON COLUMN knowledge_bases.created_at IS '知识库创建时间';

COMMENT ON COLUMN agent_knowledge_bases.id IS '智能体知识库关联主键';
COMMENT ON COLUMN agent_knowledge_bases.agent_id IS '智能体 ID';
COMMENT ON COLUMN agent_knowledge_bases.knowledge_base_id IS '知识库 ID';

COMMENT ON COLUMN knowledge_documents.id IS '知识文档主键';
COMMENT ON COLUMN knowledge_documents.knowledge_base_id IS '所属知识库 ID';
COMMENT ON COLUMN knowledge_documents.filename IS '原始文件名';
COMMENT ON COLUMN knowledge_documents.content_type IS '文档内容类型';
COMMENT ON COLUMN knowledge_documents.source_type IS '文档来源类型';
COMMENT ON COLUMN knowledge_documents.text IS '文档全文文本';
COMMENT ON COLUMN knowledge_documents.text_preview IS '文档内容预览';
COMMENT ON COLUMN knowledge_documents.status IS '文档处理状态';
COMMENT ON COLUMN knowledge_documents.chunk_count IS '切分后的分块数量';
COMMENT ON COLUMN knowledge_documents.error_message IS '处理失败错误信息';
COMMENT ON COLUMN knowledge_documents.created_at IS '知识文档创建时间';
COMMENT ON COLUMN knowledge_documents.updated_at IS '知识文档更新时间';

COMMENT ON COLUMN knowledge_chunks.id IS '知识分块主键';
COMMENT ON COLUMN knowledge_chunks.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN knowledge_chunks.knowledge_base_id IS '所属知识库 ID';
COMMENT ON COLUMN knowledge_chunks.document_id IS '所属文档 ID';
COMMENT ON COLUMN knowledge_chunks.chunk_index IS '文档内分块序号';
COMMENT ON COLUMN knowledge_chunks.text IS '分块文本内容';
COMMENT ON COLUMN knowledge_chunks.vector_id IS '向量存储记录 ID';
COMMENT ON COLUMN knowledge_chunks.parent_id IS '父级分块标识';
COMMENT ON COLUMN knowledge_chunks.chunk_id IS '分块业务标识';
COMMENT ON COLUMN knowledge_chunks.title IS '分块标题';
COMMENT ON COLUMN knowledge_chunks.page IS '所在页码';
COMMENT ON COLUMN knowledge_chunks.section IS '所属章节';
COMMENT ON COLUMN knowledge_chunks.content_hash IS '分块内容哈希';
COMMENT ON COLUMN knowledge_chunks.embedding_model IS '向量化模型名称';
COMMENT ON COLUMN knowledge_chunks.embedding_dimension IS '向量维度';
COMMENT ON COLUMN knowledge_chunks.metadata IS '分块扩展元数据';

COMMENT ON COLUMN workflow_definitions.id IS '工作流定义主键';
COMMENT ON COLUMN workflow_definitions.agent_id IS '所属智能体 ID';
COMMENT ON COLUMN workflow_definitions.nodes IS '工作流节点定义';
COMMENT ON COLUMN workflow_definitions.updated_at IS '工作流定义更新时间';

COMMENT ON COLUMN uploads.id IS '上传对象 ID';
COMMENT ON COLUMN uploads.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN uploads.user_id IS '上传用户 ID';
COMMENT ON COLUMN uploads.filename IS '原始文件名';
COMMENT ON COLUMN uploads.content_type IS '文件内容类型';
COMMENT ON COLUMN uploads.kind IS '上传文件类型';
COMMENT ON COLUMN uploads.data_url IS '文件 Data URL 内容';
COMMENT ON COLUMN uploads.text IS '解析后的文本内容';
COMMENT ON COLUMN uploads.size IS '文件大小（字节）';
COMMENT ON COLUMN uploads.created_at IS '上传时间';

COMMENT ON COLUMN sessions.id IS '会话主键';
COMMENT ON COLUMN sessions.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN sessions.agent_id IS '所属智能体 ID';
COMMENT ON COLUMN sessions.user_id IS '发起会话的用户 ID';
COMMENT ON COLUMN sessions.title IS '会话标题';
COMMENT ON COLUMN sessions.is_debug IS '是否为调试会话';
COMMENT ON COLUMN sessions.created_at IS '会话创建时间';
COMMENT ON COLUMN sessions.updated_at IS '会话更新时间';

COMMENT ON COLUMN messages.id IS '消息主键';
COMMENT ON COLUMN messages.session_id IS '所属会话 ID';
COMMENT ON COLUMN messages.role IS '消息角色';
COMMENT ON COLUMN messages.content IS '消息内容';
COMMENT ON COLUMN messages.reasoning IS '模型推理文本';
COMMENT ON COLUMN messages.reasoning_duration_ms IS '推理耗时（毫秒）';
COMMENT ON COLUMN messages.sources IS '引用来源列表';
COMMENT ON COLUMN messages.created_at IS '消息创建时间';

COMMENT ON COLUMN runs.id IS '运行记录主键';
COMMENT ON COLUMN runs.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN runs.agent_id IS '所属智能体 ID';
COMMENT ON COLUMN runs.session_id IS '关联会话 ID';
COMMENT ON COLUMN runs.status IS '运行状态';
COMMENT ON COLUMN runs.started_at IS '开始时间';
COMMENT ON COLUMN runs.completed_at IS '完成时间';

COMMENT ON COLUMN run_steps.id IS '运行步骤主键';
COMMENT ON COLUMN run_steps.run_id IS '所属运行记录 ID';
COMMENT ON COLUMN run_steps.node_id IS '工作流节点 ID';
COMMENT ON COLUMN run_steps.node_type IS '工作流节点类型';
COMMENT ON COLUMN run_steps.status IS '步骤状态';
COMMENT ON COLUMN run_steps.input IS '步骤输入快照';
COMMENT ON COLUMN run_steps.output IS '步骤输出快照';
COMMENT ON COLUMN run_steps.created_at IS '步骤创建时间';

COMMENT ON COLUMN session_memory.id IS '会话记忆主键';
COMMENT ON COLUMN session_memory.session_id IS '所属会话 ID';
COMMENT ON COLUMN session_memory.summary IS '会话摘要';
COMMENT ON COLUMN session_memory.message_count IS '累计消息数';
COMMENT ON COLUMN session_memory.updated_at IS '会话记忆更新时间';

COMMENT ON COLUMN agent_memory_profiles.id IS '智能体记忆画像主键';
COMMENT ON COLUMN agent_memory_profiles.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN agent_memory_profiles.user_id IS '所属用户 ID';
COMMENT ON COLUMN agent_memory_profiles.agent_id IS '所属智能体 ID';
COMMENT ON COLUMN agent_memory_profiles.enabled IS '是否启用长期记忆';
COMMENT ON COLUMN agent_memory_profiles.summary IS '用户记忆摘要';
COMMENT ON COLUMN agent_memory_profiles.facts IS '结构化事实列表';
COMMENT ON COLUMN agent_memory_profiles.preferences IS '用户偏好信息';
COMMENT ON COLUMN agent_memory_profiles.created_at IS '记忆画像创建时间';
COMMENT ON COLUMN agent_memory_profiles.updated_at IS '记忆画像更新时间';

COMMENT ON COLUMN prompt_templates.id IS '提示词模板主键';
COMMENT ON COLUMN prompt_templates.workspace_id IS '所属工作区 ID';
COMMENT ON COLUMN prompt_templates.user_id IS '所属用户 ID';
COMMENT ON COLUMN prompt_templates.title IS '模板标题';
COMMENT ON COLUMN prompt_templates.description IS '模板描述';
COMMENT ON COLUMN prompt_templates.content IS '模板内容';
COMMENT ON COLUMN prompt_templates.category IS '模板分类';
COMMENT ON COLUMN prompt_templates.tags IS '模板标签列表';
COMMENT ON COLUMN prompt_templates.enabled IS '模板是否启用';
COMMENT ON COLUMN prompt_templates.created_at IS '模板创建时间';
COMMENT ON COLUMN prompt_templates.updated_at IS '模板更新时间';

COMMENT ON COLUMN feedback.id IS '反馈主键';
COMMENT ON COLUMN feedback.message_id IS '关联消息 ID';
COMMENT ON COLUMN feedback.user_id IS '反馈用户 ID';
COMMENT ON COLUMN feedback.rating IS '反馈评级';
COMMENT ON COLUMN feedback.comment IS '反馈备注';
COMMENT ON COLUMN feedback.created_at IS '反馈创建时间';

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_user_model_configs_updated_at ON user_model_configs;
CREATE TRIGGER trg_user_model_configs_updated_at
BEFORE UPDATE ON user_model_configs
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_agents_updated_at ON agents;
CREATE TRIGGER trg_agents_updated_at
BEFORE UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_agent_settings_updated_at ON agent_settings;
CREATE TRIGGER trg_agent_settings_updated_at
BEFORE UPDATE ON agent_settings
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_prompt_templates_updated_at ON prompt_templates;
CREATE TRIGGER trg_prompt_templates_updated_at
BEFORE UPDATE ON prompt_templates
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_tools_updated_at ON tools;
CREATE TRIGGER trg_tools_updated_at
BEFORE UPDATE ON tools
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_skills_updated_at ON skills;
CREATE TRIGGER trg_skills_updated_at
BEFORE UPDATE ON skills
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_knowledge_documents_updated_at ON knowledge_documents;
CREATE TRIGGER trg_knowledge_documents_updated_at
BEFORE UPDATE ON knowledge_documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_workflow_definitions_updated_at ON workflow_definitions;
CREATE TRIGGER trg_workflow_definitions_updated_at
BEFORE UPDATE ON workflow_definitions
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_sessions_updated_at ON sessions;
CREATE TRIGGER trg_sessions_updated_at
BEFORE UPDATE ON sessions
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_session_memory_updated_at ON session_memory;
CREATE TRIGGER trg_session_memory_updated_at
BEFORE UPDATE ON session_memory
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_agent_memory_profiles_updated_at ON agent_memory_profiles;
CREATE TRIGGER trg_agent_memory_profiles_updated_at
BEFORE UPDATE ON agent_memory_profiles
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO model_configs (
    provider,
    model_name,
    display_name,
    supports_text,
    supports_image,
    supports_document,
    supports_reasoning,
    reasoning_type,
    reasoning_label,
    max_context,
    default_temperature,
    enabled
)
VALUES
    ('openai-compatible', 'qwen-plus', 'Qwen Plus', TRUE, FALSE, TRUE, TRUE, 'prompt', '提示词增强', 131072, 0.4, TRUE),
    ('openai-compatible', 'qwen-vl-plus', 'Qwen VL Plus', TRUE, TRUE, TRUE, TRUE, 'prompt', '提示词增强', 32768, 0.4, TRUE)
ON CONFLICT (model_name) DO UPDATE SET
    provider = EXCLUDED.provider,
    display_name = EXCLUDED.display_name,
    supports_text = EXCLUDED.supports_text,
    supports_image = EXCLUDED.supports_image,
    supports_document = EXCLUDED.supports_document,
    supports_reasoning = EXCLUDED.supports_reasoning,
    reasoning_type = EXCLUDED.reasoning_type,
    reasoning_label = EXCLUDED.reasoning_label,
    max_context = EXCLUDED.max_context,
    default_temperature = EXCLUDED.default_temperature,
    enabled = EXCLUDED.enabled;

INSERT INTO tools (
    type,
    name,
    label,
    description,
    schema,
    headers_schema,
    query_schema,
    body_schema,
    search_options,
    enabled
)
VALUES
    ('builtin', 'weather', 'Weather tool', 'Demo weather advice by city.', '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE),
    ('builtin', 'report', 'Report tool', 'Demo device report by user id.', '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE),
    ('builtin_search', 'web_search', 'Web search', 'Search public web pages and return short snippets.', '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{"max_results": 5}'::jsonb, TRUE)
ON CONFLICT (name) WHERE workspace_id IS NULL AND user_id IS NULL DO UPDATE SET
    label = EXCLUDED.label,
    description = EXCLUDED.description,
    enabled = EXCLUDED.enabled;

-- Secret rules:
-- - user_model_configs.encrypted_api_key stores encrypted model keys only.
-- - tools.encrypted_secret stores encrypted tool secrets only.
-- - agent_versions.snapshot must never contain raw or encrypted model keys or tool secrets.

COMMIT;

