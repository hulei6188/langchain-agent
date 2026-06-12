import React, { useMemo, useState } from 'react';
import { X, Plus, Save, Search, Wand2, Database } from 'lucide-react';

function toolTypeLabel(type) {
  return { builtin: '内置', builtin_search: '搜索', http: 'HTTP', mcp: 'MCP' }[type] || type || '未知';
}

const ACTIVATION_MODE_OPTIONS = [
  { value: 'auto', label: '自动选择' },
  { value: 'always', label: '每轮加载' },
  { value: 'manual', label: '手动触发' },
  { value: 'disabled', label: '不参与运行' },
];

export function SkillDialog({
  form,
  onCancel,
  onChange,
  onSubmit,
  saving,
  tools = [],
  knowledgeBases = [],
  title = "新建技能",
  description = "创建一个可复用的能力包：包含 Prompt 片段、专属工具和知识库。",
  submitText = "创建技能",
  savingText = "创建中...",
  isEdit = false
}) {
  const [resourceTab, setResourceTab] = useState('tools');
  const [toolSearch, setToolSearch] = useState('');
  const [kbSearch, setKbSearch] = useState('');

  const filteredTools = useMemo(() => {
    const q = toolSearch.trim().toLowerCase();
    if (!q) return tools;
    return tools.filter((t) =>
      `${t.label} ${t.name} ${t.description}`.toLowerCase().includes(q)
    );
  }, [tools, toolSearch]);

  const filteredKbs = useMemo(() => {
    const q = kbSearch.trim().toLowerCase();
    if (!q) return knowledgeBases;
    return knowledgeBases.filter((kb) =>
      `${kb.name} ${kb.description || ''}`.toLowerCase().includes(q)
    );
  }, [knowledgeBases, kbSearch]);

  function toggleToolId(id) {
    const exists = form.tool_ids.includes(id);
    onChange({
      ...form,
      tool_ids: exists ? form.tool_ids.filter((item) => item !== id) : [...form.tool_ids, id],
    });
  }

  function toggleKbId(id) {
    const exists = form.knowledge_base_ids.includes(id);
    onChange({
      ...form,
      knowledge_base_ids: exists
        ? form.knowledge_base_ids.filter((item) => item !== id)
        : [...form.knowledge_base_ids, id],
    });
  }

  const selectedToolCount = form.tool_ids.length;
  const selectedKbCount = form.knowledge_base_ids.length;

  return (
    <div className="profile-dialog-backdrop" onClick={onCancel}>
      <div
        className="skill-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        {/* ── Header ── */}
        <div className="skill-modal-header">
          <div className="skill-modal-header-main">
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
          <button
            className="skill-modal-close"
            type="button"
            title="关闭"
            aria-label="关闭技能表单"
            onClick={onCancel}
            disabled={saving}
          >
            <X size={18} />
          </button>
        </div>

        {/* ── Body ── */}
        <div className="skill-modal-body">
          <form id="skill-form" className="skill-form-layout" onSubmit={onSubmit}>
            {/* Left: Basic Info */}
            <div className="skill-main-form">
              <label className="field-stack">
                <span>名称</span>
                <input
                  value={form.name}
                  maxLength={50}
                  onChange={(event) => onChange({ ...form, name: event.target.value })}
                  placeholder="例如：代码审查专家"
                  autoFocus
                />
                <em>{String(form.name || '').length}/50</em>
              </label>

              <label className="field-stack">
                <span>描述</span>
                <textarea
                  value={form.description}
                  maxLength={500}
                  onChange={(event) => onChange({ ...form, description: event.target.value })}
                  placeholder="简要说明此技能的用途和适用场景"
                  rows={3}
                />
                <em>{String(form.description || '').length}/500</em>
              </label>

              <div className="skill-form-inline">
                <label className="field-stack">
                  <span>分类</span>
                  <input
                    value={form.category}
                    onChange={(event) => onChange({ ...form, category: event.target.value })}
                    placeholder="例如：开发、写作、分析"
                  />
                </label>
                <label className="field-stack">
                  <span>标签（逗号分隔）</span>
                  <input
                    value={form.tagsText}
                    onChange={(event) => onChange({ ...form, tagsText: event.target.value })}
                    placeholder="例如：Python, 代码审查"
                  />
                </label>
              </div>

              <label className="field-stack">
                <span>激活策略</span>
                <select
                  value={form.activation_mode || 'auto'}
                  onChange={(event) => onChange({ ...form, activation_mode: event.target.value })}
                >
                  {ACTIVATION_MODE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </label>

              <label className="field-stack">
                <span>技能 Prompt（技能被加载后拼接到 Agent System Prompt 末尾）</span>
                <textarea
                  className="skill-prompt-textarea"
                  value={form.system_prompt}
                  onChange={(event) => onChange({ ...form, system_prompt: event.target.value })}
                  placeholder="写出该技能专属的提示词片段，例如：你擅长审查 Python 代码，关注 PEP8、类型安全和异常处理。"
                />
              </label>
            </div>

            {/* Right: Resource Binding */}
            <div className="skill-resource-panel">
              {/* Tabs */}
              <div className="skill-resource-tabs">
                <button
                  type="button"
                  className={`skill-resource-tab ${resourceTab === 'tools' ? 'active' : ''}`}
                  onClick={() => setResourceTab('tools')}
                >
                  <Wand2 size={14} />
                  <span>绑定工具</span>
                  {selectedToolCount > 0 && <em>{selectedToolCount}</em>}
                </button>
                <button
                  type="button"
                  className={`skill-resource-tab ${resourceTab === 'knowledge' ? 'active' : ''}`}
                  onClick={() => setResourceTab('knowledge')}
                >
                  <Database size={14} />
                  <span>绑定知识库</span>
                  {selectedKbCount > 0 && <em>{selectedKbCount}</em>}
                </button>
              </div>

              {/* Tab Content */}
              <div className="skill-resource-tab-content">
                {resourceTab === 'tools' ? (
                  <>
                    <div className="resource-search">
                      <Search size={14} />
                      <input
                        type="text"
                        placeholder="搜索工具..."
                        value={toolSearch}
                        onChange={(e) => setToolSearch(e.target.value)}
                      />
                    </div>
                    <div className="resource-list">
                      {filteredTools.map((tool) => {
                        const isAdded = form.tool_ids.includes(tool.id);
                        return (
                          <div
                            key={tool.id}
                            className={`resource-item ${isAdded ? 'selected' : ''}`}
                          >
                            <div className="resource-item-main" onClick={() => toggleToolId(tool.id)}>
                              <div className="resource-item-name">
                                <span className="resource-item-icon">
                                  {tool.type === 'builtin_search' ? '🔍' : tool.type === 'mcp' ? '🧩' : '🛠️'}
                                </span>
                                {tool.label}
                              </div>
                              <div className="resource-item-desc">
                                {tool.description || toolTypeLabel(tool.type)}
                              </div>
                            </div>
                            <button
                              type="button"
                              className={`resource-item-action ${isAdded ? 'on' : ''}`}
                              onClick={() => toggleToolId(tool.id)}
                            >
                              {isAdded ? '已选' : '选择'}
                            </button>
                          </div>
                        );
                      })}
                      {filteredTools.length === 0 && (
                        <p className="resource-list-empty">无匹配工具</p>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="resource-search">
                      <Search size={14} />
                      <input
                        type="text"
                        placeholder="搜索知识库..."
                        value={kbSearch}
                        onChange={(e) => setKbSearch(e.target.value)}
                      />
                    </div>
                    <div className="resource-list">
                      {filteredKbs.map((kb) => {
                        const isAdded = form.knowledge_base_ids.includes(kb.id);
                        return (
                          <div
                            key={kb.id}
                            className={`resource-item ${isAdded ? 'selected' : ''}`}
                          >
                            <div className="resource-item-main" onClick={() => toggleKbId(kb.id)}>
                              <div className="resource-item-name">
                                <span className="resource-item-icon">📄</span>
                                {kb.name}
                              </div>
                              <div className="resource-item-desc">
                                {kb.description || '无描述'}
                              </div>
                            </div>
                            <button
                              type="button"
                              className={`resource-item-action ${isAdded ? 'on' : ''}`}
                              onClick={() => toggleKbId(kb.id)}
                            >
                              {isAdded ? '已选' : '选择'}
                            </button>
                          </div>
                        );
                      })}
                      {filteredKbs.length === 0 && (
                        <p className="resource-list-empty">无匹配知识库</p>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </form>
        </div>

        {/* ── Footer ── */}
        <div className="skill-modal-footer">
          <button type="button" className="skill-footer-cancel" onClick={onCancel} disabled={saving}>
            取消
          </button>
          <button
            className="skill-footer-submit"
            type="submit"
            form="skill-form"
            disabled={saving || !form.name.trim()}
          >
            {isEdit ? <Save size={15} /> : <Plus size={15} />}
            {saving ? savingText : submitText}
          </button>
        </div>
      </div>
    </div>
  );
}
