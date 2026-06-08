import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { AlertCircle, Brain, CheckCircle2, FileText, Loader2, Search, ThumbsDown, ThumbsUp, Wrench } from 'lucide-react';
import { AgentAvatar } from './AgentAvatar.jsx';

export function MessageList({ messages, feedbackByMessage = {}, submitFeedback = () => {}, avatar = 'AI' }) {
  return (
    <>
      {messages.map((message, index) => (
        <div key={`${message.role}-${index}-${message.id || ''}`} className={`message ${message.role}`}>
          {message.role === 'user' ? (
            <span>我</span>
          ) : (
            <AgentAvatar value={avatar} />
          )}
          <div className="message-body">
            {message.role === 'assistant' ? (
              <div className={message.error ? 'message-error' : ''}>
                {(message.reasoning || message.reasoningPending || message.reasoningTimeline?.length || message.toolCalls?.length) && (
                  <MessageReasoning
                    content={message.reasoning || ''}
                    timeline={message.reasoningTimeline || []}
                    toolCalls={message.toolCalls || []}
                    pending={message.reasoningPending}
                    startedAt={message.reasoningStartedAt}
                    finishedAt={message.reasoningFinishedAt}
                    durationMs={message.reasoningDurationMs}
                  />
                )}
                {message.pending && !message.content ? (
                  <p className="message-pending">{message.reasoning ? '正在组织回答...' : '思考中...'}</p>
                ) : (
                  <MarkdownContent content={message.content || ''} />
                )}
              </div>
            ) : <>
              <p>{message.content}</p>
              {message.attachments?.length > 0 && (
                <div className="message-attachments">
                  {message.attachments.map((att) => (
                    <div key={att.id} className={att.kind === 'image' || att.type === 'image' ? 'message-attachment-image' : 'message-attachment-document'}>
                      {att.kind === 'image' || att.type === 'image' ? (
                        <img src={att.data_url || att.preview_url} alt={att.filename} />
                      ) : (
                        <span className="message-attachment-file"><FileText size={14} />{att.filename}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>}
            {message.role === 'assistant' && message.id && (
              <div className="feedback-actions">
                <button
                  type="button"
                  className={feedbackByMessage[message.id] === 'positive' ? 'selected' : ''}
                  title="回答有帮助"
                  onClick={() => submitFeedback(message.id, 'positive').catch((err) => console.error(err))}
                >
                  <ThumbsUp size={14} />
                </button>
                <button
                  type="button"
                  className={feedbackByMessage[message.id] === 'negative' ? 'selected' : ''}
                  title="回答不理想"
                  onClick={() => submitFeedback(message.id, 'negative').catch((err) => console.error(err))}
                >
                  <ThumbsDown size={14} />
                </button>
              </div>
            )}
            {message.role === 'assistant' && (message.sources || []).length > 0 && (
              <MessageSources sources={message.sources} />
            )}
          </div>
        </div>
      ))}
    </>
  );
}

function toTimestamp(value) {
  if (!value) return null;
  if (typeof value === 'number') return value;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDurationMs(durationMs) {
  if (durationMs === null || durationMs === undefined || durationMs === '') return '';
  const numeric = Number(durationMs);
  if (!Number.isFinite(numeric) || numeric < 0) return '';
  const totalSeconds = Math.max(1, Math.ceil(numeric / 1000));
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`;
}

function formatThinkingDuration(startedAt, finishedAt, now, durationMs) {
  const storedDuration = formatDurationMs(durationMs);
  if (storedDuration) return storedDuration;
  const start = toTimestamp(startedAt);
  if (!start) return '';
  const end = toTimestamp(finishedAt) || now;
  const totalSeconds = Math.max(1, Math.ceil(Math.max(0, end - start) / 1000));
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`;
}

function MessageReasoning({ content, timeline = [], toolCalls = [], pending, startedAt, finishedAt, durationMs }) {
  const [open, setOpen] = useState(true);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (pending) setOpen(true);
  }, [pending]);

  useEffect(() => {
    if (!pending || !startedAt) return undefined;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [pending, startedAt]);

  const duration = formatThinkingDuration(startedAt, finishedAt, now, pending ? null : durationMs);
  const statusText = pending ? '思考中' : '已思考';
  const durationText = duration ? (pending ? `（${duration}）` : `（用时 ${duration}）`) : '';
  const timelineItems = reasoningTimelineItems(content, timeline, toolCalls, pending);
  const contentLength = timelineItems
    .filter((item) => item.type === 'reasoning')
    .reduce((total, item) => total + String(item.content || '').length, 0);

  return (
    <details className={pending ? 'message-reasoning is-pending' : 'message-reasoning is-done'} open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <Brain size={14} />
        <strong>{statusText}{durationText}</strong>
        {contentLength ? <small className="message-reasoning-count">{contentLength} 字</small> : null}
      </summary>
      <div className="message-reasoning-content">
        <ol className="reasoning-timeline">
          {timelineItems.map((item, index) => (
            <ReasoningTimelineItem item={item} key={item.id || `${item.type}-${index}`} />
          ))}
        </ol>
      </div>
    </details>
  );
}

function ReasoningTimelineItem({ item }) {
  if (item.type === 'tool' || item.type === 'search') {
    const rawInput = item.rawInput || item.inputRaw || '';
    const rawResult = item.rawResult || item.resultRaw || '';
    const icon = item.type === 'search'
      ? <Search size={15} />
      : item.status === 'error'
        ? <AlertCircle size={15} />
        : item.status === 'running'
          ? <Loader2 size={15} />
          : <Wrench size={15} />;
    return (
      <li className={`reasoning-timeline-item is-${item.type} status-${item.status || 'success'}`}>
        <span className="reasoning-timeline-node" aria-hidden="true">{icon}</span>
        <div className="reasoning-timeline-main">
          <div className="reasoning-tool-title">
            <strong>{item.title || '调用工具'}</strong>
            {item.status === 'success' && <CheckCircle2 size={13} />}
          </div>
          <div className="reasoning-tool-meta">
            {item.meta && <span>{item.meta}</span>}
            {item.latency && <span>{item.latency}</span>}
          </div>
          {item.inputPreview && <p className="reasoning-tool-input"><span>{item.inputLabel || '参数'}：</span>{item.inputPreview}</p>}
          {item.summary && <p className="reasoning-tool-summary"><span>结果摘要：</span>{item.summary}</p>}
          {(rawInput || rawResult) && (
            <div className="reasoning-tool-raw-list">
              <ToolRawDetails label="查看原始参数" value={rawInput} />
              <ToolRawDetails label="查看原始结果" value={rawResult} />
            </div>
          )}
        </div>
      </li>
    );
  }

  return (
    <li className="reasoning-timeline-item is-reasoning">
      <span className="reasoning-timeline-node" aria-hidden="true" />
      <div className="reasoning-timeline-main">
        <MarkdownContent content={item.content || ''} />
      </div>
    </li>
  );
}

function ToolRawDetails({ label, value }) {
  if (!value) return null;
  return (
    <details className="reasoning-tool-raw">
      <summary>{label}</summary>
      <pre><code>{formatRawToolValue(value)}</code></pre>
    </details>
  );
}

function formatRawToolValue(value) {
  if (typeof value !== 'string') return JSON.stringify(value || '', null, 2);
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

function reasoningTimelineItems(content, timeline, toolCalls, pending) {
  const source = Array.isArray(timeline) && timeline.length > 0 ? timeline : [{ type: 'reasoning', content }];
  const items = [];
  source.forEach((item, index) => {
    if (!item) return;
    if (item.type === 'reasoning') {
      const block = normalizeReasoningBlock(item.content);
      if (block) {
        items.push({ type: 'reasoning', content: block, id: item.id || `reasoning-${index}` });
      }
      return;
    }
    items.push({ ...item, id: item.id || `${item.type}-${index}` });
  });

  if (!Array.isArray(timeline) || timeline.length === 0) {
    (toolCalls || []).forEach((call, index) => {
      const name = call?.function?.name || call?.tool_name || call?.name || `tool_${index + 1}`;
      items.push({
        id: `stored-tool-${index}`,
        type: 'tool',
        status: 'success',
        title: `调用 ${name}`,
        meta: call?.type || 'tool',
        summary: compactText(call?.function?.arguments || call?.arguments || ''),
      });
    });
  }

  if (!items.length) {
    items.push({
      id: 'reasoning-waiting',
      type: 'reasoning',
      content: pending ? '等待模型返回推理过程...' : '暂无推理过程。',
    });
  }
  return items;
}

export function splitReasoningSteps(text) {
  const normalized = normalizeReasoningBlock(text);
  return normalized ? [normalized] : [];
}

function normalizeReasoningBlock(value) {
  return String(value || '').replace(/\r\n/g, '\n').trim();
}

function compactText(value, limit = 160) {
  const text = typeof value === 'string' ? value : JSON.stringify(value || '', null, 0);
  const compact = String(text || '').replace(/\s+/g, ' ').trim();
  return compact.length > limit ? `${compact.slice(0, limit)}...` : compact;
}

export function MarkdownContent({ content }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          table({ children, node, ...props }) {
            return (
              <div className="markdown-table-wrapper">
                <table {...props}>{children}</table>
              </div>
            );
          },
          pre({ children, ...props }) {
            const child = React.Children.toArray(children).find(React.isValidElement);
            if (!child) {
              return <pre {...props}>{children}</pre>;
            }
            const className = child.props?.className || '';
            const match = /language-([\w-]+)/.exec(className);
            const code = React.Children.toArray(child.props?.children || '').join('').replace(/\n$/, '');
            return <CodeBlock language={match?.[1] || 'text'} code={code} />;
          },
          code({ className, children, ...props }) {
            const inlineClassName = className ? `${className} inline-code` : 'inline-code';
            return <code className={inlineClassName} {...props}>{children}</code>;
          },
          a({ children, href, ...props }) {
            return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>;
          },
        }}
      >
        {content || ''}
      </ReactMarkdown>
    </div>
  );
}

export function CodeBlock({ language, code }) {
  const normalizedLanguage = String(language || '').trim().toLowerCase();
  const isPlainText = !normalizedLanguage || ['text', 'txt', 'plain', 'plaintext'].includes(normalizedLanguage);
  async function copyCode() {
    await navigator.clipboard?.writeText(code);
  }
  return (
    <div className={isPlainText ? 'code-block code-block-plain' : 'code-block'}>
      {!isPlainText && (
        <div className="code-header">
          <span>{language}</span>
          <button type="button" onClick={copyCode}>复制</button>
        </div>
      )}
      <pre><code>{code}</code></pre>
    </div>
  );
}

export function MessageSources({ sources }) {
  // 根据文档ID或标题进行去重，避免对同一文档的多个分片重复渲染完全相同的卡片
  const uniqueSources = [];
  const seenDocs = new Set();
  
  for (const src of sources) {
    const docKey = src.document_id || src.title || src.source_id;
    if (docKey) {
      if (!seenDocs.has(docKey)) {
        seenDocs.add(docKey);
        uniqueSources.push(src);
      }
    } else {
      uniqueSources.push(src);
    }
  }

  const visible = uniqueSources.slice(0, 4);
  const hiddenCount = Math.max(0, uniqueSources.length - visible.length);
  
  return (
    <details className="message-sources">
      <summary>引用来源 <span>{uniqueSources.length}</span></summary>
      <div className="message-source-list">
        {visible.map((source) => <SourceChip key={source.chunk_id || `${source.title}-${source.snippet}`} source={source} />)}
        {hiddenCount > 0 && <span className="source-more">还有 {hiddenCount} 个</span>}
      </div>
    </details>
  );
}

export function SourceChip({ source }) {
  const meta = [
    source.page ? `p.${source.page}` : '',
    source.section || '',
    source.retrieval_channel || '',
    Number.isFinite(Number(source.score)) ? Number(source.score).toFixed(2) : '',
  ].filter(Boolean).join(' · ');
  const content = (
    <>
      {source.url ? <Search size={14} /> : <FileText size={14} />}
      <strong>{source.title || source.source_id || 'source'}</strong>
      {meta && <small>{meta}</small>}
    </>
  );
  return source.url
    ? <a className="source-link" title={source.snippet || source.title} href={source.url} target="_blank" rel="noreferrer">{content}</a>
    : <span title={source.snippet || source.title}>{content}</span>;
}
