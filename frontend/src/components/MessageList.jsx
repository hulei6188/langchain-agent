import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Brain, ThumbsUp, ThumbsDown, FileText, Search, ImagePlus } from 'lucide-react';
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
                {(message.reasoning || message.reasoningPending) && (
                  <MessageReasoning
                    content={message.reasoning || ''}
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
  const numeric = Number(durationMs);
  if (!Number.isFinite(numeric) || numeric < 0) return '';
  const totalSeconds = Math.max(0, Math.floor(numeric / 1000));
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
  const totalSeconds = Math.max(0, Math.floor((end - start) / 1000));
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`;
}

function MessageReasoning({ content, pending, startedAt, finishedAt, durationMs }) {
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

  return (
    <details className={pending ? 'message-reasoning is-pending' : 'message-reasoning is-done'} open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <Brain size={14} />
        <strong>{statusText}{durationText}</strong>
        {content ? <small className="message-reasoning-count">{content.length} 字</small> : null}
      </summary>
      <div className="message-reasoning-content">
        {content ? <MarkdownContent content={content} /> : <p>等待模型返回推理过程...</p>}
      </div>
    </details>
  );
}

export function MarkdownContent({ content }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const code = String(children || '').replace(/\n$/, '');
            if (inline) {
              return <code className={className} {...props}>{children}</code>;
            }
            return <CodeBlock language={match?.[1] || 'text'} code={code} />;
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
  async function copyCode() {
    await navigator.clipboard?.writeText(code);
  }
  return (
    <div className="code-block">
      <div className="code-header">
        <span>{language || 'text'}</span>
        <button type="button" onClick={copyCode}>复制</button>
      </div>
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
