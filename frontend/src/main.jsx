import React, { Component, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ResegmentModal } from './components/ResegmentModal.jsx';
import { ChatView } from './views/ChatView.jsx';
import { BuilderView } from './views/BuilderView.jsx';
import { AgentAvatar, UserAvatar } from './components/AgentAvatar.jsx';
import { PromptTemplateDialog } from './components/PromptTemplateDialog.jsx';
import { KnowledgeBaseDialog } from './components/KnowledgeBaseDialog.jsx';
import { SkillDialog } from './components/SkillDialog.jsx';
import { KnowledgeDocumentList, KnowledgeUploadBox } from './components/KnowledgeDocumentList.jsx';

import {
  AlertTriangle,
  Bot,
  Boxes,
  Brain,
  Check,
  ChevronLeft,
  ChevronRight,
  Database,
  FileText,
  FileX2,
  ImagePlus,
  Home,
  Layers,
  KeyRound,
  LogIn,
  LogOut,
  MessageSquare,
  MoreHorizontal,
  Monitor,
  Moon,
  Plus,
  Rocket,
  Search,
  Send,
  ServerCog,
  Settings2,
  Shield,
  Sparkles,
  SquarePen,
  Sun,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  UploadCloud,
  Wand2,
  X,
} from 'lucide-react';
import './styles.css';
import {
  API_BASE,
  MAX_UPLOAD_BYTES,
  KNOWLEDGE_FILE_ACCEPT,
  KNOWLEDGE_FILE_EXTENSIONS,
  AUTH_TOKEN_KEY,
  LEGACY_AUTH_TOKEN_KEY,
  ApiError,
  isAuthError,
  notifyAuthExpired,
  initialAuthToken,
  JIGE_PROMPT,
  SAMPLE_MESSAGES,
  roleLabel,
  avatarInitial,
  isAdminRole,
  statusLabel,
  errorMessage,
  api,
  createAvatarDataUrl,
  validateAvatarFile,
  pickAgentIdentity,
  normalizeAgentIdentity,
  defaultMemoryProfile,
  normalizeMemoryProfile,
  profileToDraft,
  draftFacts,
  memoryProfilePayload,
  parsePreferences,
  safeJsonPreview,
  isPlainObject,
  isJsonCompatiblePreference,
  formatDateTime,
  defaultAgentForm,
  agentPayload,
  numericIdList,
  filterResourceItems,
  filterPromptTemplates,
  defaultPromptTemplateForm,
  defaultKnowledgeBaseForm,
  defaultSkillForm,
  skillFormPayload,
  formFromPromptTemplate,
  promptTemplateFormPayload,
  insertPromptIntoAgent,
  insertPromptAtEditor,
  joinPromptText,
  findModelForForm,
  normalizeUserModelForUi,
  modelLabel,
  modelCapabilityChips,
  reasoningCapabilityForModel,
  reasoningLabel,
  imageCapabilityFromTest,
  thinkingStatusText,
  capabilityCheckLabel,
  defaultRuntimeStatus,
  getRagRuntime,
  ragStatusText,
  runtimeStatusMessage,
  getWebSearchRuntime,
  webSearchStatusText,
  attachmentAcceptForModel,
  attachmentHintForModel,
  userModelFormPayload,
  userModelEditPayload,
  uploadTypeFromContentType,
  attachmentKind,
  modelCapabilityWarning,
  toggleKb,
  toggleSkill,
  toggleTool,
  buildToolDisplayGroups,
  initVariableValues,
  castVariables,
  fileToBase64,
  guessContentType,
  validateKnowledgeFile,
  handleKnowledgeFileInput,
  handleAttachmentInput,
  handleAttachmentPaste,
  handleAttachmentDrop,
  uploadAttachmentFiles,
  filesFromList,
  filesFromClipboard,
  hasTransferFiles,
} from './utils.js';

const THEME_STORAGE_KEY = 'agentbase_theme';
const SIDEBAR_COLLAPSED_STORAGE_KEY = 'agentbase_sidebar_collapsed';
const CHAT_AGENT_QUERY_PARAM = 'agent';
const CHAT_SESSION_QUERY_PARAM = 'session';
const APP_VIEW_QUERY_PARAM = 'view';
const APP_NAV_QUERY_PARAM = 'nav';
const THEME_MODES = ['light', 'dark', 'system'];
const APP_NAV_VALUES = new Set([
  'chat',
  'agents',
  'market',
  'my-models',
  'tools',
  'skills',
  'resources',
  'knowledge',
  'reviews',
  'members',
]);

function initialThemeMode() {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  return THEME_MODES.includes(stored) ? stored : 'light';
}

function initialSidebarHidden() {
  if (typeof window === 'undefined') return false;
  return localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === 'true';
}

function sameRouteId(left, right) {
  if (left === null || left === undefined || right === null || right === undefined) return false;
  return String(left) === String(right);
}

function normalizeAppView(value) {
  return value === 'builder' ? 'builder' : 'home';
}

function normalizeAppNav(value, view = 'home') {
  if (view === 'builder') return 'agents';
  return APP_NAV_VALUES.has(value) ? value : 'chat';
}

function isChatHomeRoute(route) {
  return route?.view === 'home' && route?.nav === 'chat';
}

function readChatRoute() {
  if (typeof window === 'undefined') {
    return { agentId: null, sessionId: null, view: 'home', nav: 'chat' };
  }
  const params = new URLSearchParams(window.location.search);
  const view = normalizeAppView(params.get(APP_VIEW_QUERY_PARAM));
  const nav = normalizeAppNav(params.get(APP_NAV_QUERY_PARAM), view);
  return {
    agentId: params.get(CHAT_AGENT_QUERY_PARAM) || null,
    sessionId: params.get(CHAT_SESSION_QUERY_PARAM) || null,
    view,
    nav,
  };
}

function writeChatRoute(agentId, sessionId = null, options = {}) {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  if (agentId) url.searchParams.set(CHAT_AGENT_QUERY_PARAM, String(agentId));
  else url.searchParams.delete(CHAT_AGENT_QUERY_PARAM);
  if (sessionId) url.searchParams.set(CHAT_SESSION_QUERY_PARAM, String(sessionId));
  else url.searchParams.delete(CHAT_SESSION_QUERY_PARAM);
  const nextUrl = `${url.pathname}${url.search}${url.hash}`;
  const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextUrl === currentUrl) return;
  const method = options.mode === 'push' ? 'pushState' : 'replaceState';
  window.history[method]({}, '', nextUrl);
}

function clearChatRoute(options = {}) {
  writeChatRoute(null, null, options);
}

function writePageRoute(view, nav, options = {}) {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  const nextView = normalizeAppView(view);
  const nextNav = normalizeAppNav(nav, nextView);
  if (nextView === 'home') url.searchParams.delete(APP_VIEW_QUERY_PARAM);
  else url.searchParams.set(APP_VIEW_QUERY_PARAM, nextView);
  if (nextView === 'home' && nextNav === 'chat') url.searchParams.delete(APP_NAV_QUERY_PARAM);
  else url.searchParams.set(APP_NAV_QUERY_PARAM, nextNav);
  if (Object.prototype.hasOwnProperty.call(options, 'agentId')) {
    if (options.agentId) url.searchParams.set(CHAT_AGENT_QUERY_PARAM, String(options.agentId));
    else url.searchParams.delete(CHAT_AGENT_QUERY_PARAM);
  }
  if (nextView === 'home' && nextNav === 'chat') {
    if (Object.prototype.hasOwnProperty.call(options, 'sessionId')) {
      if (options.sessionId) url.searchParams.set(CHAT_SESSION_QUERY_PARAM, String(options.sessionId));
      else url.searchParams.delete(CHAT_SESSION_QUERY_PARAM);
    }
  } else {
    url.searchParams.delete(CHAT_SESSION_QUERY_PARAM);
  }
  const nextUrl = `${url.pathname}${url.search}${url.hash}`;
  const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextUrl === currentUrl) return;
  const method = options.mode === 'push' ? 'pushState' : 'replaceState';
  window.history[method]({}, '', nextUrl);
}

function systemPrefersDark() {
  return Boolean(typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
}

function App() {
  const [token, setToken] = useState(initialAuthToken);
  const [appBootstrapping, setAppBootstrapping] = useState(() => Boolean(initialAuthToken()));
  const [routeRestoring, setRouteRestoring] = useState(() => {
    const route = readChatRoute();
    return Boolean(initialAuthToken() && isChatHomeRoute(route) && route.sessionId);
  });
  const initialRoute = readChatRoute();
  const [me, setMe] = useState(null);
  const [workspace, setWorkspace] = useState(null);
  const [agents, setAgents] = useState([]);
  const [activeAgentId, setActiveAgentId] = useState(null);
  const [activeAgent, setActiveAgent] = useState(null);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [tools, setTools] = useState([]);
  const [skills, setSkills] = useState([]);
  const [promptTemplates, setPromptTemplates] = useState([]);
  const [models, setModels] = useState([]);
  const [adminModels, setAdminModels] = useState([]);
  const [userModels, setUserModels] = useState([]);
  const [runtimeStatus, setRuntimeStatus] = useState(() => defaultRuntimeStatus());
  const [marketAgents, setMarketAgents] = useState([]);
  const [reviewItems, setReviewItems] = useState([]);
  const [members, setMembers] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(() => (isChatHomeRoute(initialRoute) ? initialRoute.sessionId : null));
  const [sessionTitleDraft, setSessionTitleDraft] = useState('');
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [toolDebugEvents, setToolDebugEvents] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [feedbackByMessage, setFeedbackByMessage] = useState({});
  const [chatMode, setChatMode] = useState('published');
  const [chatVariables, setChatVariables] = useState({});
  const [ragEnabled, setRagEnabled] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(true);
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [chatAttachments, setChatAttachments] = useState([]);
  const [uploadingAttachment, setUploadingAttachment] = useState(false);
  const [draft, setDraft] = useState('');
  // Per-session running state used for UI re-renders.
  const [runningBySessionId, setRunningBySessionId] = useState({});
  const [activeNewRunKey, setActiveNewRunKey] = useState('');
  // Per-session mutable run state (no re-render needed): { [sessionId]: { runId, controller, genId } }
  const sessionRunRef = useRef({});
  const activeAgentIdRef = useRef(activeAgentId);
  activeAgentIdRef.current = activeAgentId;
  // Keep a ref in sync so SSE handlers can check latest activeSessionId
  const activeSessionIdRef = useRef(activeSessionId);
  activeSessionIdRef.current = activeSessionId;
  const messagesRef = useRef(messages);
  const messageCacheRef = useRef({});
  const sourcesCacheRef = useRef({});
  // Track newly-created session ID from done event for finally cleanup
  const newSessionIdRef = useRef(null);
  // Track placeholder session ID so we can replace it with the real one in the sidebar
  const newSessionPlaceholderIdRef = useRef(null);
  // Track the temp __new__ runKey so stopGeneration can find it before the real session ID arrives
  const activeNewRunKeyRef = useRef(null);
  const pendingChatRestoreRef = useRef(readChatRoute());
  // Derived: busy only when CURRENT session is running
  const busy = activeSessionId ? !!runningBySessionId[activeSessionId]?.running : !!(activeNewRunKey && runningBySessionId[activeNewRunKey]?.running);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  function sessionCacheKey(value) {
    if (value === null || value === undefined || value === '') return '';
    return String(value);
  }

  function isSessionKeyVisible(sessionKey) {
    const key = sessionCacheKey(sessionKey);
    if (!key) return false;
    const currentSession = sessionCacheKey(activeSessionIdRef.current);
    if (currentSession) return key === currentSession;
    const currentNewRunKey = sessionCacheKey(activeNewRunKeyRef.current);
    return Boolean(currentNewRunKey && key === currentNewRunKey);
  }

  function latestSourcesFromMessages(items) {
    return [...(items || [])].reverse().find((item) => item.sources?.length)?.sources || [];
  }

  function replaceVisibleMessages(items) {
    const next = Array.isArray(items) ? items : [];
    messagesRef.current = next;
    setMessages(next);
    return next;
  }

  function writeSessionMessages(sessionKey, items, options = {}) {
    const key = sessionCacheKey(sessionKey);
    const next = Array.isArray(items) ? items : [];
    if (key) {
      messageCacheRef.current[key] = next;
    }
    if (options.visible || isSessionKeyVisible(key)) {
      replaceVisibleMessages(next);
    }
    return next;
  }

  function updateSessionMessages(sessionKey, updater, options = {}) {
    const key = sessionCacheKey(sessionKey);
    const current = key
      ? (messageCacheRef.current[key] || (isSessionKeyVisible(key) ? messagesRef.current : []))
      : messagesRef.current;
    const next = updater(current);
    return writeSessionMessages(key, next, options);
  }

  function writeSessionSources(sessionKey, items, options = {}) {
    const key = sessionCacheKey(sessionKey);
    const next = Array.isArray(items) ? items : [];
    if (key) {
      sourcesCacheRef.current[key] = next;
    }
    if (options.visible || isSessionKeyVisible(key)) {
      setSources(next);
    }
    return next;
  }

  function moveSessionCache(fromKey, toKey) {
    const from = sessionCacheKey(fromKey);
    const to = sessionCacheKey(toKey);
    if (!from || !to || from === to) return;
    if (messageCacheRef.current[from]) {
      messageCacheRef.current[to] = messageCacheRef.current[from];
      delete messageCacheRef.current[from];
    }
    if (sourcesCacheRef.current[from]) {
      sourcesCacheRef.current[to] = sourcesCacheRef.current[from];
      delete sourcesCacheRef.current[from];
    }
  }

  function setSessionRunning(sessionId) {
    if (!sessionId) return;
    setRunningBySessionId((prev) => ({ ...prev, [sessionId]: { running: true } }));
  }
  function syncListedSessionRuns(items) {
    setRunningBySessionId((prev) => {
      const next = { ...prev };
      let changed = false;
      (items || []).forEach((session) => {
        const key = sessionCacheKey(session?.id);
        if (!key) return;
        const activeRun = session.active_run;
        if (activeRun?.status === 'running') {
          if (!next[key]?.running) {
            next[key] = { running: true };
            changed = true;
          }
        } else if (next[key]?.running) {
          delete next[key];
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }
  function clearSessionRunning(sessionId) {
    if (!sessionId) return;
    setRunningBySessionId((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
  }

  const [error, setError] = useState('');
  const [toastMsg, setToastMsg] = useState('');
  const [toastTone, setToastTone] = useState('success');

  function notify(msg, tone = 'success') {
    setToastTone(tone);
    setToastMsg(msg);
    setTimeout(() => {
      setToastMsg((current) => current === msg ? '' : current);
    }, 3000);
  }
  const [authMode, setAuthMode] = useState('register');
  const [authForm, setAuthForm] = useState({ email: 'admin@example.com', name: 'Admin', password: 'password123' });
  const [agentForm, setAgentForm] = useState(defaultAgentForm());
  const [docForm, setDocForm] = useState({ filename: 'guide.txt', text: '这里是一段知识库资料。', kb_id: '' });
  const [uploadingKnowledgeFile, setUploadingKnowledgeFile] = useState(false);
  const [uploadingKnowledgeItems, setUploadingKnowledgeItems] = useState([]);
  const [view, setView] = useState(() => readChatRoute().view);
  const [activeNav, setActiveNav] = useState(() => readChatRoute().nav);
  const [homePrompt, setHomePrompt] = useState('');
  const [submittedPromptHistory, setSubmittedPromptHistory] = useState([]);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [settingsPanel, setSettingsPanel] = useState('general');
  const [profileError, setProfileError] = useState('');
  const [confirmDialog, setConfirmDialog] = useState(null);
  const [memoryProfile, setMemoryProfile] = useState(() => defaultMemoryProfile());
  const [memoryProfileDraft, setMemoryProfileDraft] = useState(() => profileToDraft(defaultMemoryProfile()));
  const [memoryProfileLoading, setMemoryProfileLoading] = useState(false);
  const [memoryProfileSaving, setMemoryProfileSaving] = useState(false);
  const [memoryProfileError, setMemoryProfileError] = useState('');
  const [agentIdentityDialog, setAgentIdentityDialog] = useState(null);
  const [agentIdentitySaving, setAgentIdentitySaving] = useState(false);
  const [agentIdentityError, setAgentIdentityError] = useState('');
  const [themeMode, setThemeMode] = useState(initialThemeMode);
  const [systemDark, setSystemDark] = useState(systemPrefersDark);
  const isDarkTheme = themeMode === 'dark' || (themeMode === 'system' && systemDark);

  const activeSummary = useMemo(() => agents.find((item) => item.id === activeAgentId), [agents, activeAgentId]);
  const chatAgents = useMemo(
    () => agents.filter((item) => item.status === 'published' && item.published_version_id),
    [agents],
  );
  const selectedDraftModel = useMemo(
    () => findModelForForm(models, userModels, agentForm),
    [models, userModels, agentForm.model_id, agentForm.user_model_config_id, agentForm.model],
  );
  const currentThinkingModel = useMemo(
    () => (view === 'builder' ? selectedDraftModel : activeAgent?.user_model_config || activeAgent?.model_config || null),
    [view, selectedDraftModel, activeAgent],
  );
  const ragRuntime = useMemo(() => getRagRuntime(runtimeStatus), [runtimeStatus]);
  const webSearchRuntime = useMemo(() => getWebSearchRuntime(runtimeStatus), [runtimeStatus]);
  const activeKbId = Number(docForm.kb_id || knowledgeBases[0]?.id || 0);
  const canManage = isAdminRole(workspace?.role);
  const canEditActive = !!activeAgent && (canManage || activeAgent.created_by === me?.id);

  useEffect(() => {
    document.body.classList.toggle('dark', isDarkTheme);
    document.documentElement.dataset.theme = isDarkTheme ? 'dark' : 'light';
    document.documentElement.dataset.themeMode = themeMode;
    localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [isDarkTheme, themeMode]);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const syncSystemTheme = (event) => setSystemDark(event.matches);
    setSystemDark(media.matches);
    if (media.addEventListener) {
      media.addEventListener('change', syncSystemTheme);
      return () => media.removeEventListener('change', syncSystemTheme);
    }
    media.addListener?.(syncSystemTheme);
    return () => media.removeListener?.(syncSystemTheme);
  }, []);

  async function loadDocuments(kbId) {
    if (!kbId || !token) return;
    const data = await api(`/api/knowledge-bases/${kbId}/documents`, { token });
    setDocuments(data.items || []);
  }

  useEffect(() => {
    function handleAuthExpired() {
      logout();
      setError('登录已失效，请重新登录。');
    }
    window.addEventListener('agentbase-auth-expired', handleAuthExpired);
    return () => window.removeEventListener('agentbase-auth-expired', handleAuthExpired);
  }, []);

  useEffect(() => {
    if (token) {
      const route = readChatRoute();
      setAppBootstrapping(true);
      setRouteRestoring(Boolean(isChatHomeRoute(route) && route.sessionId));
      bootstrap().catch((err) => {
        if (isAuthError(err)) {
          logout();
          setError('登录已失效，请重新登录。');
        } else {
          setError(errorMessage(err));
        }
        setRouteRestoring(false);
      }).finally(() => setAppBootstrapping(false));
    }
  }, [token]);

  useEffect(() => {
    if (activeAgentId) {
      loadAgent(activeAgentId).catch((err) => {
        if (isAuthError(err)) {
          logout();
          setError('登录已失效，请重新登录。');
        } else {
          setError(errorMessage(err));
        }
        setRouteRestoring(false);
      });
    }
  }, [activeAgentId]);

  useEffect(() => {
    if (!token) return undefined;
    const handlePopState = () => {
      const route = readChatRoute();
      const routeIsChat = isChatHomeRoute(route);
      pendingChatRestoreRef.current = route;
      setView(route.view);
      setActiveNav(route.nav);
      if (route.agentId && !sameRouteId(route.agentId, activeAgentIdRef.current)) {
        setRouteRestoring(Boolean(routeIsChat && route.sessionId));
        setActiveAgentId(Number(route.agentId));
        return;
      }
      if (!routeIsChat) {
        setRouteRestoring(false);
        return;
      }
      if (route.sessionId) {
        if (sameRouteId(route.sessionId, activeSessionIdRef.current)) {
          setRouteRestoring(false);
          setView('home');
          setActiveNav('chat');
          return;
        }
        setRouteRestoring(true);
        loadSession(route.sessionId, { openHome: true, updateUrl: false })
          .catch((err) => {
            setError(errorMessage(err));
            writeChatRoute(route.agentId || activeAgentIdRef.current, null, { mode: 'replace' });
          })
          .finally(() => setRouteRestoring(false));
        return;
      }
      setRouteRestoring(false);
      startNewChat({ updateUrl: false });
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [token, activeAgentId, activeSessionId, agentForm.variables, view]);

  useEffect(() => {
    if (!token || routeRestoring) return;
    const currentRoute = readChatRoute();
    if (isChatHomeRoute(currentRoute) && currentRoute.sessionId && activeNav === 'chat' && !activeSessionId) {
      return;
    }
    const options = {
      mode: 'replace',
      sessionId: activeNav === 'chat' ? activeSessionId : null,
    };
    if (activeAgentId) options.agentId = activeAgentId;
    writePageRoute(view, activeNav, options);
  }, [token, routeRestoring, view, activeNav, activeAgentId, activeSessionId]);

  useEffect(() => {
    if (activeNav !== 'chat' || !chatAgents.length) return;
    if (!activeSummary || activeSummary.status !== 'published' || !activeSummary.published_version_id) {
      setActiveAgentId(chatAgents[0].id);
    }
  }, [activeNav, activeSummary, chatAgents]);

  useEffect(() => {
    if (!thinkingEnabled) return;
    const capability = reasoningCapabilityForModel(currentThinkingModel);
    if (!capability.supported) {
      setThinkingEnabled(false);
      if (currentThinkingModel) {
        setError('当前模型不支持深度思考，请更换支持推理的模型。');
      }
    }
  }, [
    currentThinkingModel?.id,
    currentThinkingModel?.source,
    currentThinkingModel?.model_name,
    currentThinkingModel?.chat_model,
    currentThinkingModel?.supports_reasoning,
    currentThinkingModel?.reasoning_type,
  ]);

  useEffect(() => {
    if (activeKbId) {
      loadDocuments(activeKbId).catch((err) => {
        if (isAuthError(err)) {
          logout();
          setError('登录已失效，请重新登录。');
        } else {
          setError(errorMessage(err));
        }
      });
    } else {
      setDocuments([]);
    }
  }, [activeKbId, token]);

  useEffect(() => {
    refreshRuntimeStatus().catch(() => {});
  }, []);

  async function refreshRuntimeStatus() {
    const health = await api('/api/health').catch(() => defaultRuntimeStatus());
    setRuntimeStatus(health);
    return health;
  }

  function requestDeleteConfirm(options) {
    return new Promise((resolve) => {
      setConfirmDialog({
        title: options.title || '确认删除',
        message: options.message || '删除后不可恢复。',
        detail: options.detail || '',
        confirmLabel: options.confirmLabel || '删除',
        cancelLabel: options.cancelLabel || '取消',
        tone: options.tone || 'danger',
        resolve,
      });
    });
  }

  function closeConfirmDialog(confirmed) {
    setConfirmDialog((dialog) => {
      if (dialog?.resolve) dialog.resolve(Boolean(confirmed));
      return null;
    });
  }

  async function bootstrap() {
    const profile = await api('/api/auth/me', { token });
    setMe(profile.user);
    const ws = await api('/api/workspaces/current', { token });
    setWorkspace(ws.workspace);
    const [health, agentList, kbList, toolList, skillList, modelList, userModelList, marketList, reviewList, promptTemplateList, memberList] = await Promise.all([
      api('/api/health').catch(() => defaultRuntimeStatus()),
      api('/api/agents', { token }),
      api('/api/knowledge-bases', { token }),
      api('/api/tools', { token }),
      api('/api/skills', { token }),
      api('/api/models', { token }),
      api('/api/user-models', { token }).catch(() => ({ items: [] })),
      api('/api/market/agents', { token }).catch(() => ({ items: [] })),
      api('/api/admin/agent-reviews', { token }).catch(() => ({ items: [] })),
      api('/api/prompt-templates', { token }).catch(() => ({ items: [] })),
      api('/api/workspaces/members', { token }).catch(() => ({ items: [] })),
    ]);
    setAgents(agentList.items);
    setKnowledgeBases(kbList.items);
    setTools(toolList.items);
    setSkills(skillList.items || []);
    setPromptTemplates(promptTemplateList.items || []);
    setModels(modelList.items || []);
    setAdminModels(modelList.items || []);
    setUserModels(userModelList.items || []);
    setRuntimeStatus(health);
    if (isAdminRole(ws.workspace?.role)) {
      const adminModelList = await api('/api/models?include_disabled=true', { token }).catch(() => modelList);
      setAdminModels(adminModelList.items || []);
    }
    setMarketAgents(marketList.items || []);
    setReviewItems(reviewList.items || []);
    setMembers(memberList.items || []);
    const publishedAgents = agentList.items.filter((item) => item.status === 'published' && item.published_version_id);
    const rememberedRoute = pendingChatRestoreRef.current || readChatRoute();
    const rememberedAgent = rememberedRoute.agentId
      ? agentList.items.find((item) => sameRouteId(item.id, rememberedRoute.agentId))
      : null;
    if (rememberedRoute.agentId && !rememberedAgent) {
      pendingChatRestoreRef.current = null;
      clearChatRoute();
    }
    const fallbackAgent = rememberedAgent || publishedAgents[0] || agentList.items[0];
    if (!activeAgentId && fallbackAgent) {
      setActiveAgentId(fallbackAgent.id);
    } else if (activeAgentId && !agentList.items.some((item) => item.id === activeAgentId) && fallbackAgent) {
      setActiveAgentId(fallbackAgent.id);
    } else if (!fallbackAgent) {
      setRouteRestoring(false);
    }
  }

  function logout() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY);
    clearChatRoute();
    setToken('');
    setMe(null);
    setWorkspace(null);
    setAgents([]);
    setActiveAgentId(null);
    setActiveAgent(null);
    setKnowledgeBases([]);
    setTools([]);
    setPromptTemplates([]);
    setModels([]);
    setAdminModels([]);
    setUserModels([]);
    setRuntimeStatus(defaultRuntimeStatus());
    setMarketAgents([]);
    setReviewItems([]);
    setMembers([]);
    setSessions([]);
    setActiveSessionId(null);
    setActiveNewRunKey('');
    activeNewRunKeyRef.current = null;
    setSessionTitleDraft('');
    messageCacheRef.current = {};
    sourcesCacheRef.current = {};
    replaceVisibleMessages([]);
    setSources([]);
    setToolDebugEvents([]);
    setDocuments([]);
    setFeedbackByMessage({});
    setChatMode('published');
    setChatVariables({});
    setRagEnabled(false);
    setThinkingEnabled(true);
    setSearchEnabled(false);
    setDraft('');
    setHomePrompt('');
    setProfileError('');
    setMemoryProfile(defaultMemoryProfile());
    setMemoryProfileDraft(profileToDraft(defaultMemoryProfile()));
    setMemoryProfileLoading(false);
    setMemoryProfileSaving(false);
    setMemoryProfileError('');
    setError('');
    setAppBootstrapping(false);
    setView('home');
    setActiveNav('chat');
    setAccountMenuOpen(false);
  }

  async function authenticate(event) {
    event.preventDefault();
    setError('');
    const path = authMode === 'register' ? '/api/auth/register' : '/api/auth/login';
    const payload = authMode === 'register'
      ? { email: authForm.email, name: authForm.name, password: authForm.password }
      : { email: authForm.email, password: authForm.password };
    const data = await api(path, { method: 'POST', body: payload });
    localStorage.setItem(AUTH_TOKEN_KEY, data.access_token);
    setToken(data.access_token);
  }

  async function loadAgent(agentId) {
    const rememberedRoute = pendingChatRestoreRef.current;
    const isPendingChatSessionRestore = rememberedRoute?.sessionId
      && isChatHomeRoute(rememberedRoute)
      && sameRouteId(rememberedRoute.agentId, agentId);
    const data = await api(`/api/agents/${agentId}`, { token });
    const agent = data.agent;
    setActiveAgent(agent);
    setAgentForm({
      name: agent.name || '',
      avatar: agent.avatar || 'AI',
      description: agent.description || '',
      opening_message: '',
      system_prompt: agent.system_prompt || '',
      model_id: agent.model_id || agent.model_config?.id || '',
      user_model_config_id: agent.user_model_config_id || agent.user_model_config?.id || '',
      model: agent.model || '',
      temperature: agent.temperature ?? 0.4,
      knowledge_base_ids: agent.knowledge_base_ids || [],
      tool_ids: (agent.tools || []).map((tool) => tool.id),
      skill_ids: (agent.skills || []).map((s) => s.id),
      suggested_questions: agent.suggested_questions || [],
      variables: agent.variables || [],
      memory: agent.memory || { enabled: false, strategy: 'session_summary', max_messages: 12 },
      rag: agent.rag || { enabled_by_default: true, top_k: 4 },
      tool_policy: agent.tool_policy || { mode: 'auto', allowed_tool_names: [] },
    });
    setRagEnabled(false);
    setThinkingEnabled(true);
    setSearchEnabled(false);
    setChatVariables(initVariableValues(agent.variables || []));
    setActiveSessionId(isPendingChatSessionRestore ? rememberedRoute.sessionId : null);
    setActiveNewRunKey('');
    activeNewRunKeyRef.current = null;
    if (!isPendingChatSessionRestore) {
      replaceVisibleMessages([]);
      setSources([]);
    }
    setToolDebugEvents([]);
    setFeedbackByMessage({});
    await loadMemoryProfile(agentId);
    const loadedSessions = await loadSessions(agentId);
    const latestRememberedRoute = pendingChatRestoreRef.current;
    const rememberedIsChatRoute = isChatHomeRoute(latestRememberedRoute);
    const shouldRestoreSession = latestRememberedRoute?.sessionId
      && rememberedIsChatRoute
      && sameRouteId(latestRememberedRoute.agentId, agentId)
      && loadedSessions.some((session) => sameRouteId(session.id, latestRememberedRoute.sessionId));
    if (shouldRestoreSession) {
      pendingChatRestoreRef.current = null;
      try {
        await loadSession(latestRememberedRoute.sessionId, { openHome: true, updateUrl: false });
      } finally {
        setRouteRestoring(false);
      }
    } else if (latestRememberedRoute?.agentId && sameRouteId(latestRememberedRoute.agentId, agentId)) {
      pendingChatRestoreRef.current = null;
      if (rememberedIsChatRoute) {
        writeChatRoute(agentId, null, { mode: 'replace' });
      }
      setRouteRestoring(false);
    } else {
      setRouteRestoring(false);
    }
  }

  async function loadSessions(agentId) {
    const data = await api(`/api/agents/${agentId}/sessions`, { token });
    const items = data.items || [];
    setSessions(items);
    syncListedSessionRuns(items);
    return items;
  }

  async function loadMemoryProfile(agentId = activeAgentId) {
    if (!agentId) return null;
    setMemoryProfileLoading(true);
    setMemoryProfileError('');
    try {
      const data = await api(`/api/agents/${agentId}/memory-profile`, { token });
      const profile = normalizeMemoryProfile(data.profile, agentId);
      setMemoryProfile(profile);
      setMemoryProfileDraft(profileToDraft(profile));
      return profile;
    } catch (err) {
      setMemoryProfile(defaultMemoryProfile(agentId));
      setMemoryProfileDraft(profileToDraft(defaultMemoryProfile(agentId)));
      setMemoryProfileError(errorMessage(err));
      return null;
    } finally {
      setMemoryProfileLoading(false);
    }
  }

  async function saveMemoryProfile() {
    if (!activeAgentId) return;
    setMemoryProfileSaving(true);
    setMemoryProfileError('');
    try {
      const payload = memoryProfilePayload(memoryProfileDraft);
      const data = await api(`/api/agents/${activeAgentId}/memory-profile`, { token, method: 'PATCH', body: payload });
      const profile = normalizeMemoryProfile(data.profile, activeAgentId);
      setMemoryProfile(profile);
      setMemoryProfileDraft(profileToDraft(profile));
    } catch (err) {
      setMemoryProfileError(errorMessage(err));
      throw err;
    } finally {
      setMemoryProfileSaving(false);
    }
  }

  async function deleteMemoryProfile() {
    if (!activeAgentId) return;
    const confirmed = await requestDeleteConfirm({
      title: '删除用户记忆',
      message: '删除当前用户在这个智能体上的资料记忆？',
      detail: '会话摘要不会被删除。',
      confirmLabel: '删除资料',
    });
    if (!confirmed) return;
    setMemoryProfileSaving(true);
    setMemoryProfileError('');
    try {
      await api(`/api/agents/${activeAgentId}/memory-profile`, { token, method: 'DELETE' });
      const profile = defaultMemoryProfile(activeAgentId);
      setMemoryProfile(profile);
      setMemoryProfileDraft(profileToDraft(profile));
    } catch (err) {
      setMemoryProfileError(errorMessage(err));
      throw err;
    } finally {
      setMemoryProfileSaving(false);
    }
  }

  async function loadSession(sessionId, options = {}) {
    const data = await api(`/api/sessions/${sessionId}`, { token });
    const activeRun = data.active_run;
    const cacheKey = sessionCacheKey(sessionId);
    const cachedMessages = messageCacheRef.current[cacheKey];
    const loaded = data.messages || [];
    const displayMessages = activeRun?.status === 'running' && cachedMessages?.length
      ? cachedMessages
      : loaded;
    if (options.updateUrl !== false) {
      writeChatRoute(data.session?.agent_id || activeAgentId, sessionId, { mode: options.historyMode || 'replace' });
    }
    setActiveSessionId(sessionId);
    setActiveNewRunKey('');
    activeNewRunKeyRef.current = null;
    setSessionTitleDraft(data.session?.title || '');
    writeSessionMessages(sessionId, displayMessages, { visible: true });
    writeSessionSources(sessionId, sourcesCacheRef.current[cacheKey] || latestSourcesFromMessages(displayMessages), { visible: true });
    setToolDebugEvents([]);
    setFeedbackByMessage({});
    if (options.openHome) {
      setView('home');
      setActiveNav('chat');
    }
    // If there's an active run, reconnect to its SSE stream
    if (activeRun && activeRun.status === 'running') {
      const existingRun = sessionRunRef.current[sessionId];
      if (!existingRun?.controller) {
        reconnectToRun(activeRun.id, sessionId, displayMessages).catch(() => {});
      }
    }
  }

  async function reconnectToRun(runId, sessionId, existingMessages) {
    setSessionRunning(sessionId);
    setError('');
    const controller = new AbortController();
    const prevState = sessionRunRef.current[sessionId] || {};
    prevState.controller?.abort();
    const genId = (prevState.genId || 0) + 1;
    sessionRunRef.current[sessionId] = { runId, controller, genId };
    // Don't append a duplicate pending message if one already exists
    const lastMsg = existingMessages[existingMessages.length - 1];
    const needPending = !lastMsg || lastMsg.role === 'user'
      || (lastMsg.role === 'assistant' && lastMsg.meta?.cancelled);
    let streamEnded = false;
    try {
      const response = await fetch(`${API_BASE}/api/runs/${runId}/events`, {
        signal: controller.signal,
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) { streamEnded = true; return; }
      if (!response.body) { streamEnded = true; return; }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      if (needPending) {
        writeSessionMessages(sessionId, [...existingMessages, {
          role: 'assistant', content: '', pending: true, reasoning: '',
          reasoningTimeline: [], reasoningPending: false,
        }], { visible: true });
      }
      const ctx = { sessionKey: sessionId, genId, realSessionId: sessionId, isNewSession: false };
      while (true) {
        if (sessionRunRef.current[sessionId]?.genId !== genId) break;
        const { done, value } = await reader.read();
        if (done) { streamEnded = true; break; }
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          if (sessionRunRef.current[sessionId]?.genId !== genId) break;
          handleSse(part, ctx);
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') { /* ignored */ }
    } finally {
      if (sessionRunRef.current[sessionId]?.genId === genId) {
        clearSessionRunning(sessionId);
        // Reload session messages from DB so final assistant message
        // is properly synced (handles missed done/cancelled events).
        if (streamEnded && activeAgentId) {
          loadSession(sessionId, { historyMode: 'replace' }).catch(() => {});
        }
      }
      if (sessionRunRef.current[sessionId]?.controller === controller) {
        delete sessionRunRef.current[sessionId];
      }
    }
  }

  async function renameSession() {
    if (!activeSessionId || !sessionTitleDraft.trim()) return;
    const data = await api(`/api/sessions/${activeSessionId}`, {
      token,
      method: 'PATCH',
      body: { title: sessionTitleDraft.trim() },
    });
    setSessionTitleDraft(data.session.title);
    if (activeAgentId) await loadSessions(activeAgentId);
  }

  async function renameSessionById(sessionId, title) {
    const nextTitle = title.trim();
    if (!sessionId || !nextTitle) return null;
    const data = await api(`/api/sessions/${sessionId}`, {
      token,
      method: 'PATCH',
      body: { title: nextTitle },
    });
    if (sessionId === activeSessionId) {
      setSessionTitleDraft(data.session.title);
    }
    if (activeAgentId) await loadSessions(activeAgentId);
    return data.session;
  }

  async function deleteSession(sessionId) {
    if (!sessionId) return;
    await api(`/api/sessions/${sessionId}`, { token, method: 'DELETE' });
    delete messageCacheRef.current[sessionCacheKey(sessionId)];
    delete sourcesCacheRef.current[sessionCacheKey(sessionId)];
    const rememberedRoute = readChatRoute();
    if (sameRouteId(rememberedRoute.sessionId, sessionId)) {
      writeChatRoute(rememberedRoute.agentId || activeAgentId, null, { mode: 'replace' });
    }
    if (sessionId === activeSessionId) {
      setActiveSessionId(null);
      setActiveNewRunKey('');
      activeNewRunKeyRef.current = null;
      setSessionTitleDraft('');
      replaceVisibleMessages([]);
      setSources([]);
      setToolDebugEvents([]);
      setFeedbackByMessage({});
      setDraft('');
      setHomePrompt('');
      setChatAttachments([]);
    }
    if (activeAgentId) await loadSessions(activeAgentId);
  }

  async function updateProfile(patch) {
    setProfileError('');
    const data = await api('/api/auth/me', { token, method: 'PATCH', body: patch });
    setMe(data.user);
    return data.user;
  }

  async function refreshModels(includeDisabled = false) {
    const data = await api(includeDisabled ? '/api/models?include_disabled=true' : '/api/models', { token });
    if (includeDisabled) {
      setAdminModels(data.items || []);
    } else {
      setModels(data.items || []);
    }
    return data.items || [];
  }

  async function createModelConfig(payload) {
    await api('/api/admin/models', { token, method: 'POST', body: payload });
    await refreshModels(false);
    await refreshModels(true);
  }

  async function updateModelConfig(modelId, patch) {
    await api(`/api/admin/models/${modelId}`, { token, method: 'PATCH', body: patch });
    await refreshModels(false);
    await refreshModels(true);
  }

  async function deleteModelConfig(modelId) {
    await api(`/api/admin/models/${modelId}`, { token, method: 'DELETE' });
    await refreshModels(false);
    await refreshModels(true);
  }

  async function refreshUserModels() {
    const data = await api('/api/user-models', { token });
    setUserModels(data.items || []);
    return data.items || [];
  }

  async function createUserModelConfig(payload) {
    const data = await api('/api/user-models', { token, method: 'POST', body: payload });
    await refreshUserModels();
    return data.model_config;
  }

  async function testUserModelDraft(payload) {
    return api('/api/user-models/test', { token, method: 'POST', body: payload });
  }

  async function updateUserModelConfig(configId, patch) {
    const data = await api(`/api/user-models/${configId}`, { token, method: 'PATCH', body: patch });
    await refreshUserModels();
    return data.model_config;
  }


  async function deleteUserModelConfig(configId) {
    await api(`/api/user-models/${configId}`, { token, method: 'DELETE' });
    await refreshUserModels();
  }

  async function testUserModelConfig(configId) {
    return api(`/api/user-models/${configId}/test?detect_image=true`, { token, method: 'POST' });
  }

  async function refreshTools() {
    const data = await api('/api/tools', { token });
    setTools(data.items || []);
    return data.items || [];
  }

  async function createToolConfig(payload) {
    if (Array.isArray(payload)) {
      const created = await Promise.all(payload.map((item) => api('/api/tools', { token, method: 'POST', body: item })));
      await refreshTools();
      return created.map((item) => item.tool).filter(Boolean);
    }
    const data = await api('/api/tools', { token, method: 'POST', body: payload });
    await refreshTools();
    return data.tool;
  }

  async function discoverMcpTools(payload) {
    return api('/api/tools/mcp/discover', { token, method: 'POST', body: payload });
  }

  async function updateToolConfig(toolId, patch) {
    if (Array.isArray(toolId)) {
      const updated = await Promise.all(
        toolId.map((item) => api(`/api/tools/${item.id}`, { token, method: 'PATCH', body: item.patch }))
      );
      await refreshTools();
      if (activeAgentId) {
        await loadAgent(activeAgentId);
      }
      return updated.map((item) => item.tool).filter(Boolean);
    }
    await api(`/api/tools/${toolId}`, { token, method: 'PATCH', body: patch });
    await refreshTools();
    if (activeAgentId) {
      await loadAgent(activeAgentId);
    }
  }

  async function deleteToolConfig(toolId) {
    if (Array.isArray(toolId)) {
      await Promise.all(toolId.map((id) => api(`/api/tools/${id}`, { token, method: 'DELETE' })));
      await refreshTools();
      if (activeAgentId) {
        await loadAgent(activeAgentId);
      }
      return;
    }
    await api(`/api/tools/${toolId}`, { token, method: 'DELETE' });
    await refreshTools();
    if (activeAgentId) {
      await loadAgent(activeAgentId);
    }
  }

  async function testToolConfig(toolId, payload) {
    return api(`/api/tools/${toolId}/test`, { token, method: 'POST', body: payload });
  }

  async function refreshPromptTemplates(includeDisabled = false) {
    const data = await api(includeDisabled ? '/api/prompt-templates?include_disabled=true' : '/api/prompt-templates', { token });
    setPromptTemplates(data.items || []);
    return data.items || [];
  }

  async function createPromptTemplate(payload) {
    const data = await api('/api/prompt-templates', { token, method: 'POST', body: payload });
    await refreshPromptTemplates();
    return data.template;
  }

  async function updatePromptTemplate(templateId, patch) {
    const data = await api(`/api/prompt-templates/${templateId}`, { token, method: 'PATCH', body: patch });
    await refreshPromptTemplates(true);
    return data.template;
  }

  async function deletePromptTemplate(templateId) {
    await api(`/api/prompt-templates/${templateId}`, { token, method: 'DELETE' });
    await refreshPromptTemplates(true);
  }

  async function copyBuiltinPromptTemplate(payload) {
    const data = await api('/api/prompt-templates/copy-builtin', { token, method: 'POST', body: payload });
    await refreshPromptTemplates();
    return data.template;
  }

  function openAgentIdentityDialog(mode = 'edit') {
    setAgentIdentityError('');
    setAgentIdentityDialog({
      mode,
      form: mode === 'create'
        ? pickAgentIdentity(defaultAgentForm())
        : pickAgentIdentity(agentForm),
    });
  }

  async function submitAgentIdentity(identity) {
    const nextIdentity = normalizeAgentIdentity(identity);
    if (!nextIdentity.name) {
      setAgentIdentityError('请填写智能体名称。');
      return;
    }
    setAgentIdentitySaving(true);
    setAgentIdentityError('');
    try {
      if (agentIdentityDialog?.mode === 'create') {
        const form = { ...defaultAgentForm(), ...nextIdentity };
        const data = await api('/api/agents', { token, method: 'POST', body: agentPayload(form) });
        await bootstrap();
        setActiveAgentId(data.agent.id);
        setView('builder');
        setActiveNav('agents');
      } else if (activeAgentId) {
        const form = { ...agentForm, ...nextIdentity };
        setAgentForm(form);
        await api(`/api/agents/${activeAgentId}`, { token, method: 'PATCH', body: agentPayload(form) });
        await bootstrap();
        await loadAgent(activeAgentId);
      }
      setAgentIdentityDialog(null);
    } catch (err) {
      setAgentIdentityError(errorMessage(err));
    } finally {
      setAgentIdentitySaving(false);
    }
  }

  async function createAgent(openBuilder = true) {
    if (openBuilder) {
      openAgentIdentityDialog('create');
      return;
    }
    const data = await api('/api/agents', { token, method: 'POST', body: agentPayload(defaultAgentForm()) });
    await bootstrap();
    setActiveAgentId(data.agent.id);
  }

  async function saveAgent() {
    if (!activeAgentId) return;
    const body = agentPayload(agentForm, { model: selectedDraftModel });
    await api(`/api/agents/${activeAgentId}`, { token, method: 'PATCH', body });
    // Sync skills binding separately
    await updateAgentSkills(activeAgentId, numericIdList(agentForm.skill_ids || []));
    await bootstrap();
    await loadAgent(activeAgentId);
  }

  async function publishAgent() {
    if (!activeAgentId) return;
    await saveAgent();
    const data = await api(`/api/agents/${activeAgentId}/publish`, { token, method: 'POST' });
    notify(data.review_required ? '已提交管理员审核，通过后会出现在市场。' : '已发布到市场。');
    await bootstrap();
  }

  async function copyMarketAgent(agentId) {
    const data = await api(`/api/market/agents/${agentId}/copy`, { token, method: 'POST' });
    await bootstrap();
    setActiveAgentId(data.agent.id);
    setView('builder');
    setActiveNav('agents');
    notify('已复制到你的智能体草稿。');
  }

  async function approveReview(agentId) {
    await api(`/api/admin/agent-reviews/${agentId}/approve`, { token, method: 'POST' });
    await bootstrap();
    notify('审核通过，智能体已上架市场。');
  }

  async function rejectReview(agentId) {
    await api(`/api/admin/agent-reviews/${agentId}/reject`, { token, method: 'POST' });
    await bootstrap();
    notify('已驳回该智能体发布申请。');
  }

  async function createKnowledgeBase(payload = defaultKnowledgeBaseForm()) {
    const body = {
      name: String(payload?.name || '').trim(),
      description: String(payload?.description || '').trim(),
    };
    if (!body.name) {
      throw new Error('知识库名称不能为空。');
    }
    const data = await api('/api/knowledge-bases', { token, method: 'POST', body });
    await bootstrap();
    return data.knowledge_base;
  }

  async function createSkill(rawForm) {
    const body = skillFormPayload(rawForm);
    if (!body.name) {
      throw new Error('技能名称不能为空。');
    }
    const data = await api('/api/skills', { token, method: 'POST', body });
    await loadSkills();
    return data.skill;
  }

  async function updateSkill(skillId, rawForm) {
    const body = skillFormPayload(rawForm);
    if (!body.name) {
      throw new Error('技能名称不能为空。');
    }
    const data = await api(`/api/skills/${skillId}`, { token, method: 'PATCH', body });
    await loadSkills();
    return data.skill;
  }

  async function deleteSkill(skillId) {
    await api(`/api/skills/${skillId}`, { token, method: 'DELETE' });
    await loadSkills();
  }

  async function loadSkills() {
    const data = await api('/api/skills', { token });
    setSkills(data.items || []);
  }

  async function updateAgentSkills(agentId, skillIds) {
    await api(`/api/agents/${agentId}/skills`, {
      token,
      method: 'PUT',
      body: { skill_ids: skillIds },
    });
  }

  async function updateKnowledgeBase(kbId, payload) {
    const body = {
      name: String(payload?.name || '').trim(),
      description: String(payload?.description || '').trim(),
    };
    if (!body.name) {
      throw new Error('知识库名称不能为空。');
    }
    const data = await api(`/api/knowledge-bases/${kbId}`, { token, method: 'PATCH', body });
    await bootstrap();
    return data.knowledge_base;
  }

  async function uploadDocument() {
    const kbId = Number(docForm.kb_id || knowledgeBases[0]?.id);
    if (!kbId) {
      setError('请先创建知识库。');
      return;
    }
    const filename = String(docForm.filename || 'guide.txt').trim();
    const text = String(docForm.text || '').trim();
    if (!text) {
      setError('请先粘贴要写入知识库的文本。');
      return;
    }
    await api(`/api/knowledge-bases/${kbId}/documents`, {
      token,
      method: 'POST',
      body: {
        title: filename,
        filename,
        content: text,
        content_type: 'text/plain',
        source_type: 'text',
      },
    });
    setDocForm({ filename: 'guide.txt', text: '', kb_id: String(kbId) });
    await loadDocuments(kbId);
    await bootstrap();
  }

  async function uploadKnowledgeFile(fileOrFiles, targetKbId = null) {
    const files = Array.isArray(fileOrFiles) ? fileOrFiles.filter(Boolean) : [fileOrFiles].filter(Boolean);
    if (!files.length || !token) return;
    const kbId = Number(targetKbId || docForm.kb_id || knowledgeBases[0]?.id);
    if (!kbId) {
      setError('请先创建或选择知识库。');
      return;
    }
    const uploadBatchId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const uploadItems = files.map((file, index) => ({
      id: `${uploadBatchId}-${index}`,
      kbId,
      filename: file.name,
      status: 'queued',
      statusText: '等待上传',
    }));
    const updateUploadItem = (index, patch) => {
      const itemId = uploadItems[index]?.id;
      if (!itemId) return;
      setUploadingKnowledgeItems((items) => items.map((item) => (item.id === itemId ? { ...item, ...patch } : item)));
    };
    setUploadingKnowledgeFile(true);
    setUploadingKnowledgeItems((items) => [...items, ...uploadItems]);
    setError('');
    try {
      await new Promise((resolve) => setTimeout(resolve, 0));
      const documents = [];
      const preparationErrors = [];
      for (const [index, file] of files.entries()) {
        try {
          updateUploadItem(index, { status: 'reading', statusText: '读取文件' });
          validateKnowledgeFile(file);
          const contentType = file.type || guessContentType(file.name);
          const contentBase64 = await fileToBase64(file);
          documents.push({
            uploadIndex: index,
            filename: file.name,
            title: file.name,
            content_type: contentType,
            content_base64: contentBase64,
            source_type: 'file',
          });
          updateUploadItem(index, { status: 'queued', statusText: '等待上传' });
        } catch (prepareErr) {
          preparationErrors.push({ filename: file.name, message: errorMessage(prepareErr) });
          updateUploadItem(index, { status: 'failed', statusText: errorMessage(prepareErr) });
        }
      }
      if (!documents.length) {
        const firstError = preparationErrors[0];
        setError(firstError ? `${firstError.filename} ${firstError.message}` : '没有可上传的文档。');
        return;
      }
      const requestDocuments = documents.map(({ uploadIndex, ...document }) => document);
      if (documents.length > 1) {
        let result;
        try {
          setUploadingKnowledgeItems((items) => items.map((item) => (
            documents.some((document) => uploadItems[document.uploadIndex]?.id === item.id)
              ? { ...item, status: 'uploading', statusText: '上传并索引中' }
              : item
          )));
          result = await api(`/api/knowledge-bases/${kbId}/documents/batch`, {
            token,
            method: 'POST',
            body: { documents: requestDocuments },
          });
        } catch (err) {
          if (err?.status !== 404 && err?.status !== 405) throw err;
          result = { succeeded: 0, failed: 0, errors: [] };
          for (const [requestIndex, document] of requestDocuments.entries()) {
            const uploadIndex = documents[requestIndex].uploadIndex;
            updateUploadItem(uploadIndex, { status: 'uploading', statusText: '上传并索引中' });
            try {
              await api(`/api/knowledge-bases/${kbId}/documents`, {
                token,
                method: 'POST',
                body: document,
              });
              result.succeeded += 1;
              updateUploadItem(uploadIndex, { status: 'indexed', statusText: '已上传并索引' });
            } catch (singleErr) {
              result.failed += 1;
              updateUploadItem(uploadIndex, { status: 'failed', statusText: errorMessage(singleErr) });
              result.errors.push({
                filename: document.filename,
                message: errorMessage(singleErr),
              });
            }
          }
        }
        if (result.documents || result.errors) {
          const failedRequestIndexes = new Map((result.errors || []).map((item) => [Number(item.index), item]));
          requestDocuments.forEach((document, requestIndex) => {
            const uploadIndex = documents[requestIndex].uploadIndex;
            const failed = failedRequestIndexes.get(requestIndex);
            updateUploadItem(uploadIndex, failed
              ? { status: 'failed', statusText: failed.message || '上传失败' }
              : { status: 'indexed', statusText: '已上传并索引' });
          });
        }
        if (result.failed) {
          const firstError = result.errors?.[0];
          setError(`${result.succeeded || 0} 个文档上传成功，${result.failed} 个失败${firstError ? `：${firstError.filename || '文档'} ${firstError.message || ''}` : '。'}`);
        }
      } else {
        const uploadIndex = documents[0].uploadIndex;
        updateUploadItem(uploadIndex, { status: 'uploading', statusText: '上传并索引中' });
        await api(`/api/knowledge-bases/${kbId}/documents`, {
          token,
          method: 'POST',
          body: requestDocuments[0],
        });
        updateUploadItem(uploadIndex, { status: 'indexed', statusText: '已上传并索引' });
      }
      const lastFile = files[files.length - 1];
      setDocForm((form) => ({ ...form, filename: lastFile.name, text: '', kb_id: String(kbId) }));
      await loadDocuments(kbId);
      await bootstrap();
    } catch (err) {
      const message = errorMessage(err);
      setError(message);
      setUploadingKnowledgeItems((items) => items.map((item) => (
        uploadItems.some((uploadItem) => uploadItem.id === item.id) && item.status !== 'indexed'
          ? { ...item, status: 'failed', statusText: message }
          : item
      )));
    } finally {
      setUploadingKnowledgeFile(false);
      setTimeout(() => {
        setUploadingKnowledgeItems((items) => items.filter((item) => !uploadItems.some((uploadItem) => uploadItem.id === item.id)));
      }, 1800);
    }
  }

  async function deleteDocument(documentId, targetKbId = null) {
    const kbId = Number(targetKbId || activeKbId);
    if (!kbId || !documentId) return false;
    const document = documents.find((item) => item.id === documentId);
    const confirmed = await requestDeleteConfirm({
      title: '删除知识文档',
      message: `\u5220\u9664\u6587\u6863\u300c${document?.title || document?.filename || `document-${documentId}`}\u300d\uff1f`,
      detail: '该文档在 PostgreSQL 中的记录、分块，以及 Milvus 中的向量索引会一起删除。',
      confirmLabel: '删除文档',
    });
    if (!confirmed) return false;
    await api(`/api/knowledge-bases/${kbId}/documents/${documentId}`, { token, method: 'DELETE' });
    await loadDocuments(kbId);
    await bootstrap();
    notify('知识文档已删除。');
    return true;
  }

  async function clearKnowledgeBaseDocuments(kb) {
    if (!kb?.id) return false;
    const confirmed = await requestDeleteConfirm({
      title: '清除文档库',
      message: `清空知识库「${kb.name || '未命名知识库'}」中的所有文档？`,
      detail: '知识库本身会保留，但 PostgreSQL 中的所有文档、分块，以及 Milvus 中该知识库的向量索引会一起删除。',
      confirmLabel: '清除文档库',
    });
    if (!confirmed) return false;
    const result = await api(`/api/knowledge-bases/${kb.id}/documents`, { token, method: 'DELETE' });
    if (String(activeKbId) === String(kb.id)) {
      setDocuments([]);
    }
    await bootstrap();
    notify(`已清除 ${result.documents_deleted ?? 0} 个文档。`);
    return true;
  }

  async function deleteKnowledgeBase(kb) {
    if (!kb?.id) return;
    const confirmed = await requestDeleteConfirm({
      title: '删除知识库',
      message: `删除知识库「${kb.name}」？`,
      detail: '所有文档、分块和向量数据会一起删除。',
      confirmLabel: '删除知识库',
    });
    if (!confirmed) return;
    await api(`/api/knowledge-bases/${kb.id}`, { token, method: 'DELETE' });
    if (String(activeKbId) === String(kb.id)) {
      setDocuments([]);
    }
    await bootstrap();
    notify('知识库已删除。');
  }

  async function deleteAgent(agent) {
    if (!agent?.id || agent.is_template) return;
    const allowed = canManage || agent.created_by === me?.id;
    if (!allowed) return;
    const confirmed = await requestDeleteConfirm({
      title: '删除智能体',
      message: `删除智能体「${agent.name || '未命名智能体'}」？`,
      detail: '删除后无法恢复，确定要删除吗？',
      confirmLabel: '删除智能体',
    });
    if (!confirmed) return;

    await api(`/api/agents/${agent.id}`, { token, method: 'DELETE' });
    const remaining = agents.filter((item) => item.id !== agent.id);
    const nextId = remaining[0]?.id || null;
    setAgents(remaining);
    setActiveAgentId(nextId);
    setActiveSessionId(null);
    setActiveNewRunKey('');
    activeNewRunKeyRef.current = null;
    setSessionTitleDraft('');
    messageCacheRef.current = {};
    sourcesCacheRef.current = {};
    replaceVisibleMessages([]);
    setSources([]);
    setFeedbackByMessage({});
    setChatVariables({});
    setView('home');
    setActiveNav(nextId ? 'agents' : 'chat');
    const rememberedRoute = readChatRoute();
    if (sameRouteId(rememberedRoute.agentId, agent.id)) {
      if (nextId) writeChatRoute(nextId, null, { mode: 'replace' });
      else clearChatRoute({ mode: 'replace' });
    }
    await bootstrap();
    if (nextId) {
      await loadAgent(nextId);
    } else {
      setActiveAgent(null);
      setAgentForm(defaultAgentForm());
      setSessions([]);
    }
  }

  function startNewChat(options = {}) {
    setActiveSessionId(null);
    setActiveNewRunKey('');
    activeNewRunKeyRef.current = null;
    setSessionTitleDraft('');
    newSessionPlaceholderIdRef.current = null;
    replaceVisibleMessages([]);
    setSources([]);
    setToolDebugEvents([]);
    setFeedbackByMessage({});
    setDraft('');
    setHomePrompt('');
    setError('');
    setChatVariables(initVariableValues(agentForm.variables || []));
    setRagEnabled(false);
    setThinkingEnabled(true);
    setSearchEnabled(false);
    setChatAttachments([]);
    if (options.updateUrl !== false) {
      writeChatRoute(activeAgentId, null, { mode: options.historyMode || 'push' });
    }
    if (view !== 'builder') {
      setView('home');
      setActiveNav('chat');
    }
  }

  async function submitFeedback(messageId, rating) {
    if (!messageId) return;
    const data = await api(`/api/messages/${messageId}/feedback`, {
      token,
      method: 'POST',
      body: { rating, comment: '' },
    });
    setFeedbackByMessage((items) => ({ ...items, [messageId]: data.feedback.rating }));
  }

  const chatModeRef = useRef(chatMode);
  useEffect(() => {
    chatModeRef.current = chatMode;
  }, [chatMode]);

  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);

  // Clean chat state when switching between draft debugging and published preview
  useEffect(() => {
    replaceVisibleMessages([]);
    setSources([]);
    setToolDebugEvents([]);
    setActiveSessionId(null);
    setActiveNewRunKey('');
    activeNewRunKeyRef.current = null;
    setError('');
  }, [chatMode]);

  async function sendMessage(event, explicitText) {
    event?.preventDefault();
    const text = (explicitText ?? draft ?? homePrompt).trim();
    const outgoingAttachments = chatAttachments;
    console.log("[DEBUG FRONTEND] sendMessage called. chatMode state:", chatMode, "chatModeRef.current:", chatModeRef.current, "view:", view, "viewRef.current:", viewRef.current);
    if ((!text && !outgoingAttachments.length) || !activeAgentId || busy) return;
    if (viewRef.current !== 'builder' && (!activeSummary || activeSummary.status !== 'published' || !activeSummary.published_version_id)) {
      const firstPublished = chatAgents[0];
      if (firstPublished) {
        setActiveAgentId(firstPublished.id);
      }
      setError('对话只能使用已经过审核并上架的智能体。');
      return;
    }
    const currentModel = viewRef.current === 'builder' ? selectedDraftModel : activeAgent?.user_model_config || activeAgent?.model_config || null;
    const modelWarning = modelCapabilityWarning(currentModel, outgoingAttachments);
    if (modelWarning) {
      setError(modelWarning);
      return;
    }
    const effectiveRagEnabled = ragRuntime.available && ragEnabled;
    const thinkingCapability = reasoningCapabilityForModel(currentModel);
    const effectiveThinkingEnabled = thinkingEnabled && thinkingCapability.supported;
    const effectiveSearchEnabled = webSearchRuntime.available && searchEnabled;
    if (thinkingEnabled && !thinkingCapability.supported) {
      setThinkingEnabled(false);
    }
    if (text) {
      setSubmittedPromptHistory((items) => [...items, text].slice(-100));
    }
    setDraft('');
    setHomePrompt('');
    // Capture session at request start — use this throughout, never activeSessionId directly
    const capturedSessionId = activeSessionId;
    newSessionIdRef.current = null;
    // Use a valid key for sessionRunRef (null is not a valid object key for our logic)
    const runKey = capturedSessionId || `__new__${Date.now()}`;
    let placeholderId = null;
    let streamCtx = null;
    const requestIsVisible = () => {
      if (capturedSessionId) {
        return String(activeSessionIdRef.current || '') === String(capturedSessionId);
      }
      return activeNewRunKeyRef.current === runKey;
    };
    // Track temp key so stopGeneration can find it before real session ID arrives
    if (!capturedSessionId) {
      activeNewRunKeyRef.current = runKey;
      setActiveNewRunKey(runKey);
    }
    // Abort previous in-flight request for THIS session only
    const prevRun = sessionRunRef.current[runKey];
    if (prevRun?.controller) {
      prevRun.controller.abort();
    }
    // Increment per-session generation to invalidate stale SSE events
    const newGenId = (prevRun?.genId || 0) + 1;
    // Set running state for this session
    setRunningBySessionId((prev) => ({ ...prev, [runKey]: { running: true } }));
    if (!capturedSessionId) {
      // Immediately show a placeholder session in the sidebar
      placeholderId = -Date.now();
      newSessionPlaceholderIdRef.current = placeholderId;
      setRunningBySessionId((prev) => ({ ...prev, [placeholderId]: { running: true } }));
      setSessions((prev) => [
        { id: placeholderId, agent_id: activeAgentId, title: '新会话', message_count: 0, created_at: new Date().toISOString(), updated_at: new Date().toISOString() },
        ...prev,
      ]);
    }
    setError('');
    setChatAttachments([]);
    const initialMessageKey = capturedSessionId || runKey;
    const initialBaseMessages = capturedSessionId
      ? (messageCacheRef.current[sessionCacheKey(capturedSessionId)] || messagesRef.current)
      : messagesRef.current;
    writeSessionMessages(initialMessageKey, [
      ...initialBaseMessages,
      { role: 'user', content: text, attachments: outgoingAttachments },
      {
        role: 'assistant',
        content: '',
        pending: true,
        reasoning: '',
        reasoningTimeline: [],
        reasoningVisible: effectiveThinkingEnabled,
        reasoningPending: effectiveThinkingEnabled,
        reasoningStartedAt: effectiveThinkingEnabled ? Date.now() : null,
        reasoningFinishedAt: null,
      },
    ], { visible: true });
    setToolDebugEvents([]);
    const controller = new AbortController();
    sessionRunRef.current[runKey] = { runId: null, controller, genId: newGenId };
    try {
      const response = await fetch(`${API_BASE}/api/agents/${activeAgentId}/chat/stream`, {
        signal: controller.signal,
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          message: text || '请分析附件内容。',
          session_id: capturedSessionId || null,
          mode: viewRef.current === 'builder' ? chatModeRef.current : 'published',
          is_debug: viewRef.current === 'builder',
          rag_enabled: effectiveRagEnabled,
          rag_options: agentForm.rag || undefined,
          thinking_enabled: effectiveThinkingEnabled,
          search_enabled: effectiveSearchEnabled,
          variables: castVariables(agentForm.variables || [], chatVariables),
          attachments: outgoingAttachments.map((item) => ({ id: item.id, type: item.type, mime_type: item.content_type })),
        }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        const error = new ApiError(errorMessage(data.detail || data.message || `HTTP ${response.status}`), response.status, data);
        if (isAuthError(error)) notifyAuthExpired();
        throw error;
      }
      if (!response.body) throw new Error('当前浏览器不支持流式响应。');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const genId = newGenId;
      streamCtx = { sessionKey: runKey, genId, realSessionId: capturedSessionId, isNewSession: !capturedSessionId, placeholderId, resolvedSessionId: null };
      while (true) {
        // Stop processing if a new generation has started for this session
        if (sessionRunRef.current[runKey]?.genId !== genId) break;
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          if (sessionRunRef.current[runKey]?.genId !== genId) break;
          handleSse(part, streamCtx);
        }
      }
    } catch (err) {
      // Don't show error for user-initiated abort
      if (err.name === 'AbortError') {
        // Already handled by stopGeneration or cancelled event
      } else if (isAuthError(err)) {
        logout();
        setError('登录已失效，请重新登录。');
      } else {
        if (!requestIsVisible()) return;
        const message = errorMessage(err);
        setError(message);
        updateSessionMessages(streamCtx?.resolvedSessionId || capturedSessionId || runKey, (items) => {
          const next = [...items];
          const last = next[next.length - 1];
          if (last?.role === 'assistant' && last.pending) {
            next[next.length - 1] = {
              ...last,
              pending: false,
              reasoningPending: false,
              reasoningFinishedAt: last.reasoningStartedAt && !last.reasoningFinishedAt ? Date.now() : last.reasoningFinishedAt,
              error: true,
              content: message,
            };
          }
          return next;
        }, { visible: true });
      }
    } finally {
      // Clean up per-session state — use runKey, not current activeSessionId
      const visibleAtCleanup = requestIsVisible();
      const finalKey = streamCtx?.resolvedSessionId || runKey;
      if (sessionRunRef.current[finalKey]?.genId === newGenId) {
        clearSessionRunning(finalKey);
      }
      if (finalKey !== runKey) {
        clearSessionRunning(runKey);
      }
      // Clean up ref entries
      if (sessionRunRef.current[finalKey]?.controller === controller) {
        delete sessionRunRef.current[finalKey];
      }
      if (finalKey !== runKey && sessionRunRef.current[runKey]?.controller === controller) {
        delete sessionRunRef.current[runKey];
      }
      if (newSessionIdRef.current === finalKey) {
        newSessionIdRef.current = null;
      }
      if (placeholderId) {
        clearSessionRunning(placeholderId);
      }
      if (placeholderId && newSessionPlaceholderIdRef.current === placeholderId) {
        newSessionPlaceholderIdRef.current = null;
      }
      if (activeNewRunKeyRef.current === runKey) {
        activeNewRunKeyRef.current = null;
        setActiveNewRunKey('');
      }
      if (activeAgentId) loadSessions(activeAgentId).catch((err) => setError(errorMessage(err)));
      if (visibleAtCleanup) {
        setChatAttachments([]);
      }
    }
  }

  async function stopGeneration() {
    // Only stop the CURRENT session's run — read from per-session state
    const lookupKey = activeSessionId || newSessionIdRef.current || activeNewRunKeyRef.current;
    const sessionState = lookupKey ? sessionRunRef.current[lookupKey] : null;
    const runId = sessionState?.runId;
    if (runId) {
      // Notify backend to cancel — it will close the SSE stream cleanly
      fetch(`${API_BASE}/api/runs/${runId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      }).catch(() => {});
    }
    // Don't abort the fetch here — let the backend send the cancelled SSE event
    // which properly persists the partial response. The stream will end naturally.
    if (lookupKey) {
      clearSessionRunning(lookupKey);
    }
    if (activeNewRunKeyRef.current === lookupKey) {
      activeNewRunKeyRef.current = null;
      setActiveNewRunKey('');
    }
    updateSessionMessages(lookupKey, (items) => {
      const next = [...items];
      const last = next[next.length - 1];
      if (last?.role === 'assistant' && last.pending) {
        next[next.length - 1] = {
          ...last,
          pending: false,
          reasoningPending: false,
          reasoningFinishedAt: last.reasoningStartedAt && !last.reasoningFinishedAt ? Date.now() : last.reasoningFinishedAt,
        };
      }
      return next;
    }, { visible: true });
  }

  async function uploadChatAttachment(file) {
    if (!file || !token) return;
    setUploadingAttachment(true);
    setError('');
    try {
      if (file.size > MAX_UPLOAD_BYTES) {
        throw new Error('Upload file cannot exceed 8MB');
      }
      const contentType = file.type || guessContentType(file.name);
      const contentBase64 = await fileToBase64(file);
      const data = await api('/api/uploads', {
        token,
        method: 'POST',
        body: { filename: file.name, content_type: contentType, content_base64: contentBase64 },
      });
      setChatAttachments((items) => [...items, data.upload]);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUploadingAttachment(false);
    }
  }

  // Helper: is this stream's session currently displayed in the UI?
  function _streamIsVisible(realSessionId, isNewSession, sessionKey) {
    const currentView = sessionCacheKey(activeSessionIdRef.current);
    const realKey = sessionCacheKey(realSessionId);
    const streamKey = sessionCacheKey(sessionKey);
    if (currentView) {
      if (realKey) return currentView === realKey;
      return Boolean(isNewSession && streamKey && sessionCacheKey(activeNewRunKeyRef.current) === streamKey);
    }
    return Boolean(isNewSession && streamKey && sessionCacheKey(activeNewRunKeyRef.current) === streamKey);
  }

  function _streamCacheKey(ctx) {
    return sessionCacheKey(ctx?.resolvedSessionId || ctx?.realSessionId || ctx?.sessionKey);
  }

  function _streamVisible(ctx) {
    return _streamIsVisible(ctx?.realSessionId, ctx?.isNewSession, ctx?.sessionKey);
  }

  function ensureAssistantTail(items, defaults = {}) {
    const next = [...items];
    const last = next[next.length - 1];
    if (last?.role === 'assistant') {
      return { next, last, index: next.length - 1 };
    }
    const assistant = {
      role: 'assistant',
      content: '',
      pending: true,
      reasoning: '',
      reasoningTimeline: [],
      reasoningVisible: false,
      reasoningPending: false,
      ...defaults,
    };
    next.push(assistant);
    return { next, last: assistant, index: next.length - 1 };
  }

  function handleSse(raw, ctx) {
    const { sessionKey, genId } = ctx || {};
    // Abandon if a new generation started for this session
    if (sessionKey && sessionRunRef.current[sessionKey]?.genId !== genId) return;

    const event = raw.match(/^event: (.+)$/m)?.[1];
    const dataLine = raw.match(/^data: (.+)$/m)?.[1];
    if (!event) return;
    const data = dataLine ? JSON.parse(dataLine) : {};
    let visible = _streamVisible(ctx);
    let messageKey = _streamCacheKey(ctx);

    const resolveRealSession = (sessionId) => {
      if (!sessionId || !ctx) return;
      const previousKey = _streamCacheKey(ctx);
      ctx.resolvedSessionId = sessionId;
      ctx.realSessionId = sessionId;
      moveSessionCache(previousKey || sessionKey, sessionId);
      messageKey = _streamCacheKey(ctx);
      visible = _streamVisible(ctx);
    };

    if (event === 'run_snapshot') {
      const snapshotContent = String(data.content || '');
      const snapshotReasoning = String(data.reasoning || '');
      const snapshotSources = Array.isArray(data.sources) ? data.sources : [];
      if (snapshotSources.length) {
        writeSessionSources(messageKey, snapshotSources, { visible });
      }
      updateSessionMessages(messageKey, (currentMessages) => {
        const { next, last, index } = ensureAssistantTail(currentMessages, {
          pending: true,
          reasoningVisible: Boolean(snapshotReasoning),
          reasoningPending: Boolean(snapshotReasoning) && !snapshotContent,
          reasoningStartedAt: snapshotReasoning ? Date.now() : null,
        });
        let assistant = {
          ...last,
          pending: true,
          content: snapshotContent || last.content || '',
          reasoning: snapshotReasoning || last.reasoning || '',
          reasoningVisible: Boolean(snapshotReasoning) || last.reasoningVisible,
          reasoningPending: Boolean(snapshotReasoning) && !snapshotContent,
          reasoningStartedAt: snapshotReasoning ? (last.reasoningStartedAt || Date.now()) : last.reasoningStartedAt,
          sources: snapshotSources.length ? snapshotSources : last.sources,
          _provisionalContent: data.provisional_content || last._provisionalContent,
        };
        if (snapshotReasoning) {
          assistant.reasoningTimeline = [{
            id: `snapshot-reasoning-${data.event_index || Date.now()}`,
            type: 'reasoning',
            content: snapshotReasoning,
          }];
        }
        (Array.isArray(data.timeline_events) ? data.timeline_events : []).forEach((item) => {
          assistant = appendToolTimelineItem(assistant, item?.data || {}, item?.event || 'tool_call');
        });
        next[index] = assistant;
        return next;
      }, { visible });
      return;
    }

    if (event === 'run_started') {
      if (sessionKey && sessionRunRef.current[sessionKey]) {
        sessionRunRef.current[sessionKey].runId = data.run_id || null;
      }
      // Replace placeholder session ID with real session ID as early as possible
      if (data.session_id && ctx?.placeholderId) {
        resolveRealSession(data.session_id);
        if (ctx?.isNewSession && visible) {
          writeChatRoute(activeAgentId, data.session_id, { mode: 'replace' });
        }
        if (sessionKey?.startsWith?.('__new__') && sessionRunRef.current[sessionKey]) {
          sessionRunRef.current[data.session_id] = sessionRunRef.current[sessionKey];
          if (activeNewRunKeyRef.current === sessionKey) {
            newSessionIdRef.current = data.session_id;
          }
        }
        setRunningBySessionId((prev) => {
          const next = {
            ...prev,
            [data.session_id]: { running: true },
          };
          delete next[ctx.placeholderId];
          return next;
        });
        setSessions((prev) =>
          prev.map((s) => (s.id === ctx.placeholderId ? { ...s, id: data.session_id } : s))
        );
        if (newSessionPlaceholderIdRef.current === ctx.placeholderId) {
          newSessionPlaceholderIdRef.current = null;
        }
      }
      return;
    }
    if (event === 'cancelled') {
      // Always clean up running state for this session
      if (data.session_id) clearSessionRunning(data.session_id);
      if (sessionKey && sessionKey !== data.session_id) clearSessionRunning(sessionKey);
      // If temp key, transition so finally block can clean up properly
      if (sessionKey?.startsWith?.('__new__') && data.session_id) {
        resolveRealSession(data.session_id);
        if (sessionRunRef.current[sessionKey]) {
          sessionRunRef.current[data.session_id] = sessionRunRef.current[sessionKey];
        }
        if (activeNewRunKeyRef.current === sessionKey) {
          newSessionIdRef.current = data.session_id;
        }
      }
      // Replace placeholder session in sidebar with real session ID
      if (data.session_id && ctx?.placeholderId) {
        setSessions((prev) =>
          prev.map((s) => (s.id === ctx.placeholderId ? { ...s, id: data.session_id } : s))
        );
        if (newSessionPlaceholderIdRef.current === ctx.placeholderId) {
          newSessionPlaceholderIdRef.current = null;
        }
      }
      if (visible) {
        if (ctx?.isNewSession && data.session_id && !activeSessionIdRef.current) {
          writeChatRoute(activeAgentId, data.session_id, { mode: 'replace' });
          setActiveSessionId(data.session_id);
        }
      }
      updateSessionMessages(messageKey, (currentMessages) => {
        const { next, last, index } = ensureAssistantTail(currentMessages);
        next[index] = {
          ...last,
          id: data.message_id || last.id,
          run_id: data.run_id || last.run_id,
          pending: false,
          reasoningPending: false,
          reasoningFinishedAt: last.reasoningStartedAt && !last.reasoningFinishedAt ? Date.now() : last.reasoningFinishedAt,
          content: data.content || last.content || '',
          meta: { ...(last.meta || {}), cancelled: true },
          _provisionalContent: undefined,
        };
        return next;
      }, { visible });
      return;
    }
    if (event === 'done') {
      // Always clean up running state and transition temp keys
      if (sessionKey?.startsWith?.('__new__') && data.session_id) {
        resolveRealSession(data.session_id);
        setRunningBySessionId((prev) => {
          const next = { ...prev };
          delete next[sessionKey];
          next[data.session_id] = { running: true };
          return next;
        });
        sessionRunRef.current[data.session_id] = sessionRunRef.current[sessionKey];
        delete sessionRunRef.current[sessionKey];
        if (activeNewRunKeyRef.current === sessionKey) {
          newSessionIdRef.current = data.session_id;
        }
      }
      // Replace placeholder session in sidebar with real session ID
      if (data.session_id && ctx?.placeholderId) {
        setSessions((prev) =>
          prev.map((s) => (s.id === ctx.placeholderId ? { ...s, id: data.session_id } : s))
        );
        if (newSessionPlaceholderIdRef.current === ctx.placeholderId) {
          newSessionPlaceholderIdRef.current = null;
        }
      }
      // Immediately update session title in sidebar (auto-generated by backend)
      if (data.session_id && data.title) {
        setSessions((prev) =>
          prev.map((s) => (s.id === data.session_id ? { ...s, title: data.title } : s))
        );
      }
      if (visible) {
        if (ctx?.isNewSession && data.session_id && !activeSessionIdRef.current) {
          writeChatRoute(activeAgentId, data.session_id, { mode: 'replace' });
          setActiveSessionId(data.session_id);
        }
      }
      updateSessionMessages(messageKey, (currentMessages) => {
        const { next, last, index } = ensureAssistantTail(currentMessages);
        const finalContent = data.content || last._intermediateContent || last.content || '';
        next[index] = {
          ...last,
          id: data.message_id || last.id,
          run_id: data.run_id || last.run_id,
          pending: false,
          reasoningPending: false,
          reasoningFinishedAt: last.reasoningStartedAt && !last.reasoningFinishedAt ? Date.now() : last.reasoningFinishedAt,
          reasoningDurationMs: data.reasoning_duration_ms ?? last.reasoningDurationMs,
          content: finalContent,
          meta: { ...(last.meta || {}), is_intermediate: false },
          _intermediateContent: undefined,
          _provisionalContent: undefined,
        };
        return next;
      }, { visible });
      if (data.session_id) clearSessionRunning(data.session_id);
      if (sessionKey && sessionKey !== data.session_id) clearSessionRunning(sessionKey);
      if (ctx?.placeholderId) clearSessionRunning(ctx.placeholderId);
      return;
    }
    if (event === 'token') {
      updateSessionMessages(messageKey, (items) => {
        const { next, last, index } = ensureAssistantTail(items);
        next[index] = {
          ...last,
          pending: false,
          reasoningPending: false,
          reasoningFinishedAt: last?.reasoningPending && last?.reasoningStartedAt && !last?.reasoningFinishedAt ? Date.now() : last?.reasoningFinishedAt,
          content: (last.content || '') + (data.content || ''),
        };
        return next;
      }, { visible });
    }
    if (event === 'provisional_token') {
      updateSessionMessages(messageKey, (items) => {
        const { next, last, index } = ensureAssistantTail(items);
        const content = data.content || '';
        next[index] = {
          ...last,
          pending: false,
          reasoningPending: false,
          reasoningFinishedAt: last?.reasoningPending && last?.reasoningStartedAt && !last?.reasoningFinishedAt ? Date.now() : last?.reasoningFinishedAt,
          content: (last.content || '') + content,
          _provisionalContent: (last._provisionalContent || '') + content,
        };
        return next;
      }, { visible });
    }
    if (event === 'provisional_clear') {
      updateSessionMessages(messageKey, (items) => {
        const { next, last, index } = ensureAssistantTail(items);
        const provisionalContent = last._provisionalContent || '';
        next[index] = {
          ...last,
          pending: true,
          content: removeTextSuffix(last.content || '', provisionalContent),
          _provisionalContent: undefined,
        };
        return next;
      }, { visible });
    }
    if (event === 'provisional_commit') {
      updateSessionMessages(messageKey, (items) => {
        const { next, last, index } = ensureAssistantTail(items);
        next[index] = {
          ...last,
          _provisionalContent: undefined,
        };
        return next;
      }, { visible });
    }
    if (event === 'reasoning_token') {
      updateSessionMessages(messageKey, (items) => {
        const { next, last, index } = ensureAssistantTail(items, {
          reasoningVisible: true,
          reasoningPending: true,
          reasoningStartedAt: Date.now(),
        });
        next[index] = appendReasoningTimelineItem({
          ...last,
          reasoningVisible: true,
          reasoningPending: true,
          reasoningStartedAt: last.reasoningStartedAt || Date.now(),
          reasoning: (last.reasoning || '') + (data.content || ''),
        }, data.content || '');
        return next;
      }, { visible });
    }
    if (event === 'sources') {
      const items = data.items || [];
      writeSessionSources(messageKey, items, { visible });
      updateSessionMessages(messageKey, (currentMessages) => {
        const { next, last, index } = ensureAssistantTail(currentMessages);
        next[index] = { ...last, sources: items };
        return next;
      }, { visible });
    }
    if (['rag_status', 'tool_call', 'tool_call_start', 'tool_call_result', 'memory_used', 'thinking_status', 'search_status'].includes(event)) {
      if (visible) {
        setToolDebugEvents((items) => [
          ...items,
          {
            event,
            received_at: new Date().toLocaleTimeString(),
            ...data,
          },
        ].slice(-30));
      }
      if (['tool_call', 'tool_call_start', 'tool_call_result', 'search_status'].includes(event)) {
        updateSessionMessages(messageKey, (items) => {
          const { next, last, index } = ensureAssistantTail(items);
          const shouldKeepReasoningPending = last.reasoningVisible !== false && (last.reasoningPending || (last.pending && !last.content));
          next[index] = appendToolTimelineItem({
            ...last,
            reasoningPending: shouldKeepReasoningPending,
            reasoningStartedAt: shouldKeepReasoningPending ? (last.reasoningStartedAt || Date.now()) : last.reasoningStartedAt,
          }, data, event);
          return next;
        }, { visible });
      }
    }
    if (event === 'error') {
      const detail = errorMessage(data.detail || data.message || '智能体运行失败');
      if (visible) {
        setError(detail);
      }
      updateSessionMessages(messageKey, (items) => {
        const { next, last, index } = ensureAssistantTail(items);
        next[index] = {
          ...last,
          pending: false,
          reasoningPending: false,
          reasoningFinishedAt: last.reasoningStartedAt && !last.reasoningFinishedAt ? Date.now() : last.reasoningFinishedAt,
          error: true,
          content: detail,
        };
        return next;
      }, { visible });
    }
  }

  function addSuggestedQuestion() {
    setAgentForm((form) => ({ ...form, suggested_questions: [...(form.suggested_questions || []), '新的推荐问题'] }));
  }

  function updateSuggestedQuestion(index, value) {
    setAgentForm((form) => ({
      ...form,
      suggested_questions: (form.suggested_questions || []).map((item, itemIndex) => (itemIndex === index ? value : item)),
    }));
  }

  function removeSuggestedQuestion(index) {
    setAgentForm((form) => ({
      ...form,
      suggested_questions: (form.suggested_questions || []).filter((_, itemIndex) => itemIndex !== index),
    }));
  }

  function sendSuggestedQuestion(question) {
    sendMessage(null, question).catch((err) => setError(errorMessage(err)));
  }

  function addVariable() {
    setAgentForm((form) => {
      const key = `var_${(form.variables || []).length + 1}`;
      return {
        ...form,
        variables: [...(form.variables || []), { key, label: '变量', type: 'string', required: false, default_value: '' }],
      };
    });
  }

  function updateVariable(index, patch) {
    setAgentForm((form) => ({
      ...form,
      variables: (form.variables || []).map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)),
    }));
  }

  function removeVariable(index) {
    setAgentForm((form) => ({
      ...form,
      variables: (form.variables || []).filter((_, itemIndex) => itemIndex !== index),
    }));
  }

  function updateChatVariable(key, value) {
    setChatVariables((items) => ({ ...items, [key]: value }));
  }

  function selectChatAgent(agentId, options = {}) {
    const nextAgentId = typeof agentId === 'string' || typeof agentId === 'number' ? Number(agentId) : null;
    if (!nextAgentId) return;
    if (options.updateUrl !== false) {
      writeChatRoute(nextAgentId, null, { mode: options.historyMode || 'push' });
    }
    if (!sameRouteId(nextAgentId, activeAgentIdRef.current)) {
      pendingChatRestoreRef.current = readChatRoute();
    }
    setActiveAgentId(nextAgentId);
  }

  function openBuilder(agentId = activeAgentId) {
    const nextAgentId = typeof agentId === 'string' || typeof agentId === 'number' ? agentId : activeAgentId;
    if (nextAgentId && nextAgentId !== activeAgentId) setActiveAgentId(nextAgentId);
    setView('builder');
    setActiveNav('agents');
  }

  function applyRoleTemplate() {
    setAgentForm((form) => ({
      ...form,
      name: '智能体一号',
      avatar: '66',
      description: '喜欢唱跳 rap 篮球的活泼角色扮演智能体。',
      opening_message: '',
      system_prompt: JIGE_PROMPT,
      suggested_questions: ['你想问啥？'],
      variables: [
        { key: 'city', label: '城市', type: 'string', required: false, default_value: '杭州' },
        { key: 'device_model', label: '设备型号', type: 'string', required: false, default_value: 'S10' },
      ],
      memory: { enabled: true, strategy: 'session_summary', max_messages: 48 },
    }));
  }

  if (!token) {
    return (
      <main className="auth-shell">
        <section className="auth-panel">
          <div className="brand-mark"><Sparkles size={22} /> AgentBase</div>
          <h1>创建你的智能体工作台</h1>
          <p>本地账号、智能体聊天主页、我的模型配置和可发布的智能体工作台。</p>
          <form onSubmit={authenticate} className="auth-form">
            <div className="auth-toggle">
              <button type="button" className={authMode === 'register' ? 'active' : ''} onClick={() => setAuthMode('register')}>注册</button>
              <button type="button" className={authMode === 'login' ? 'active' : ''} onClick={() => setAuthMode('login')}>登录</button>
            </div>
            <input value={authForm.email} onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })} placeholder="邮箱" />
            {authMode === 'register' && <input value={authForm.name} onChange={(e) => setAuthForm({ ...authForm, name: e.target.value })} placeholder="姓名" />}
            <input type="password" value={authForm.password} onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })} placeholder="密码，至少 8 位" />
            <button className="primary" type="submit"><LogIn size={18} />进入工作台</button>
          </form>
          {error && <p className="error">{error}</p>}
        </section>
      </main>
    );
  }

  const shellProps = {
    activeAgent,
    activeAgentId,
    activeNav,
    activeSessionId,
    activeSummary,
    appBootstrapping,
    agentForm,
    agents,
    adminModels,
    busy,
    canManage,
    canEditActive,
    chatMode,
    chatAgents,
    chatAttachments,
    chatVariables,
    clearKnowledgeBaseDocuments,
    copyMarketAgent,
    copyBuiltinPromptTemplate,
    createAgent,
    createKnowledgeBase,
    createSkill,
    updateSkill,
    deleteSkill,
    updateAgentSkills,
    updateKnowledgeBase,
    createModelConfig,
    createPromptTemplate,
    createToolConfig,
    discoverMcpTools,
    createUserModelConfig,
    deleteAgent,
    deleteDocument,
    deleteKnowledgeBase,
    deleteModelConfig,
    deletePromptTemplate,
    deleteSession,
    deleteToolConfig,
    deleteUserModelConfig,
    requestDeleteConfirm,
    docForm,
    documents,
    draft,
    error,
    feedbackByMessage,
    homePrompt,
    knowledgeBases,
    loadSession,
    logout,
    marketAgents,
    me,
    members,
    messages,
    models,
    openBuilder,
    publishAgent,
    promptTemplates,
    memoryProfile,
    memoryProfileDraft,
    memoryProfileError,
    memoryProfileLoading,
    memoryProfileSaving,
    renameSessionById,
    ragRuntime,
    searchEnabled,
    thinkingEnabled,
    saveAgent,
    saveMemoryProfile,
    sendMessage,
    sendSuggestedQuestion,
    sessions,
    selectChatAgent,
    setError,
    setActiveAgentId,
    setActiveNav,
    setAgentForm,
    setChatMode,
    setAccountMenuOpen,
    setDocForm,
    setDraft,
    setHomePrompt,
    setMemoryProfileDraft,
    setRagEnabled,
    setSearchEnabled,
    setThinkingEnabled,
    setView,
    sources,
    startNewChat,
    submittedPromptHistory,
    submitFeedback,
    deleteMemoryProfile,
    approveReview,
    rejectReview,
    reviewItems,
    tools,
    skills,
    toolDebugEvents,
    testToolConfig,
    testUserModelDraft,
    uploadChatAttachment,
    uploadingAttachment,
    uploadingKnowledgeFile,
    uploadingKnowledgeItems,
    uploadDocument,
    uploadKnowledgeFile,
    loadDocuments,
    notify,
    updateChatVariable,
    updateModelConfig,
    updatePromptTemplate,
    updateProfile,
    updateToolConfig,
    updateUserModelConfig,
    view,
    workspace,
    themeMode,
    setThemeMode,
    isDarkTheme,
    accountMenuOpen,
    settingsDialogOpen,
    settingsPanel,
    profileError,
    setProfileError,
    setSettingsDialogOpen,
    setSettingsPanel,
    ragEnabled,
    setChatAttachments,
    testUserModelConfig,
    token,
    userModels,
    webSearchRuntime,
    onStopGeneration: stopGeneration,
    runningBySessionId,
    routeRestoring,
  };

  const builderProps = {
    ...shellProps,
    addSuggestedQuestion,
    addVariable,
    applyRoleTemplate,
    openAgentIdentityDialog,
    removeSuggestedQuestion,
    removeVariable,
    setAgentForm,
    updateSuggestedQuestion,
    updateVariable,
  };

  return (
    <>
      {view === 'builder' ? <BuilderView {...builderProps} /> : <HomeView {...shellProps} />}
      {agentIdentityDialog && (
        <AgentIdentityDialog
          error={agentIdentityError}
          initialForm={agentIdentityDialog.form}
          mode={agentIdentityDialog.mode}
          onCancel={() => {
            if (!agentIdentitySaving) setAgentIdentityDialog(null);
          }}
          onSubmit={submitAgentIdentity}
          saving={agentIdentitySaving}
        />
      )}
      {confirmDialog && (
        <ConfirmDialog
          cancelLabel={confirmDialog.cancelLabel}
          confirmLabel={confirmDialog.confirmLabel}
          detail={confirmDialog.detail}
          message={confirmDialog.message}
          onCancel={() => closeConfirmDialog(false)}
          onConfirm={() => closeConfirmDialog(true)}
          title={confirmDialog.title}
          tone={confirmDialog.tone}
        />
      )}
      {toastMsg && <div className={`toast ${toastTone}`}>{toastMsg}</div>}
    </>
  );
}


function HomeView(props) {
  const [sidebarHidden, setSidebarHidden] = useState(initialSidebarHidden);
  const [moreMenuOpen, setMoreMenuOpen] = useState(false);
  const {
    activeAgent,
    activeAgentId,
    activeNav,
    activeSessionId,
    activeSummary,
    appBootstrapping,
    agentForm,
    agents,
    adminModels,
    busy,
    canManage,
    canEditActive,
    chatMode,
    chatAgents,
    chatAttachments,
    chatVariables,
    clearKnowledgeBaseDocuments,
    copyBuiltinPromptTemplate,
    copyMarketAgent,
    createAgent,
    createKnowledgeBase,
    createSkill,
    updateSkill,
    deleteSkill,
    updateAgentSkills,
    updateKnowledgeBase,
    createModelConfig,
    createPromptTemplate,
    createToolConfig,
    discoverMcpTools,
    createUserModelConfig,
    deleteAgent,
    deleteDocument,
    deleteKnowledgeBase,
    deleteModelConfig,
    deletePromptTemplate,
    deleteSession,
    deleteToolConfig,
    deleteUserModelConfig,
    docForm,
    documents,
    draft,
    error,
    feedbackByMessage,
    homePrompt,
    knowledgeBases,
    loadSession,
    logout,
    marketAgents,
    me,
    members,
    messages,
    onStopGeneration,
    openBuilder,
    promptTemplates,
    requestDeleteConfirm,
    renameSessionById,
    runningBySessionId,
    routeRestoring,
    sendMessage,
    sendSuggestedQuestion,
    sessions,
    selectChatAgent,
    setError,
    setActiveAgentId,
    setActiveNav,
    setAgentForm,
    setAccountMenuOpen,
    setChatMode,
    setChatAttachments,
    setDocForm,
    setHomePrompt,
    setView,
    sources,
    startNewChat,
    submittedPromptHistory,
    submitFeedback,
    approveReview,
    rejectReview,
    reviewItems,
    tools,
    skills,
    testToolConfig,
    uploadChatAttachment,
    uploadingAttachment,
    uploadingKnowledgeFile,
    uploadingKnowledgeItems,
    uploadDocument,
    uploadKnowledgeFile,
    loadDocuments,
    notify,
    updateChatVariable,
    updateModelConfig,
    updatePromptTemplate,
    updateProfile,
    updateToolConfig,
    updateUserModelConfig,
    workspace,
    themeMode,
    setThemeMode,
    isDarkTheme,
    accountMenuOpen,
    settingsDialogOpen,
    settingsPanel,
    profileError,
    ragEnabled,
    ragRuntime,
    searchEnabled,
    setSearchEnabled,
    setRagEnabled,
    setThinkingEnabled,
    setProfileError,
    setSettingsDialogOpen,
    setSettingsPanel,
    testUserModelDraft,
    testUserModelConfig,
    token,
    userModels,
    webSearchRuntime,
    thinkingEnabled,
  } = props;

  useEffect(() => {
    localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, sidebarHidden ? 'true' : 'false');
  }, [sidebarHidden]);

  const [sessionMenuId, setSessionMenuId] = useState(null);
  const [sessionMenuPosition, setSessionMenuPosition] = useState(null);
  const [renamingSessionId, setRenamingSessionId] = useState(null);
  const [sessionRenameDraft, setSessionRenameDraft] = useState('');
  const showWelcome = messages.length === 0;
  const selectedMenuSession = useMemo(
    () => sessions.find((session) => session.id === sessionMenuId),
    [sessionMenuId, sessions],
  );
  const restoringListedSession = appBootstrapping
    || routeRestoring
    || (activeSessionId && !sessions.some((session) => sameRouteId(session.id, activeSessionId)));
  const hideRoutedContent = appBootstrapping && activeNav !== 'chat';

  useEffect(() => {
    if (!sessionMenuId) return undefined;
    const closeMenu = () => {
      setSessionMenuId(null);
      setSessionMenuPosition(null);
    };
    window.addEventListener('click', closeMenu);
    window.addEventListener('resize', closeMenu);
    window.addEventListener('scroll', closeMenu, true);
    return () => {
      window.removeEventListener('click', closeMenu);
      window.removeEventListener('resize', closeMenu);
      window.removeEventListener('scroll', closeMenu, true);
    };
  }, [sessionMenuId]);

  function beginSessionRename(session) {
    setRenamingSessionId(session.id);
    setSessionRenameDraft(session.title || '');
    setSessionMenuId(null);
    setSessionMenuPosition(null);
  }

  function cancelSessionRename() {
    setRenamingSessionId(null);
    setSessionRenameDraft('');
  }

  async function submitSessionRename(event) {
    event?.preventDefault();
    const title = sessionRenameDraft.trim();
    if (!renamingSessionId || !title) return;
    await renameSessionById(renamingSessionId, title);
    cancelSessionRename();
  }

  async function confirmDeleteSession(session) {
    setSessionMenuId(null);
    setSessionMenuPosition(null);
    const confirmed = await requestDeleteConfirm({
      title: '删除会话',
      message: `删除会话「${session.title || '未命名会话'}」？`,
      detail: '该会话中的消息、反馈和调试记录会一起删除。',
      confirmLabel: '删除会话',
    });
    if (!confirmed) return;
    try {
      setError('');
      await deleteSession(session.id);
      if (renamingSessionId === session.id) {
        cancelSessionRename();
      }
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  return (
    <main className={`chat-app ${sidebarHidden ? 'sidebar-hidden' : ''}`}>
      {sidebarHidden && (
        <div className="sidebar-collapsed-controls" aria-label="侧栏快捷操作">
          <span className="collapsed-brand-dot" aria-hidden="true"><Sparkles size={18} /></span>
          <div className="collapsed-action-pill">
            <button
              type="button"
              title="展开侧栏"
              aria-label="展开侧栏"
              onClick={() => setSidebarHidden(false)}
            >
              <ChevronRight size={17} />
            </button>
            <button
              type="button"
              title="新建会话"
              aria-label="新建会话"
              onClick={() => {
                setActiveNav('chat');
                startNewChat();
              }}
            >
              <SquarePen size={17} />
            </button>
          </div>
        </div>
      )}
      <aside className="chat-sidebar">
        <div className="sidebar-brand">
          <span className="brand-dot"><Sparkles size={18} /></span>
          <strong>AgentBase</strong>
          <button
            className="sidebar-toggle"
            type="button"
            title="隐藏侧栏"
            aria-label="隐藏侧栏"
            onClick={() => {
              setSidebarHidden(true);
              setAccountMenuOpen(false);
              setSessionMenuId(null);
              setSessionMenuPosition(null);
            }}
          >
            <ChevronLeft size={16} />
          </button>
        </div>
        <nav className="main-nav">
          <NavButton icon={<SquarePen size={17} />} label="新建会话" active={activeNav === 'chat' && !activeSessionId && !routeRestoring} onClick={() => { setActiveNav('chat'); startNewChat(); }} />
          <NavButton icon={<Bot size={17} />} label="智能体" active={activeNav === 'agents'} onClick={() => setActiveNav('agents')} />
          <NavButton icon={<Boxes size={17} />} label="市场" active={activeNav === 'market'} onClick={() => setActiveNav('market')} />
          <NavButton icon={<ServerCog size={17} />} label="我的模型" active={activeNav === 'my-models'} onClick={() => setActiveNav('my-models')} />
          <NavButton icon={<Wand2 size={17} />} label="工具" active={activeNav === 'tools'} onClick={() => setActiveNav('tools')} />
          <NavButton icon={<Sparkles size={17} />} label="技能" active={activeNav === 'skills'} onClick={() => setActiveNav('skills')} />
          <div className="more-menu-container">
            <button
              type="button"
              className={`more-menu-trigger ${['resources', 'knowledge', 'reviews', 'members'].includes(activeNav) ? 'active' : ''}`}
              onClick={() => setMoreMenuOpen(!moreMenuOpen)}
            >
              <MoreHorizontal size={17} />
              更多
            </button>
            {moreMenuOpen && (
              <>
                <div className="more-menu-backdrop" onClick={() => setMoreMenuOpen(false)} />
                <div className="more-menu-dropdown">
                  <button type="button" className={activeNav === 'resources' ? 'active' : ''} onClick={() => { setActiveNav('resources'); setMoreMenuOpen(false); }}>
                    <Layers size={16} />资源库
                  </button>
                  <button type="button" className={activeNav === 'knowledge' ? 'active' : ''} onClick={() => { setActiveNav('knowledge'); setMoreMenuOpen(false); }}>
                    <Database size={16} />知识库
                  </button>
                  {canManage && (
                    <button type="button" className={activeNav === 'reviews' ? 'active' : ''} onClick={() => { setActiveNav('reviews'); setMoreMenuOpen(false); }}>
                      <Shield size={16} />审核
                    </button>
                  )}
                  {canManage && (
                    <button type="button" className={activeNav === 'members' ? 'active' : ''} onClick={() => { setActiveNav('members'); setMoreMenuOpen(false); }}>
                      <KeyRound size={16} />成员
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </nav>
        <div className="sidebar-section">
          <div className="sidebar-heading">
            <span>会话</span>
          </div>
          <div className="session-list">
            {!restoringListedSession && sessions.map((session) => {
              const isActiveSession = String(session.id) === String(activeSessionId || '');
              const isSessionRunning = !!runningBySessionId[session.id]?.running;
              return (
                <div key={session.id} className={`session-row ${isActiveSession ? 'active' : ''}`}>
                  {renamingSessionId === session.id ? (
                    <form className="session-rename-form" onSubmit={(event) => submitSessionRename(event).catch((err) => console.error(err))}>
                      <input
                        autoFocus
                        value={sessionRenameDraft}
                        onChange={(event) => setSessionRenameDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === 'Escape') {
                            event.preventDefault();
                            cancelSessionRename();
                          }
                        }}
                        placeholder="会话标题"
                      />
                      <button type="submit" title="保存" aria-label="保存会话标题" disabled={!sessionRenameDraft.trim()}><Check size={14} /></button>
                      <button type="button" title="取消" aria-label="取消重命名" onClick={cancelSessionRename}><X size={14} /></button>
                    </form>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="session-main"
                        aria-current={isActiveSession ? 'page' : undefined}
                        onClick={() => {
                          setSessionMenuId(null);
                          setSessionMenuPosition(null);
                          loadSession(session.id, { openHome: true, historyMode: 'push' }).catch((err) => console.error(err));
                        }}
                      >
                        {isSessionRunning ? (
                          <span className="session-running-dot" title="运行中" aria-label="运行中" />
                        ) : (
                          <MessageSquare size={14} />
                        )}
                        <span>{session.title}</span>
                      </button>
                      <button
                        type="button"
                        className="session-more"
                        title="会话操作"
                        aria-label="会话操作"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (sessionMenuId === session.id) {
                            setSessionMenuId(null);
                            setSessionMenuPosition(null);
                            return;
                          }
                          const rect = event.currentTarget.getBoundingClientRect();
                          const sidebarRect = event.currentTarget.closest('.chat-sidebar')?.getBoundingClientRect();
                          setSessionMenuPosition({
                            left: (sidebarRect?.right ?? rect.right) + 8,
                            top: Math.max(8, Math.min(rect.top - 2, window.innerHeight - 104)),
                          });
                          setSessionMenuId(session.id);
                        }}
                      >
                        <MoreHorizontal size={15} />
                      </button>
                    </>
                  )}
                </div>
              );
            })}
            {!restoringListedSession && sessions.length === 0 && <p className="sidebar-empty">还没有历史会话</p>}
          </div>
        </div>
        <div className="sidebar-user-wrap">
          {!appBootstrapping && (
            <>
              {accountMenuOpen && (
                <div className="account-menu">
                  <div className="account-menu-group">
                    <button
                      type="button"
                      onClick={() => {
                        setSettingsPanel('general');
                        setSettingsDialogOpen(true);
                        setAccountMenuOpen(false);
                      }}
                    >
                      <Settings2 size={16} />
                      系统设置
                    </button>
                  </div>
                  <button
                    className="account-menu-logout"
                    type="button"
                    onClick={logout}
                  >
                    <LogOut size={16} />
                    退出登录
                  </button>
                </div>
              )}
              <button
                className={`sidebar-user ${accountMenuOpen ? 'is-open' : ''}`}
                type="button"
                onClick={() => setAccountMenuOpen(!accountMenuOpen)}
              >
                <UserAvatar user={me} className="account-avatar small" />
                <span className="sidebar-user-copy">
                  <strong>{me?.name || me?.email || '当前用户'}</strong>
                  <small>{roleLabel(workspace?.role)}</small>
                </span>
                <MoreHorizontal size={18} />
              </button>
            </>
          )}
        </div>
      </aside>

      {selectedMenuSession && sessionMenuPosition && (
        <div
          className="session-menu session-menu-floating"
          style={{ left: `${sessionMenuPosition.left}px`, top: `${sessionMenuPosition.top}px` }}
          onClick={(event) => event.stopPropagation()}
        >
          <button type="button" onClick={() => beginSessionRename(selectedMenuSession)}>
            <SquarePen size={14} />
            重命名
          </button>
          <button type="button" className="danger" onClick={() => confirmDeleteSession(selectedMenuSession).catch((err) => console.error(err))}>
            <Trash2 size={14} />
            删除
          </button>
        </div>
      )}

      <section className={`chat-main ${activeNav === 'chat' ? '' : 'no-topbar'}`}>
         {activeNav === 'chat' && (
          <ChatView
            activeAgent={activeAgent}
            activeAgentId={activeAgentId}
            activeSummary={activeSummary}
            chatAgents={chatAgents}
            canEditActive={canEditActive}
            openBuilder={openBuilder}
            setActiveAgentId={selectChatAgent}
            activeSessionId={activeSessionId}
            agentForm={agentForm}
            busy={busy}
            chatAttachments={chatAttachments}
            chatVariables={chatVariables}
            error={error}
            feedbackByMessage={feedbackByMessage}
            homePrompt={homePrompt}
            messages={messages}
            submittedPromptHistory={submittedPromptHistory}
            sendMessage={sendMessage}
            sendSuggestedQuestion={sendSuggestedQuestion}
            setChatAttachments={setChatAttachments}
            setHomePrompt={setHomePrompt}
            ragEnabled={ragEnabled}
            setRagEnabled={setRagEnabled}
            searchEnabled={searchEnabled}
            setSearchEnabled={setSearchEnabled}
            sources={sources}
            submitFeedback={submitFeedback}
            ragRuntime={ragRuntime}
            webSearchRuntime={webSearchRuntime}
            thinkingEnabled={thinkingEnabled}
            setThinkingEnabled={setThinkingEnabled}
            uploadChatAttachment={uploadChatAttachment}
            uploadingAttachment={uploadingAttachment}
            updateChatVariable={updateChatVariable}
            onStopGeneration={onStopGeneration}
            routeRestoring={routeRestoring}
          />
        )}

        {!hideRoutedContent && (
          <>
            {activeNav === 'members' && canManage && (
              <MembersHome members={members} />
            )}

            {activeNav === 'knowledge' && (
              <KnowledgeHome
                canManage={canManage}
                clearKnowledgeBaseDocuments={clearKnowledgeBaseDocuments}
                createKnowledgeBase={createKnowledgeBase}
                updateKnowledgeBase={updateKnowledgeBase}
                deleteDocument={deleteDocument}
                deleteKnowledgeBase={deleteKnowledgeBase}
                docForm={docForm}
                documents={documents}
                knowledgeBases={knowledgeBases}
                setDocForm={setDocForm}
                setProfileError={setProfileError}
                token={token}
                uploadingKnowledgeFile={uploadingKnowledgeFile}
                uploadingKnowledgeItems={uploadingKnowledgeItems}
                uploadDocument={uploadDocument}
                uploadKnowledgeFile={uploadKnowledgeFile}
                loadDocuments={loadDocuments}
                notify={notify}
              />
            )}

            {activeNav === 'agents' && (
              <AgentsHome
                agents={agents}
                activeAgentId={activeAgentId}
                canManage={canManage}
                createAgent={createAgent}
                deleteAgent={deleteAgent}
                me={me}
                openBuilder={openBuilder}
                setActiveAgentId={(agentId) => selectChatAgent(agentId, { historyMode: 'replace' })}
              />
            )}

            {activeNav === 'market' && (
              <MarketHome
                agents={marketAgents}
                copyMarketAgent={copyMarketAgent}
              />
            )}

        {activeNav === 'my-models' && (
          <UserModelsHome
            adminModels={adminModels}
            canManage={canManage}
            createModelConfig={createModelConfig}
            deleteModelConfig={deleteModelConfig}
            requestDeleteConfirm={requestDeleteConfirm}
            setProfileError={setProfileError}
            updateModelConfig={updateModelConfig}
            userModels={userModels}
            createUserModelConfig={createUserModelConfig}
            updateUserModelConfig={updateUserModelConfig}
            deleteUserModelConfig={deleteUserModelConfig}
            testUserModelConfig={testUserModelConfig}
            testUserModelDraft={testUserModelDraft}
          />
        )}

        {activeNav === 'resources' && (
          <ResourceLibraryHome
            activeAgentId={activeAgentId}
            agentForm={agentForm}
            copyBuiltinPromptTemplate={copyBuiltinPromptTemplate}
            createPromptTemplate={createPromptTemplate}
            deletePromptTemplate={deletePromptTemplate}
            knowledgeBases={knowledgeBases}
            openBuilder={openBuilder}
            promptTemplates={promptTemplates}
            requestDeleteConfirm={requestDeleteConfirm}
            setActiveNav={setActiveNav}
            setAgentForm={setAgentForm}
            setProfileError={setProfileError}
            setView={setView}
            tools={tools}
            updatePromptTemplate={updatePromptTemplate}
          />
        )}

        {activeNav === 'tools' && (
          <ToolsHome
            createToolConfig={createToolConfig}
            discoverMcpTools={discoverMcpTools}
            deleteToolConfig={deleteToolConfig}
            isDarkTheme={isDarkTheme}
            notify={notify}
            openBuilder={openBuilder}
            requestDeleteConfirm={requestDeleteConfirm}
            setProfileError={setProfileError}
            testToolConfig={testToolConfig}
            tools={tools}
            updateToolConfig={updateToolConfig}
          />
        )}

        {activeNav === 'skills' && (
          <SkillsHome
            createSkill={createSkill}
            deleteSkill={deleteSkill}
            knowledgeBases={knowledgeBases}
            notify={notify}
            setProfileError={setProfileError}
            skills={skills}
            token={token}
            tools={tools}
            updateSkill={updateSkill}
          />
        )}

        {activeNav === 'reviews' && canManage && (
          <ReviewHome
            items={reviewItems}
            approveReview={approveReview}
            rejectReview={rejectReview}
          />
        )}
          </>
        )}


      </section>
      {settingsDialogOpen && (
        <SystemSettingsDialog
          activePanel={settingsPanel}
          me={me}
          onClose={() => setSettingsDialogOpen(false)}
          onPanelChange={setSettingsPanel}
          profileError={profileError}
          setProfileError={setProfileError}
          setThemeMode={setThemeMode}
          themeMode={themeMode}
          updateProfile={updateProfile}
          workspace={workspace}
        />
      )}
    </main>
  );
}

const CHAT_COPY = {
  noAgentTitle: '\u6682\u65e0\u53ef\u5bf9\u8bdd\u7684\u667a\u80fd\u4f53',
  noAgentDesc: '\u4e3b\u5bf9\u8bdd\u9875\u53ea\u5f00\u653e\u5df2\u5ba1\u6838\u5e76\u4e0a\u67b6\u7684\u667a\u80fd\u4f53\u3002\u8bf7\u5148\u53d1\u5e03\uff0c\u666e\u901a\u7528\u6237\u53d1\u5e03\u540e\u9700\u8981\u7ba1\u7406\u5458\u5ba1\u6838\u3002',
  welcomeTitle: '\u4eca\u5929\u60f3\u8ba9\u54ea\u4e2a\u667a\u80fd\u4f53\u5e2e\u4f60\uff1f',
  welcomeDesc: '\u9009\u62e9\u667a\u80fd\u4f53\u540e\u53ef\u4ee5\u76f4\u63a5\u804a\u5929\uff0c\u4e5f\u53ef\u4ee5\u8fdb\u5165\u914d\u7f6e\u9875\u8c03\u6574\u80fd\u529b\u3002',
  promptIntro: '\u4ecb\u7ecd\u4e00\u4e0b\u4f60\u7684\u80fd\u529b',
  promptPlan: '\u5e2e\u6211\u6574\u7406\u4e00\u4e2a\u65b9\u6848',
  promptKb: '\u57fa\u4e8e\u77e5\u8bc6\u5e93\u56de\u7b54\u4e00\u4e2a\u95ee\u9898',
  fallbackAgent: '\u667a\u80fd\u4f53',
  sendPrefix: '\u7ed9',
  sendSuffix: '\u53d1\u9001\u6d88\u606f',
  sendMessage: '\u53d1\u9001\u6d88\u606f...',
  uploading: '\u9644\u4ef6\u4e0a\u4f20\u4e2d...',
  pendingAttachment: '\u4e2a\u9644\u4ef6\u5f85\u53d1\u9001',
  newChat: '\u65b0\u5efa\u4f1a\u8bdd',
  thinking: '\u6df1\u5ea6\u601d\u8003',
  search: '\u8054\u7f51\u641c\u7d22',
  rag: '\u77e5\u8bc6\u5e93',
  unavailable: '\u4e0d\u53ef\u7528',
  on: '\u5f00\u542f',
  off: '\u5173\u95ed',
};





function AgentsHome({ agents, activeAgentId, canManage, createAgent, deleteAgent, me, openBuilder, setActiveAgentId }) {
  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>智能体</h1>
          <p>创建、选择和编辑你的智能体。普通用户提交发布后需要管理员审核。</p>
        </div>
        <button className="primary" type="button" onClick={() => createAgent(true)}><Plus size={16} />创建智能体</button>
      </header>
      <div className="agent-grid">
        {agents.map((agent) => {
          const isCurrent = agent.id === activeAgentId;
          return (
            <article className={`agent-card ${isCurrent ? 'active' : ''}`} key={agent.id}>
              <AgentAvatar value={agent.avatar} />
              <h3>{agent.name}</h3>
              <p>{agent.description || '暂无简介'}</p>
              <small className={`status-pill ${agent.status}`}>{statusLabel(agent.status)}</small>
              <div>
                <button
                  className={isCurrent ? 'agent-current-button' : ''}
                  type="button"
                  disabled={isCurrent}
                  onClick={() => setActiveAgentId(agent.id)}
                >
                  {isCurrent ? '已设置' : '设为当前'}
                </button>
                <button type="button" disabled={!canManage && agent.created_by !== me?.id} onClick={() => openBuilder(agent.id)}>编辑</button>
                {!agent.is_template && (
                  <button
                    className="danger-light"
                    type="button"
                    disabled={!canManage && agent.created_by !== me?.id}
                    onClick={() => deleteAgent(agent).catch((err) => console.error(err))}
                  >
                    删除
                  </button>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function MarketHome({ agents, copyMarketAgent }) {
  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>智能体市场</h1>
          <p>审核通过的智能体会显示在这里，其他用户可以复制成自己的草稿后继续配置。</p>
        </div>
      </header>
      <div className="agent-grid">
        {agents.map((agent) => (
          <article className="agent-card" key={agent.id}>
            <AgentAvatar value={agent.avatar} />
            <h3>{agent.name}</h3>
            <p>{agent.description || '暂无简介'}</p>
            <small className="status-pill published">版本 {agent.version || '-'}</small>
            <div>
              <button type="button" onClick={() => copyMarketAgent(agent.id).catch((err) => console.error(err))}>复制使用</button>
            </div>
          </article>
        ))}
        {agents.length === 0 && <p className="empty-state">市场里还没有审核通过的智能体。</p>}
      </div>
    </div>
  );
}

function ReviewHome({ approveReview, items, rejectReview }) {
  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>发布审核</h1>
          <p>普通用户提交发布后，管理员在这里审核；通过后会进入市场。</p>
        </div>
      </header>
      <div className="review-list">
        {items.map((agent) => (
          <article className="review-card" key={agent.id}>
            <AgentAvatar value={agent.avatar} className="agent-avatar" />
            <div>
              <h3>{agent.name}</h3>
              <p>{agent.description || '暂无简介'}</p>
              <small>提交版本 {agent.submitted_version || '-'} · {agent.submitted_at || '刚刚'}</small>
            </div>
            <div className="review-actions">
              <button type="button" onClick={() => rejectReview(agent.id).catch((err) => console.error(err))}>驳回</button>
              <button className="primary" type="button" onClick={() => approveReview(agent.id).catch((err) => console.error(err))}>通过</button>
            </div>
          </article>
        ))}
        {items.length === 0 && <p className="empty-state">暂无待审核智能体。</p>}
      </div>
    </div>
  );
}

function MembersHome({ members }) {
  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>成员</h1>
          <p>管理员只查看成员列表。最终版不提供邀请用户和邀请列表页面。</p>
        </div>
      </header>
      <div className="member-list">
        {members.map((member) => (
          <article className="member-card" key={member.id}>
            <UserAvatar user={member.user} className="account-avatar" />
            <div>
              <h3>{member.user?.name || member.user?.email}</h3>
              <p>{member.user?.email}</p>
            </div>
            <span className="status-pill">{roleLabel(member.role)}</span>
          </article>
        ))}
        {members.length === 0 && <p className="empty-state">暂无成员。</p>}
      </div>
    </div>
  );
}

function AgentIdentityDialog({ error, initialForm, mode, onCancel, onSubmit, saving }) {
  const [form, setForm] = useState(() => normalizeAgentIdentity(initialForm));
  const title = mode === 'create' ? '创建智能体' : '编辑智能体';

  useEffect(() => {
    setForm(normalizeAgentIdentity(initialForm));
  }, [initialForm]);

  async function uploadAgentAvatar(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    try {
      validateAvatarFile(file);
      const avatar = await createAvatarDataUrl(file);
      setForm((current) => ({ ...current, avatar }));
    } catch (err) {
      setForm((current) => ({ ...current, localError: errorMessage(err) }));
    }
  }

  function submit(event) {
    event.preventDefault();
    onSubmit(form);
  }

  return (
    <div className="modal-backdrop" role="presentation">
      <form className="agent-identity-modal" onSubmit={submit}>
        <header>
          <h2>{title}</h2>
          <button type="button" aria-label="关闭" onClick={onCancel} disabled={saving}><X size={18} /></button>
        </header>
        <label className="field-stack">
          <span>智能体名称<b>*</b></span>
          <input
            value={form.name}
            maxLength={50}
            onChange={(event) => setForm({ ...form, name: event.target.value })}
            placeholder="请输入智能体名称"
            autoFocus
          />
          <em>{form.name.length}/50</em>
        </label>
        <label className="field-stack">
          <span>智能体功能介绍</span>
          <textarea
            value={form.description}
            maxLength={500}
            onChange={(event) => setForm({ ...form, description: event.target.value })}
            placeholder="介绍智能体的功能，将会展示给智能体的用户"
          />
          <em>{form.description.length}/500</em>
        </label>
        <div className="agent-avatar-picker">
          <span>图标 <b>*</b></span>
          <div>
            <AgentAvatar value={form.avatar} className="agent-avatar-preview" />
            <label className="agent-avatar-upload" title="上传图标">
              <ImagePlus size={16} />
              <input type="file" accept="image/png,image/jpeg,image/webp,image/gif" onChange={uploadAgentAvatar} disabled={saving} />
            </label>
          </div>
        </div>
        {(error || form.localError) && <p className="error">{error || form.localError}</p>}
        <footer>
          <button type="button" onClick={onCancel} disabled={saving}>取消</button>
          <button className="primary" type="submit" disabled={saving || !form.name.trim()}>
            {saving ? '保存中...' : '确认'}
          </button>
        </footer>
      </form>
    </div>
  );
}

function ConfirmDialog({ cancelLabel = '取消', confirmLabel = '删除', detail = '', message, onCancel, onConfirm, title, tone = 'danger' }) {
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onCancel();
      if (event.key === 'Enter') onConfirm();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onCancel, onConfirm]);

  return (
    <div className="confirm-dialog-backdrop" role="presentation">
      <section className={`confirm-dialog ${tone}`} role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title" onClick={(event) => event.stopPropagation()}>
        <header>
          <span className="confirm-dialog-icon"><Trash2 size={18} /></span>
          <div>
            <h2 id="confirm-dialog-title">{title || '确认删除'}</h2>
            <p>{message}</p>
          </div>
        </header>
        {detail && <p className="confirm-dialog-detail">{detail}</p>}
        <footer>
          <button type="button" onClick={onCancel}>{cancelLabel}</button>
          <button className="danger" type="button" autoFocus onClick={onConfirm}>{confirmLabel}</button>
        </footer>
      </section>
    </div>
  );
}





function SecretInputDialog({ label = '密钥', message, onCancel, onSubmit, placeholder = '只提交一次，不会回显', saving, submitLabel = '保存', title }) {
  const [value, setValue] = useState('');

  function submit(event) {
    event.preventDefault();
    onSubmit(value);
  }

  return (
    <div className="profile-dialog-backdrop">
      <section className="resource-form-dialog secret-input-dialog" role="dialog" aria-modal="true" aria-label={title} onClick={(event) => event.stopPropagation()}>
        <button className="profile-dialog-close" type="button" title="关闭" aria-label="关闭密钥表单" onClick={onCancel} disabled={saving}>
          <X size={16} />
        </button>
        <header className="model-dialog-heading">
          <h3>{title}</h3>
          <p>{message}</p>
        </header>
        <form className="dialog-form" onSubmit={submit}>
          <label className="field-stack">
            <span>{label}</span>
            <input type="password" value={value} onChange={(event) => setValue(event.target.value)} placeholder={placeholder} autoComplete="off" autoFocus />
          </label>
          <footer className="dialog-actions">
            <button type="button" onClick={onCancel} disabled={saving}>取消</button>
            <button className="primary-model-action" type="submit" disabled={saving || !value.trim()}>
              <Check size={15} />{saving ? '保存中...' : submitLabel}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}

function KnowledgeHome({
  canManage,
  clearKnowledgeBaseDocuments,
  createKnowledgeBase,
  updateKnowledgeBase,
  deleteDocument,
  deleteKnowledgeBase,
  docForm,
  documents,
  knowledgeBases,
  setDocForm,
  setProfileError,
  uploadingKnowledgeFile,
  uploadingKnowledgeItems,
  uploadDocument,
  uploadKnowledgeFile,
  token,
  loadDocuments,
  notify,
}) {
  const [createOpen, setCreateOpen] = useState(false);
  const [form, setForm] = useState(() => defaultKnowledgeBaseForm());
  const [saving, setSaving] = useState(false);

  // Task 4 States
  const [viewMode, setViewMode] = useState('list'); // 'list' | 'detail'
  const [activeKbId, setActiveKbId] = useState(null); // number
  const [activeDoc, setActiveDoc] = useState(null); // object
  const [resegmentOpen, setResegmentOpen] = useState(false); // boolean

  // Synchronize state when entering detail view
  function handleSelectKb(kbId) {
    setActiveKbId(kbId);
    setDocForm((current) => ({ ...current, kb_id: String(kbId) }));
    setViewMode('detail');
  }

  function handleBack() {
    setViewMode('list');
    setActiveDoc(null);
  }

  function openCreate() {
    setForm(defaultKnowledgeBaseForm());
    setCreateOpen(true);
  }

  function closeCreate() {
    if (saving) return;
    setCreateOpen(false);
  }

  async function submitKnowledgeBase(event) {
    event.preventDefault();
    setSaving(true);
    setProfileError('');
    try {
      const saved = await createKnowledgeBase(form);
      if (saved?.id) {
        setDocForm((current) => ({ ...current, kb_id: String(saved.id) }));
      }
      setForm(defaultKnowledgeBaseForm());
      setCreateOpen(false);
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  const selectedKb = useMemo(() => {
    return knowledgeBases.find((kb) => kb.id === activeKbId) || null;
  }, [knowledgeBases, activeKbId]);

  return (
    <div className="content-page knowledge-home-page">
      {viewMode === 'list' ? (
        <KnowledgeDashboard
          knowledgeBases={knowledgeBases}
          deleteKnowledgeBase={deleteKnowledgeBase}
          openCreate={openCreate}
          onSelectKb={handleSelectKb}
          notify={notify}
        />
      ) : (
        <KnowledgeWorkspace
          kb={selectedKb}
          clearKnowledgeBaseDocuments={clearKnowledgeBaseDocuments}
          documents={documents}
          deleteDocument={deleteDocument}
          updateKnowledgeBase={updateKnowledgeBase}
          uploadDocument={uploadDocument}
          uploadKnowledgeFile={uploadKnowledgeFile}
          uploadingKnowledgeFile={uploadingKnowledgeFile}
          uploadingKnowledgeItems={uploadingKnowledgeItems}
          docForm={docForm}
          setDocForm={setDocForm}
          handleBack={handleBack}
          activeDoc={activeDoc}
          setActiveDoc={setActiveDoc}
          setResegmentOpen={setResegmentOpen}
          token={token}
        />
      )}

      {createOpen && (
        <KnowledgeBaseDialog
          form={form}
          onCancel={closeCreate}
          onChange={setForm}
          onSubmit={submitKnowledgeBase}
          saving={saving}
        />
      )}

      {resegmentOpen && activeDoc && (
        <ResegmentModal
          isOpen={resegmentOpen}
          onClose={() => setResegmentOpen(false)}
          kbId={activeKbId}
          doc={activeDoc}
          token={token}
          onResegmentSuccess={async () => {
            setResegmentOpen(false);
            if (activeKbId && loadDocuments) {
              await loadDocuments(activeKbId);
            }
          }}
          notify={notify}
        />
      )}
    </div>
  );
}

function KnowledgeDashboard({
  knowledgeBases,
  deleteKnowledgeBase,
  openCreate,
  onSelectKb,
  notify,
}) {
  const [openMenuKbId, setOpenMenuKbId] = useState(null);
  const [kbEnabled, setKbEnabled] = useState({});

  useEffect(() => {
    function handleOutsideClick() {
      setOpenMenuKbId(null);
    }
    if (openMenuKbId) {
      window.addEventListener('click', handleOutsideClick);
    }
    return () => window.removeEventListener('click', handleOutsideClick);
  }, [openMenuKbId]);

  return (
    <div className="knowledge-dashboard">
      <header className="dashboard-header">
        <div>
          <h2>知识库列表</h2>
          <p>选择一个知识库进行精准分段调参、上传及管理文档。</p>
        </div>
        <button className="primary" type="button" onClick={openCreate}>
          <Plus size={16} />新建知识库
        </button>
      </header>

      <div className="dashboard-table-container">
        <table className="dashboard-table">
          <thead>
            <tr>
              <th>信息</th>
              <th>类型</th>
              <th>更新时间</th>
              <th>启用状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {knowledgeBases.map((kb) => {
              const isEnabled = kbEnabled[kb.id] !== false;
              const formattedDate = kb.updated_at
                ? new Date(kb.updated_at).toLocaleString('zh-CN', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : '2026-05-30';

              return (
                <tr key={kb.id} className="kb-row-interactive" onClick={() => onSelectKb(kb.id)}>
                  <td className="kb-info-cell">
                    <div className="kb-icon-wrapper">
                      <Database size={20} />
                    </div>
                    <div className="kb-meta-details">
                      <strong>{kb.name}</strong>
                      <small>{kb.description || '暂无描述'}</small>
                      <span className="kb-doc-count-pill">{kb.document_count || 0} 个文档</span>
                    </div>
                  </td>
                  <td>
                    <span className="kb-type-badge">文档库</span>
                  </td>
                  <td className="kb-time-cell">{formattedDate}</td>
                  <td>
                    <label className="apple-switch" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={isEnabled}
                        onChange={() => {
                          const nextVal = !isEnabled;
                          setKbEnabled((prev) => ({ ...prev, [kb.id]: nextVal }));
                          notify?.(nextVal ? `「${kb.name}」已启用` : `「${kb.name}」已停用`);
                        }}
                      />
                      <span className="slider"></span>
                    </label>
                  </td>
                  <td className="kb-actions-cell">
                    <div className="actions-wrapper">
                      <button
                        className="btn-more"
                        title="操作"
                        onClick={(e) => {
                          e.stopPropagation();
                          setOpenMenuKbId(openMenuKbId === kb.id ? null : kb.id);
                        }}
                      >
                        <MoreHorizontal size={16} />
                      </button>
                      {openMenuKbId === kb.id && (
                        <div className="bubble-dropdown-menu" onClick={(e) => e.stopPropagation()}>
                          <button
                            type="button"
                            onClick={() => {
                              setOpenMenuKbId(null);
                              notify?.("复制成功！已复制到其他空间。");
                            }}
                          >
                            复制到其他空间
                          </button>
                          <button
                            type="button"
                            className="danger"
                            onClick={() => {
                              setOpenMenuKbId(null);
                              deleteKnowledgeBase(kb);
                            }}
                          >
                            删除
                          </button>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
            {knowledgeBases.length === 0 && (
              <tr>
                <td colSpan="5" className="empty-table-cell">
                  <div className="table-empty-state">
                    <Database size={40} className="muted-icon" />
                    <p>还没有知识库，点击右上角新建知识库。</p>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function KnowledgeWorkspace({
  kb,
  clearKnowledgeBaseDocuments,
  documents,
  deleteDocument,
  updateKnowledgeBase,
  uploadDocument,
  uploadKnowledgeFile,
  uploadingKnowledgeFile,
  uploadingKnowledgeItems = [],
  docForm,
  setDocForm,
  handleBack,
  activeDoc,
  setActiveDoc,
  setResegmentOpen,
  token,
}) {
  const [docSearchQuery, setDocSearchQuery] = useState('');
  const [addDropdownOpen, setAddDropdownOpen] = useState(false);
  const [customInputOpen, setCustomInputOpen] = useState(false);

  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editForm, setEditForm] = useState({ name: '', description: '' });
  const [savingEdit, setSavingEdit] = useState(false);

  async function handleEditSubmit(e) {
    if (e) e.preventDefault();
    if (!editForm.name.trim()) return;
    setSavingEdit(true);
    try {
      await updateKnowledgeBase(kb.id, {
        name: editForm.name.trim(),
        description: editForm.description.trim()
      });
      setEditDialogOpen(false);
    } catch (err) {
      console.error(err);
    } finally {
      setSavingEdit(false);
    }
  }

  const [chunks, setChunks] = useState([]);
  const [chunksLoading, setChunksLoading] = useState(false);

  const fileInputRef = useRef(null);

  // Close dropdown on click outside
  useEffect(() => {
    function handleOutsideClick() {
      setAddDropdownOpen(false);
    }
    if (addDropdownOpen) {
      window.addEventListener('click', handleOutsideClick);
    }
    return () => window.removeEventListener('click', handleOutsideClick);
  }, [addDropdownOpen]);

  // Sync / fetch chunks for the selected active document
  useEffect(() => {
    if (!activeDoc || !kb?.id) {
      setChunks([]);
      return;
    }
    let active = true;
    async function fetchChunks() {
      setChunksLoading(true);
      try {
        const data = await api(`/api/knowledge-bases/${kb.id}/documents/${activeDoc.id}/chunks`, { token });
        if (active) {
          setChunks(data.chunks || []);
        }
      } catch (err) {
        console.error(err);
        if (active) setChunks([]);
      } finally {
        if (active) setChunksLoading(false);
      }
    }
    fetchChunks();
    return () => {
      active = false;
    };
  }, [activeDoc, kb?.id, token]);

  const filteredDocs = useMemo(() => {
    if (!docSearchQuery.trim()) return documents;
    return documents.filter((doc) =>
      (doc.title || doc.filename || '').toLowerCase().includes(docSearchQuery.toLowerCase())
    );
  }, [documents, docSearchQuery]);
  const visibleUploadingItems = useMemo(() => {
    return uploadingKnowledgeItems.filter((item) => !item.kbId || item.kbId === kb?.id);
  }, [uploadingKnowledgeItems, kb?.id]);

  // Autoselect first document if none selected
  useEffect(() => {
    if (filteredDocs.length > 0 && !activeDoc) {
      setActiveDoc(filteredDocs[0]);
    }
  }, [filteredDocs, activeDoc, setActiveDoc]);

  function triggerLocalFileInput() {
    fileInputRef.current?.click();
  }

  // Handle custom paste submit
  const [pasteFilename, setPasteFilename] = useState('粘贴文档.txt');
  const [pasteText, setPasteText] = useState('');
  const [pasteSubmitting, setPasteSubmitting] = useState(false);
  const [deletingDocId, setDeletingDocId] = useState(null);
  const [clearingDocuments, setClearingDocuments] = useState(false);

  async function handlePasteSubmit(e) {
    e.preventDefault();
    if (!pasteText.trim()) return;
    setPasteSubmitting(true);
    try {
      const payload = {
        title: pasteFilename || '粘贴文档.txt',
        filename: pasteFilename || '粘贴文档.txt',
        content: pasteText,
        content_type: 'text/plain',
        source_type: 'text',
      };
      await api(`/api/knowledge-bases/${kb.id}/documents`, {
        token,
        method: 'POST',
        body: payload,
      });
      // Clear inputs
      setPasteText('');
      setCustomInputOpen(false);
      // Wait a moment and force reload
      if (setDocForm) {
        setDocForm((form) => ({ ...form, kb_id: String(kb.id) })); // triggers re-fetch in parent!
      }
    } catch (err) {
      console.error(err);
    } finally {
      setPasteSubmitting(false);
    }
  }

  async function handleDeleteActiveDoc() {
    if (!activeDoc?.id || deletingDocId) return;
    setDeletingDocId(activeDoc.id);
    try {
      const deleted = await deleteDocument(activeDoc.id, kb?.id);
      if (deleted) {
        setActiveDoc(null);
        setChunks([]);
      }
    } finally {
      setDeletingDocId(null);
    }
  }

  async function handleClearDocuments() {
    if (!kb?.id || clearingDocuments) return;
    setClearingDocuments(true);
    try {
      const cleared = await clearKnowledgeBaseDocuments(kb);
      if (cleared) {
        setActiveDoc(null);
        setChunks([]);
      }
    } finally {
      setClearingDocuments(false);
    }
  }

  return (
    <div className="knowledge-workspace-container">
      <header className="workspace-header">
        <button className="btn-back" type="button" onClick={handleBack}>
          <ChevronLeft size={16} />返回列表
        </button>
        <div className="workspace-kb-title-block" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: 'var(--workspace-title-color, #111827)' }}>{kb?.name}</h3>
              <button
                type="button"
                className="coze-icon-button"
                title="编辑名称和描述"
                style={{
                  background: 'none',
                  border: 'none',
                  padding: '2px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  color: 'var(--workspace-action-muted-color, #667085)',
                  borderRadius: '4px',
                  transition: 'all 0.2s'
                }}
                onClick={() => {
                  setEditForm({ name: kb?.name || '', description: kb?.description || '' });
                  setEditDialogOpen(true);
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = 'var(--workspace-action-hover-color, #4d43e6)';
                  e.currentTarget.style.background = 'var(--workspace-action-hover-bg, #f4f4f5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = 'var(--workspace-action-muted-color, #667085)';
                  e.currentTarget.style.background = 'none';
                }}
              >
                <SquarePen size={14} />
              </button>
            </div>
            <p style={{ margin: '4px 0 0', fontSize: '12px', color: 'var(--workspace-muted-color, #667085)' }}>{kb?.description || '暂无描述'}</p>
          </div>
        </div>
        <div className="workspace-header-actions">
          <div className="add-content-wrapper">
            <button
              className="primary"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setAddDropdownOpen(!addDropdownOpen);
              }}
            >
              <Plus size={16} />添加内容
            </button>
            {addDropdownOpen && (
              <div className="add-dropdown-menu" onClick={(e) => e.stopPropagation()}>
                <button type="button" onClick={() => { setAddDropdownOpen(false); triggerLocalFileInput(); }}>
                  💻 本地文档
                </button>
                <button type="button" onClick={() => { setAddDropdownOpen(false); setCustomInputOpen(true); }}>
                  📝 自定义输入
                </button>
              </div>
            )}
          </div>
          <button
            className="btn-clear-documents"
            type="button"
            onClick={handleClearDocuments}
            disabled={clearingDocuments || !documents.length}
            title="清空该知识库中的所有文档"
          >
            <Trash2 size={15} />{clearingDocuments ? '清除中...' : '清除文档库'}
          </button>
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept={KNOWLEDGE_FILE_ACCEPT}
            multiple
            onChange={(event) => {
              if (setDocForm) {
                setDocForm((form) => ({ ...form, kb_id: String(kb.id) }));
              }
              handleKnowledgeFileInput(event, (file) => uploadKnowledgeFile(file, kb.id));
            }}
          />
        </div>
      </header>

      <div className="workspace-main-split">
        {/* Left column: search and documents list */}
        <section className="workspace-column left-column plain-panel">
          <div className="search-box-wrapper">
            <Search size={16} className="search-icon" />
            <input
              type="text"
              placeholder="搜索文档名称"
              value={docSearchQuery}
              onChange={(e) => setDocSearchQuery(e.target.value)}
            />
          </div>

          <div className="workspace-doc-list">
            {visibleUploadingItems.map((item) => {
              const failed = item.status === 'failed';
              const done = item.status === 'indexed';
              const active = !failed && !done;
              const label = failed ? '失败' : done ? '完成' : '上传中';
              const statusClass = failed ? 'failed' : done ? 'indexed' : 'indexing';
              const color = failed ? '#b42318' : done ? '#027a48' : '#4d43e6';
              const background = failed ? 'rgba(180, 35, 24, 0.04)' : done ? 'rgba(2, 122, 72, 0.04)' : 'rgba(77, 67, 230, 0.03)';
              return (
                <div
                  key={item.id}
                  className={`workspace-doc-row uploading active upload-${item.status}`}
                  style={{
                    opacity: 0.9,
                    cursor: 'default',
                    borderLeft: `3px solid ${color}`,
                    background,
                    display: 'flex',
                    alignItems: 'center',
                    padding: '10px 12px',
                    borderBottom: '1px solid #e5e7eb',
                    gap: '8px',
                  }}
                >
                  {active ? (
                    <span className="coze-spinner"></span>
                  ) : done ? (
                    <Check size={14} color={color} />
                  ) : (
                    <FileX2 size={14} color={color} />
                  )}
                  <div className="doc-row-details" style={{ minWidth: 0, flex: 1 }}>
                    <strong style={{ color, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block', fontSize: '13px' }}>
                      {item.filename || '正在上传文档...'}
                    </strong>
                    <small style={{ color: '#667085', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block', fontSize: '11px', marginTop: '2px' }}>
                      {item.statusText || '正在解析切片向量化...'}
                    </small>
                  </div>
                  <span className={`document-status ${statusClass}`} style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>
                    {label}
                  </span>
                </div>
              );
            })}
            {filteredDocs.map((doc) => {
              const isActive = activeDoc?.id === doc.id;
              const status = doc.status || 'uploaded';
              const sourceType = doc.source_type || 'file';

              return (
                <div
                  key={doc.id}
                  className={`workspace-doc-row ${isActive ? 'active' : ''} status-${status}`}
                  onClick={() => setActiveDoc(doc)}
                >
                  <FileText size={16} className="doc-icon" />
                  <div className="doc-row-details">
                    <strong>{doc.title || doc.filename || `document-${doc.id}`}</strong>
                    <small>
                      {doc.chunk_count ?? 0} chunks · {sourceType === 'file' ? '文件' : '文本'}
                    </small>
                  </div>
                  <span className={`document-status ${status}`}>
                    {status === 'indexed' ? '已索引' : status === 'indexing' ? '索引中' : '失败'}
                  </span>
                  <button
                    className="btn-delete-doc"
                    type="button"
                    title="删除文档"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteDocument(doc.id, kb?.id).then((deleted) => {
                        if (deleted && activeDoc?.id === doc.id) {
                          setActiveDoc(null);
                          setChunks([]);
                        }
                      });
                    }}
                  >
                    <FileX2 size={14} />
                  </button>
                </div>
              );
            })}
            {filteredDocs.length === 0 && (
              <p className="muted empty-workspace-docs">无文档资料，请点击右上角添加。</p>
            )}
          </div>
        </section>

        {/* Right column: active doc details & chunk stream */}
        <section className="workspace-column right-column plain-panel">
          {activeDoc ? (
            <div className="doc-detail-view">
              <div className="doc-detail-header-card">
                <div className="doc-meta-title-row">
                  <h4>{activeDoc.title || activeDoc.filename}</h4>
                  <div className="doc-detail-actions">
                    <button
                      className="btn-resegment-trigger"
                      type="button"
                      onClick={() => setResegmentOpen(true)}
                    >
                      <Settings2 size={14} />重新切片/调参
                    </button>
                    <button
                      className="btn-delete-active-doc"
                      type="button"
                      onClick={handleDeleteActiveDoc}
                      disabled={deletingDocId === activeDoc.id}
                    >
                      <FileX2 size={14} />{deletingDocId === activeDoc.id ? '删除中...' : '删除文档'}
                    </button>
                  </div>
                </div>
                <div className="doc-meta-grid">
                  <div className="meta-item">
                    <span>文件格式</span>
                    <strong>{activeDoc.content_type || 'text/plain'}</strong>
                  </div>
                  <div className="meta-item">
                    <span>分块数量</span>
                    <strong>{activeDoc.chunk_count ?? 0} 个 chunk</strong>
                  </div>
                  <div className="meta-item">
                    <span>索引状态</span>
                    <strong className={`status-text ${activeDoc.status}`}>
                      {activeDoc.status === 'indexed' ? '已完成' : '同步中'}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="chunk-list-section">
                <h5>分块预览 ({chunks.length} 个)</h5>
                <div className="chunk-card-stream">
                  {chunksLoading ? (
                    <div className="chunks-loading">
                      <span className="spinner"></span>加载分块中...
                    </div>
                  ) : (
                    chunks.map((chunk, index) => (
                      <div key={chunk.id || index} className="chunk-card">
                        <div className="chunk-card-header">
                          <span className="chunk-index-badge">#{chunk.chunk_index ?? index}</span>
                          <span className="chunk-dim">{chunk.embedding_dimension || 768}d</span>
                          {chunk.hierarchy_path && (
                            <span className="chunk-path-badge">🌳 {chunk.hierarchy_path}</span>
                          )}
                        </div>
                        <div className="chunk-text-content">{chunk.text}</div>
                      </div>
                    ))
                  )}
                  {!chunksLoading && chunks.length === 0 && (
                    <p className="muted empty-chunks">此文档暂无分块，请重新切片或检查索引状态。</p>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="workspace-empty-detail">
              <Database size={48} className="muted-icon" />
              <p>请在左侧列表选择一个文档查看详细的分块信息和参数配置。</p>
            </div>
          )}
        </section>
      </div>

      {/* Custom Paste Text Dialog */}
      {customInputOpen && (
        <div className="profile-dialog-backdrop">
          <section
            className="resource-form-dialog"
            role="dialog"
            onClick={(e) => e.stopPropagation()}
            style={{ width: '560px' }}
          >
            <button
              className="profile-dialog-close"
              type="button"
              title="关闭"
              onClick={() => setCustomInputOpen(false)}
            >
              <X size={16} />
            </button>
            <header className="model-dialog-heading">
              <h3>自定义文本输入</h3>
              <p>直接粘贴资料文本进行快速上传并建立索引。</p>
            </header>
            <form className="dialog-form" onSubmit={handlePasteSubmit}>
              <label className="field-stack">
                <span>文档名称</span>
                <input
                  type="text"
                  value={pasteFilename}
                  onChange={(e) => setPasteFilename(e.target.value)}
                  placeholder="例如: 产品指南.txt"
                  required
                />
              </label>
              <label className="field-stack">
                <span>文本内容</span>
                <textarea
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  placeholder="在这里粘贴或直接写入知识库内容..."
                  style={{ minHeight: '180px' }}
                  required
                />
              </label>
              <footer className="dialog-actions">
                <button type="button" onClick={() => setCustomInputOpen(false)}>
                  取消
                </button>
                <button className="primary-model-action" type="submit" disabled={pasteSubmitting}>
                  {pasteSubmitting ? '上传中...' : '确认上传'}
                </button>
              </footer>
            </form>
          </section>
        </div>
      )}

      {editDialogOpen && (
        <KnowledgeBaseDialog
          title="编辑知识库"
          description="修改知识库的名称和描述，方便在智能体配置中进行管理与调用。"
          submitText="保存修改"
          savingText="保存中..."
          isEdit={true}
          form={editForm}
          onChange={setEditForm}
          onCancel={() => setEditDialogOpen(false)}
          onSubmit={handleEditSubmit}
          saving={savingEdit}
        />
      )}
    </div>
  );
}









function SystemSettingsDialog({
  activePanel,
  me,
  onClose,
  onPanelChange,
  profileError,
  setProfileError,
  setThemeMode,
  themeMode,
  updateProfile,
  workspace,
}) {
  const [nameDraft, setNameDraft] = useState(me?.name || '');
  const [savingProfile, setSavingProfile] = useState(false);
  const currentPanel = activePanel === 'account' ? 'account' : 'general';

  useEffect(() => {
    setNameDraft(me?.name || '');
  }, [me?.name]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const themeOptions = [
    { value: 'light', label: '浅色', icon: Sun },
    { value: 'dark', label: '深色', icon: Moon },
    { value: 'system', label: '跟随系统', icon: Monitor },
  ];

  async function saveName() {
    const nextName = nameDraft.trim();
    if (!nextName || nextName === me?.name) return;
    setSavingProfile(true);
    try {
      await updateProfile({ name: nextName });
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSavingProfile(false);
    }
  }

  async function uploadAvatar(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp', 'image/gif'].includes(file.type)) {
      setProfileError('头像只支持 PNG、JPG、WebP 或 GIF。');
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setProfileError('头像文件不能超过 8MB。');
      return;
    }
    setSavingProfile(true);
    try {
      const avatarUrl = await createAvatarDataUrl(file);
      await updateProfile({ avatar_url: avatarUrl });
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSavingProfile(false);
    }
  }

  return (
    <div className="profile-dialog-backdrop system-settings-backdrop" onClick={onClose}>
      <section className="system-settings-dialog" role="dialog" aria-modal="true" aria-label="系统设置" onClick={(event) => event.stopPropagation()}>
        <button className="profile-dialog-close system-settings-close" type="button" title="关闭" aria-label="关闭系统设置" onClick={onClose}>
          <X size={16} />
        </button>
        <header className="system-settings-header">
          <h2>系统设置</h2>
        </header>
        <div className="system-settings-layout">
          <nav className="system-settings-nav" aria-label="设置分类">
            <button
              className={currentPanel === 'general' ? 'active' : ''}
              type="button"
              onClick={() => onPanelChange('general')}
            >
              <Settings2 size={17} />
              通用设置
            </button>
            <button
              className={currentPanel === 'account' ? 'active' : ''}
              type="button"
              onClick={() => onPanelChange('account')}
            >
              <KeyRound size={17} />
              账号管理
            </button>
          </nav>
          <div className="system-settings-content">
            {currentPanel === 'general' ? (
              <>
                <section className="settings-section">
                  <h3>主题</h3>
                  <div className="theme-choice-grid">
                    {themeOptions.map((option) => {
                      const Icon = option.icon;
                      return (
                        <button
                          key={option.value}
                          className={themeMode === option.value ? 'active' : ''}
                          type="button"
                          aria-pressed={themeMode === option.value}
                          onClick={() => setThemeMode(option.value)}
                        >
                          <Icon size={18} />
                          <span>{option.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </section>
                <section className="settings-section settings-inline-row">
                  <h3>语言</h3>
                  <button className="settings-select" type="button" disabled>
                    简体中文
                  </button>
                </section>
              </>
            ) : (
              <section className="settings-account-panel">
                <div className="profile-card settings-profile-card">
                  <UserAvatar user={me} className="profile-avatar" />
                  <div>
                    <h3>个人资料</h3>
                    <p>{me?.name || '未设置姓名'}</p>
                  </div>
                </div>
                <div className="profile-actions">
                  <label className="avatar-upload">
                    {savingProfile ? '上传中...' : '上传头像'}
                    <input type="file" accept="image/png,image/jpeg,image/webp,image/gif" onChange={uploadAvatar} disabled={savingProfile} />
                  </label>
                </div>
                <div className="profile-edit">
                  <input value={nameDraft} onChange={(event) => setNameDraft(event.target.value)} placeholder="姓名" />
                  <button type="button" onClick={saveName} disabled={savingProfile || !nameDraft.trim()}>保存姓名</button>
                </div>
                {profileError && <p className="error">{profileError}</p>}
                <div className="profile-grid">
                  <span>邮箱</span>
                  <strong>{me?.email || '-'}</strong>
                  <span>角色</span>
                  <strong>{roleLabel(workspace?.role)}</strong>
                  <span>账号状态</span>
                  <strong>已登录</strong>
                </div>
              </section>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function UserModelsHome({ adminModels, canManage, createModelConfig, deleteModelConfig, requestDeleteConfirm, setProfileError, updateModelConfig, ...userModelProps }) {
  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>我的模型</h1>
          <p>维护你自己的 OpenAI-compatible 模型连接，保存后可在智能体配置里选择。</p>
        </div>
      </header>
      <UserModelsPanel requestDeleteConfirm={requestDeleteConfirm} setProfileError={setProfileError} {...userModelProps} />
      {canManage && (
        <ModelAdminPanel
          createModelConfig={createModelConfig}
          deleteModelConfig={deleteModelConfig}
          models={adminModels}
          requestDeleteConfirm={requestDeleteConfirm}
          setProfileError={setProfileError}
          updateModelConfig={updateModelConfig}
        />
      )}
    </div>
  );
}

function ToolsHome({ createToolConfig, discoverMcpTools, deleteToolConfig, isDarkTheme, notify, openBuilder, requestDeleteConfirm, setProfileError, testToolConfig, tools, updateToolConfig }) {
  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>工具</h1>
          <p>管理可绑定到智能体的内置搜索、HTTP 工具和 MCP 工具。密钥只在保存时提交，保存后仅显示 has_secret 状态。</p>
        </div>
        <button className="primary" type="button" onClick={() => openBuilder()}><Bot size={16} />打开 Builder</button>
      </header>
      <ToolsPanel
        createToolConfig={createToolConfig}
        discoverMcpTools={discoverMcpTools}
        deleteToolConfig={deleteToolConfig}
        isDarkTheme={isDarkTheme}
        notify={notify}
        requestDeleteConfirm={requestDeleteConfirm}
        setProfileError={setProfileError}
        testToolConfig={testToolConfig}
        tools={tools}
        updateToolConfig={updateToolConfig}
      />
    </div>
  );
}

function skillActivationLabel(mode) {
  return {
    always: '每轮加载',
    auto: '自动选择',
    manual: '手动触发',
    disabled: '已禁用',
  }[mode] || '自动选择';
}

function SkillsHome({
  createSkill,
  deleteSkill,
  knowledgeBases,
  notify,
  setProfileError,
  skills,
  token,
  tools,
  updateSkill,
}) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [form, setForm] = useState(() => defaultSkillForm());
  const [saving, setSaving] = useState(false);
  const [editingId, setEditingId] = useState(null);

  function openCreate() {
    setEditingId(null);
    setForm(defaultSkillForm());
    setProfileError('');
    setDialogOpen(true);
  }

  function openEdit(summary) {
    // Fetch full skill detail to populate the form with system_prompt, tools, kbs
    setSaving(true);
    setEditingId(summary.id);
    setProfileError('');
    api(`/api/skills/${summary.id}`, { token })
      .then((data) => {
        const skill = data.skill || {};
        setForm({
          name: skill.name || '',
          description: skill.description || '',
          system_prompt: skill.system_prompt || '',
          icon: skill.icon || 'SK',
          category: skill.category || 'general',
          activation_mode: skill.activation_mode || 'auto',
          tagsText: (skill.tags || []).join(', '),
          tags: skill.tags || [],
          tool_ids: (skill.tools || []).map((t) => t.id),
          knowledge_base_ids: (skill.knowledge_bases || []).map((kb) => kb.id),
        });
        setDialogOpen(true);
      })
      .catch((err) => {
        setProfileError(errorMessage(err));
        setEditingId(null);
      })
      .finally(() => setSaving(false));
  }

  function closeDialog() {
    if (saving) return;
    setDialogOpen(false);
    setEditingId(null);
    setForm(defaultSkillForm());
  }

  async function submitSkill(event) {
    event.preventDefault();
    setSaving(true);
    setProfileError('');
    try {
      if (editingId) {
        await updateSkill(editingId, form);
      } else {
        await createSkill(form);
      }
      setDialogOpen(false);
      setEditingId(null);
      setForm(defaultSkillForm());
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(skill) {
    const confirmed = await window.confirm(`确定要删除技能「${skill.name}」吗？`);
    if (!confirmed) return;
    try {
      await deleteSkill(skill.id);
      notify('技能已删除。');
    } catch (err) {
      setProfileError(errorMessage(err));
    }
  }

  return (
    <div className="content-page">
      <header className="page-heading">
        <div>
          <h1>技能</h1>
          <p>管理可复用的能力包。每个技能包含 Prompt 片段、专属工具和知识库，可挂载到多个智能体。</p>
        </div>
        <button className="primary" type="button" onClick={openCreate}><Plus size={16} />新建技能</button>
      </header>

      {skills.length === 0 ? (
        <div className="empty-state" style={{ textAlign: 'center', padding: '60px 20px', color: '#94a3b8' }}>
          <Sparkles size={48} style={{ marginBottom: '16px', opacity: 0.4 }} />
          <p style={{ fontSize: '15px', margin: 0 }}>暂无技能，点击右上角"新建技能"开始创建。</p>
        </div>
      ) : (
        <div className="skill-card-list">
          {skills.map((skill) => (
            <div className="skill-card" key={skill.id}>
              <span className="skill-card-icon">✨</span>
              <div className="skill-card-body">
                <div className="skill-card-head">
                  <strong className="skill-card-name">{skill.name}</strong>
                  <span className="mode-chip skill-card-category">{skill.category || 'general'}</span>
                  <span className="skill-card-tag">{skillActivationLabel(skill.activation_mode)}</span>
                  {(skill.tags || []).slice(0, 3).map((tag) => (
                    <span className="skill-card-tag" key={tag}>{tag}</span>
                  ))}
                </div>
                <p className="skill-card-desc">
                  {skill.description || '无描述'}
                </p>
              </div>
              <div className="skill-card-actions">
                <button type="button" className="ghost-icon small" title="编辑技能" onClick={() => openEdit(skill)}>
                  <SquarePen size={14} />
                </button>
                <button type="button" className="ghost-icon small danger" title="删除技能" onClick={() => handleDelete(skill)}>
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {dialogOpen && (
        <SkillDialog
          form={form}
          onCancel={closeDialog}
          onChange={setForm}
          onSubmit={submitSkill}
          saving={saving}
          tools={tools}
          knowledgeBases={knowledgeBases}
          title={editingId ? "编辑技能" : "新建技能"}
          description={editingId ? "修改技能的能力包内容。" : "创建一个可复用的能力包：包含 Prompt 片段、专属工具和知识库。"}
          submitText={editingId ? "保存修改" : "创建技能"}
          savingText={editingId ? "保存中..." : "创建中..."}
          isEdit={!!editingId}
        />
      )}
    </div>
  );
}

function ResourceLibraryHome({
  activeAgentId,
  agentForm,
  copyBuiltinPromptTemplate,
  createPromptTemplate,
  deletePromptTemplate,
  knowledgeBases,
  openBuilder,
  promptTemplates,
  requestDeleteConfirm,
  setActiveNav,
  setAgentForm,
  setProfileError,
  setView,
  tools,
  updatePromptTemplate,
}) {
  const [tab, setTab] = useState('all');
  const [query, setQuery] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState(promptTemplates[0] || null);
  const [editingTemplate, setEditingTemplate] = useState(null);
  const [form, setForm] = useState(() => defaultPromptTemplateForm());
  const [formOpen, setFormOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');

  useEffect(() => {
    if (!selectedTemplate && promptTemplates.length) {
      setSelectedTemplate(promptTemplates[0]);
    } else if (selectedTemplate && !promptTemplates.some((item) => item.id === selectedTemplate.id)) {
      setSelectedTemplate(promptTemplates[0] || null);
    }
  }, [promptTemplates, selectedTemplate?.id]);

  const filteredTemplates = filterPromptTemplates(promptTemplates, query);
  const filteredTools = filterResourceItems(tools, query, (tool) => `${tool.label || ''} ${tool.name || ''} ${tool.description || ''}`);
  const filteredKnowledge = filterResourceItems(knowledgeBases, query, (kb) => `${kb.name || ''} ${kb.description || ''}`);
  const showPrompts = tab === 'all' || tab === 'prompts';
  const showTools = tab === 'all' || tab === 'tools';
  const showKnowledge = tab === 'all' || tab === 'knowledge';

  function insertTemplate(template) {
    if (!template?.content) return;
    insertPromptIntoAgent(setAgentForm, template.content);
    setSelectedTemplate(template);
    setNotice('模板已插入当前智能体 Prompt。');
  }

  function openCreate(template = null) {
    setEditingTemplate(null);
    setForm(template ? formFromPromptTemplate(template, { title: `${template.title} 副本` }) : defaultPromptTemplateForm());
    setNotice('');
    setFormOpen(true);
  }

  function openEdit(template) {
    setEditingTemplate(template);
    setForm(formFromPromptTemplate(template));
    setSelectedTemplate(template);
    setNotice('');
    setFormOpen(true);
  }

  function closeTemplateForm() {
    if (saving) return;
    setFormOpen(false);
    setEditingTemplate(null);
    setForm(defaultPromptTemplateForm());
  }

  async function saveTemplate(event) {
    event.preventDefault();
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      const payload = promptTemplateFormPayload(form);
      const saved = editingTemplate?.db_id
        ? await updatePromptTemplate(editingTemplate.db_id, payload)
        : await createPromptTemplate(payload);
      setEditingTemplate(null);
      setForm(defaultPromptTemplateForm());
      setFormOpen(false);
      setSelectedTemplate(saved);
      setNotice('提示词模板已保存。');
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function copyBuiltin(template) {
    if (!template?.id) return;
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      const copied = await copyBuiltinPromptTemplate({
        builtin_id: template.id.replace('builtin:', ''),
        title: `${template.title} 副本`,
      });
      setSelectedTemplate(copied);
      setNotice('已复制为我的模板。');
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function removeTemplate(template) {
    if (!template?.db_id) return;
    const confirmed = await requestDeleteConfirm({
      title: '删除提示词模板',
      message: `删除「${template.title}」？`,
      detail: '删除后，资源库和 Builder 模板区都不再显示该模板。',
      confirmLabel: '删除模板',
    });
    if (!confirmed) return;
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      await deletePromptTemplate(template.db_id);
      setSelectedTemplate(null);
      setNotice('模板已删除。');
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="content-page resource-page">
      <header className="page-heading resource-heading">
        <div>
          <h1>资源库</h1>
          <p>管理当前可用资源。这里暂只展示已实现的插件、知识库和提示词。</p>
        </div>
        <div className="resource-actions">
          <label className="resource-search">
            <Search size={16} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索资源" />
          </label>
          <button className="primary" type="button" onClick={() => openCreate()}><Plus size={15} />新建提示词</button>
        </div>
      </header>

      <div className="resource-tabs">
        {[
          ['all', '全部'],
          ['tools', '插件'],
          ['knowledge', '知识库'],
          ['prompts', '提示词'],
        ].map(([key, label]) => (
          <button key={key} type="button" className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}</button>
        ))}
      </div>

      <div className="resource-layout">
        <section className="resource-list-panel">
          {showPrompts && (
            <ResourceSection
              title="提示词"
              count={filteredTemplates.length}
              emptyText="暂无提示词模板"
            >
              {filteredTemplates.map((template) => (
                <ResourceRow
                  key={template.id}
                  icon={<FileText size={17} />}
                  title={template.title}
                  desc={template.description || template.content}
                  type={template.source === 'builtin' ? '预置提示词' : '我的提示词'}
                  meta={template.category || 'general'}
                  active={selectedTemplate?.id === template.id}
                  onClick={() => setSelectedTemplate(template)}
                  actions={
                    <>
                      <button type="button" onClick={(event) => { event.stopPropagation(); setSelectedTemplate(template); }}>预览</button>
                      <button type="button" onClick={(event) => { event.stopPropagation(); insertTemplate(template); }}>插入</button>
                      {template.source === 'builtin' && <button type="button" disabled={saving} onClick={(event) => { event.stopPropagation(); copyBuiltin(template); }}>复制</button>}
                      {template.editable && <button type="button" disabled={saving} onClick={(event) => { event.stopPropagation(); openEdit(template); }}>编辑</button>}
                      {template.editable && <button type="button" disabled={saving} onClick={(event) => { event.stopPropagation(); removeTemplate(template); }}>删除</button>}
                    </>
                  }
                />
              ))}
            </ResourceSection>
          )}

          {showTools && (
            <ResourceSection title="插件" count={filteredTools.length} emptyText="暂无插件">
              {filteredTools.map((tool) => (
                <ResourceRow
                  key={`tool-${tool.id}`}
                  icon={<Wand2 size={17} />}
                  title={tool.label || tool.name}
                  desc={tool.description || tool.name}
                  type="插件"
                  meta={`${toolType(tool)} · ${tool.enabled === false ? '停用' : '启用'}`}
                  actions={<button type="button" onClick={() => setActiveNav('tools')}>管理</button>}
                />
              ))}
            </ResourceSection>
          )}

          {showKnowledge && (
            <ResourceSection title="知识库" count={filteredKnowledge.length} emptyText="暂无知识库">
              {filteredKnowledge.map((kb) => (
                <ResourceRow
                  key={`kb-${kb.id}`}
                  icon={<Database size={17} />}
                  title={kb.name}
                  desc={kb.description || `${kb.document_count || 0} 个文档`}
                  type="知识库"
                  meta={`${kb.document_count || 0} 文档`}
                  actions={<button type="button" onClick={() => setActiveNav('knowledge')}>管理</button>}
                />
              ))}
            </ResourceSection>
          )}
        </section>

        <aside className="resource-detail-panel">
          <PromptTemplatePreview
            activeAgentId={activeAgentId}
            template={selectedTemplate}
            onInsert={insertTemplate}
            onCopy={copyBuiltin}
            onEdit={openEdit}
            onDelete={removeTemplate}
            saving={saving}
          />
          <section className="resource-side-actions">
            <button type="button" onClick={() => { setView('builder'); openBuilder(); }}>打开 Builder</button>
            <button className="primary-model-action" type="button" onClick={() => openCreate()}><Plus size={15} />新建模板</button>
          </section>
        </aside>
      </div>
      {notice && <p className="model-row-warning floating-notice">{notice}</p>}
      {formOpen && (
        <PromptTemplateDialog
          editingTemplate={editingTemplate}
          form={form}
          onCancel={closeTemplateForm}
          onChange={setForm}
          onSubmit={saveTemplate}
          saving={saving}
        />
      )}
    </div>
  );
}

function ResourceSection({ children, count, emptyText, title }) {
  return (
    <div className="resource-section">
      <div className="resource-section-title">
        <strong>{title}</strong>
        <span>{count}</span>
      </div>
      <div className="resource-rows">
        {children}
        {count === 0 && <p className="muted">{emptyText}</p>}
      </div>
    </div>
  );
}

function ResourceRow({ active = false, actions, desc, icon, meta, onClick, title, type }) {
  return (
    <article className={`resource-row ${active ? 'active' : ''}`} onClick={onClick || undefined}>
      <span className="resource-icon">{icon}</span>
      <div>
        <strong>{title}</strong>
        <small>{desc}</small>
      </div>
      <span className="resource-type">{type}</span>
      <span className="resource-meta">{meta}</span>
      <div className="resource-row-actions">{actions}</div>
    </article>
  );
}

function PromptTemplatePreview({ activeAgentId, onCopy, onDelete, onEdit, onInsert, saving, template }) {
  if (!template) {
    return (
      <section className="prompt-preview-panel">
        <h3>模板预览</h3>
        <p className="muted">选择一个提示词模板后在这里预览和插入。</p>
      </section>
    );
  }
  return (
    <section className="prompt-preview-panel">
      <div className="prompt-preview-head">
        <div>
          <span>{template.source === 'builtin' ? '平台预置' : '我的模板'}</span>
          <h3>{template.title}</h3>
          <p>{template.description || '暂无描述'}</p>
        </div>
        <span className="soft-pill">{template.category || 'general'}</span>
      </div>
      <pre>{template.content}</pre>
      <div className="prompt-preview-actions">
        <button className="primary" type="button" disabled={!activeAgentId} onClick={() => onInsert(template)}>插入到当前智能体</button>
        {template.source === 'builtin' && <button type="button" disabled={saving} onClick={() => onCopy(template)}>复制为我的模板</button>}
        {template.editable && <button type="button" disabled={saving} onClick={() => onEdit(template)}>编辑</button>}
        {template.editable && <button type="button" disabled={saving} onClick={() => onDelete(template)}>删除</button>}
      </div>
    </section>
  );
}

const HTTP_TOOL_PRESET = {
  type: 'http',
  name: 'weather_lookup',
  label: 'Weather lookup',
  description: 'Fetches weather data from an HTTPS API.',
  enabled: true,
  method: 'GET',
  url: 'https://api.example.com/weather',
  headers_schema: JSON.stringify({}, null, 2),
  query_schema: JSON.stringify({ city: { type: 'string', required: true } }, null, 2),
  body_schema: JSON.stringify({}, null, 2),
  auth_type: 'none',
  auth_header_name: 'Authorization',
  auth_query_name: '',
  auth_secret: '',
  response_path: '$',
  timeout_seconds: '10',
};

const BUILTIN_SEARCH_PRESET = {
  ...HTTP_TOOL_PRESET,
  type: 'builtin_search',
  name: 'builtin_search',
  label: '内置搜索',
  description: '\u5e73\u53f0\u5185\u7f6e\u7684\u8054\u7f51\u641c\u7d22\u5de5\u5177\u3002',
  method: 'GET',
  url: '',
  query_schema: JSON.stringify({ query: { type: 'string', required: true } }, null, 2),
  auth_type: 'none',
  response_path: '$',
};

const MCP_TOOL_PRESET = {
  ...HTTP_TOOL_PRESET,
  type: 'mcp',
  name: 'mcp_tool',
  label: 'MCP Tool',
  description: 'Calls a remote MCP tool over Streamable HTTP, SSE, or stdio.',
  method: 'POST',
  url: 'http://127.0.0.1:8001/mcp',
  headers_schema: JSON.stringify({}, null, 2),
  query_schema: JSON.stringify({}, null, 2),
  body_schema: JSON.stringify({}, null, 2),
  response_path: '$',
  timeout_seconds: '30',
  mcp_transport: 'streamable_http',
  mcp_tool_name: '',
  mcp_input_schema: JSON.stringify({ type: 'object', properties: {} }, null, 2),
  mcp_command: '',
  mcp_args: '',
  mcp_env: '',
  mcp_cwd: '',
  server_label: '',
};

function createToolForm(type = 'http') {
  if (type === 'builtin_search') return { ...BUILTIN_SEARCH_PRESET };
  if (type === 'mcp') return { ...MCP_TOOL_PRESET };
  return { ...HTTP_TOOL_PRESET };
}

function formFromTool(tool, overrides = {}) {
  const type = toolType(tool);
  const mcp = tool?.mcp || {};
  return {
    ...createToolForm(type),
    type,
    name: tool?.name || '',
    label: tool?.label || '',
    description: tool?.description || '',
    enabled: tool?.enabled !== false,
    method: tool?.method || 'GET',
    url: tool?.url || '',
    headers_schema: JSON.stringify(tool?.headers_schema || {}, null, 2),
    query_schema: JSON.stringify(tool?.query_schema || {}, null, 2),
    body_schema: JSON.stringify(tool?.body_schema || {}, null, 2),
    auth_type: tool?.auth?.type || tool?.auth_type || 'none',
    auth_header_name: tool?.auth?.header_name || tool?.auth_header_name || 'Authorization',
    auth_query_name: tool?.auth?.query_name || tool?.auth_query_name || '',
    auth_secret: '',
    response_path: tool?.response_path || '$',
    timeout_seconds: String(tool?.timeout_seconds || (tool?.type === 'mcp' ? 30 : 10)),
    mcp_transport: mcp.transport || 'streamable_http',
    mcp_tool_name: mcp.tool_name || '',
    mcp_input_schema: JSON.stringify(mcp.input_schema || { type: 'object', properties: {} }, null, 2),
    mcp_command: mcp.command || '',
    mcp_args: Array.isArray(mcp.args) ? mcp.args.join(' ') : (mcp.args || ''),
    mcp_env: mcp.env && typeof mcp.env === 'object' ? JSON.stringify(mcp.env, null, 2) : '',
    mcp_cwd: mcp.cwd || '',
    server_label: tool?.server_label || '',
    ...overrides,
  };
}

function isUserTool(tool) {
  return Boolean(tool?.created_by);
}

// Dual-way parameter translator: Array <=> JSON Schema string
function paramsToSchema(paramsArray) {
  const schema = {};
  paramsArray.forEach(p => {
    if (p.name.trim()) {
      schema[p.name.trim()] = {
        type: p.type || 'string',
        required: Boolean(p.required),
        description: p.description || ''
      };
    }
  });
  return JSON.stringify(schema, null, 2);
}

function schemaToParams(schemaStr) {
  try {
    const schema = JSON.parse(schemaStr || '{}');
    return Object.entries(schema).map(([name, spec], index) => ({
      id: `${name}-${index}-${Date.now()}-${Math.random()}`,
      name,
      type: spec?.type || 'string',
      required: Boolean(spec?.required),
      description: spec?.description || ''
    }));
  } catch (e) {
    return [];
  }
}

function ParamTableEditor({ isDarkTheme = false, label, params, onChange }) {
  const editorTheme = isDarkTheme ? {
    panelBg: '#121620',
    panelBorder: '#303544',
    title: '#f3f4f7',
    muted: '#8f98aa',
    inputBg: '#171a22',
    inputBorder: '#343847',
    inputText: '#f3f4f7',
    actionBg: '#17233e',
    actionBorder: '#364f91',
    actionText: '#9fc0ff',
    dangerBg: '#2a1715',
    dangerText: '#ffb3a8',
  } : {
    panelBg: '#f8fafc',
    panelBorder: '#dfe4ef',
    title: '#1f2937',
    muted: '#94a3b8',
    inputBg: '#fff',
    inputBorder: '#dfe4ef',
    inputText: '#111827',
    actionBg: '#eef2ff',
    actionBorder: '#c7d2fe',
    actionText: '#4d43e6',
    dangerBg: '#fee2e2',
    dangerText: '#ef4444',
  };
  const addRow = () => {
    const newRow = {
      id: `param-${Date.now()}-${Math.random()}`,
      name: '',
      type: 'string',
      required: false,
      description: ''
    };
    onChange([...params, newRow]);
  };

  const removeRow = (id) => {
    onChange(params.filter(p => p.id !== id));
  };

  const updateRow = (id, patch) => {
    onChange(params.map(p => p.id === id ? { ...p, ...patch } : p));
  };

  return (
    <div className="param-table-editor-wrapper" style={{ marginTop: '14px', border: `1px solid ${editorTheme.panelBorder}`, borderRadius: '10px', padding: '14px', background: editorTheme.panelBg }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <strong style={{ fontSize: '13px', color: editorTheme.title }}>{label}</strong>
        <button 
          type="button" 
          onClick={addRow}
          style={{ background: editorTheme.actionBg, color: editorTheme.actionText, border: `1px solid ${editorTheme.actionBorder}`, padding: '4px 10px', borderRadius: '6px', fontSize: '12px', fontWeight: 'bold', cursor: 'pointer' }}
        >
          + 添加参数
        </button>
      </div>
      
      {params.length === 0 ? (
        <p style={{ fontStyle: 'italic', fontSize: '12px', color: editorTheme.muted, margin: '4px 0', textAlign: 'center' }}>暂无参数，点击右上角一键添加。</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {params.map((row) => (
            <div key={row.id} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <input 
                type="text" 
                placeholder="参数名" 
                value={row.name} 
                onChange={(e) => updateRow(row.id, { name: e.target.value })}
                style={{ flex: 2, padding: '5px 8px', border: `1px solid ${editorTheme.inputBorder}`, borderRadius: '6px', fontSize: '12px', background: editorTheme.inputBg, color: editorTheme.inputText }}
              />
              <select 
                value={row.type} 
                onChange={(e) => updateRow(row.id, { type: e.target.value })}
                style={{ flex: 1.5, padding: '5px 8px', border: `1px solid ${editorTheme.inputBorder}`, borderRadius: '6px', fontSize: '12px', background: editorTheme.inputBg, color: editorTheme.inputText }}
              >
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="integer">integer</option>
                <option value="boolean">boolean</option>
              </select>
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '12px', color: editorTheme.muted, padding: '0 4px', whiteSpace: 'nowrap' }}>
                <input 
                  type="checkbox" 
                  checked={row.required} 
                  onChange={(e) => updateRow(row.id, { required: e.target.checked })} 
                />
                必填
              </label>
              <input 
                type="text" 
                placeholder="参数描述或说明" 
                value={row.description} 
                onChange={(e) => updateRow(row.id, { description: e.target.value })}
                style={{ flex: 3, padding: '5px 8px', border: `1px solid ${editorTheme.inputBorder}`, borderRadius: '6px', fontSize: '12px', background: editorTheme.inputBg, color: editorTheme.inputText }}
              />
              <button 
                type="button" 
                onClick={() => removeRow(row.id)}
                style={{ background: editorTheme.dangerBg, color: editorTheme.dangerText, border: 'none', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', fontSize: '12px' }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ToolsPanel({ createToolConfig, discoverMcpTools, deleteToolConfig, isDarkTheme = false, notify, requestDeleteConfirm, setProfileError, testToolConfig, tools, updateToolConfig }) {
  const [form, setForm] = useState(createToolForm);
  const [formOpen, setFormOpen] = useState(false);
  const [editingTool, setEditingTool] = useState(null);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');
  const [testingId, setTestingId] = useState(null);
  const [testingTool, setTestingTool] = useState(null);
  const [testInputById, setTestInputById] = useState({});
  const [testBodyById, setTestBodyById] = useState({});
  const [testResults, setTestResults] = useState({});
  const [secretDialogTool, setSecretDialogTool] = useState(null);
  const [editingMcpGroup, setEditingMcpGroup] = useState(null);
  const [discoveringMcp, setDiscoveringMcp] = useState(false);
  const [mcpDiscoveries, setMcpDiscoveries] = useState([]);
  const [expandedMcpGroups, setExpandedMcpGroups] = useState({});
  
  // States for visual parameter editor
  const [headersParams, setHeadersParams] = useState([]);
  const [queryParams, setQueryParams] = useState([]);
  const [bodyParams, setBodyParams] = useState([]);

  const isHttpForm = form.type === 'http';
  const isMcpForm = form.type === 'mcp';
  const isMcpStdio = isMcpForm && mcpTransportValue(form.mcp_transport, form.url) === 'stdio';
  const isMcpRemote = isMcpForm && !isMcpStdio;
  const isEditingMcpGroup = Boolean(editingMcpGroup);
  const isCreatingMcpGroup = isMcpForm && !editingTool && !isEditingMcpGroup;
  const needsBodySchema = isHttpForm && !['GET', 'DELETE'].includes(String(form.method || '').toUpperCase());
  const needsAuthSecret = (isHttpForm || isMcpRemote) && form.auth_type !== 'none';
  const needsAuthHeader = (isHttpForm || isMcpRemote) && ['bearer', 'header'].includes(form.auth_type);
  const needsAuthQuery = (isHttpForm || isMcpRemote) && form.auth_type === 'query';
  const toolDisplayEntries = useMemo(() => buildToolDisplayGroups(tools), [tools]);
  const toolTheme = isDarkTheme ? {
    rowBg: '#181818',
    rowBorder: '#343434',
    rowHoverBorder: '#525252',
    rowShadow: 'none',
    rowHoverShadow: '0 10px 22px rgba(0,0,0,0.24)',
    title: '#f3f4f7',
    text: '#a8a8a8',
    muted: '#777777',
    codeBg: '#2a2a2a',
    codeText: '#d0d0d0',
    methodBg: '#252525',
    methodText: '#e0e0e0',
    buttonBg: '#252525',
    buttonBorder: '#3c3c3c',
    buttonText: '#e0e0e0',
    accentBg: '#17233a',
    accentBorder: '#31518c',
    accentText: '#a8c7ff',
    successBg: 'rgba(126, 224, 170, 0.12)',
    successText: '#7ee0aa',
    warnText: '#f7c76f',
    dangerBg: '#2a1715',
    dangerBorder: '#79372d',
    dangerText: '#ffb3a8',
    typeNeutral: { background: '#242424', color: '#d7d7d7', border: '1px solid #424242' },
    typeSearch: { background: '#182235', color: '#9fc0ff', border: '1px solid #34486f' },
    typeBuiltin: { background: '#242424', color: '#d7d7d7', border: '1px solid #424242' },
    typeHttp: { background: '#14251d', color: '#7ee0aa', border: '1px solid #2d5f48' },
    typeMcp: { background: '#2a2118', color: '#ffbe82', border: '1px solid #6a4a2d' },
    emptyBg: '#181818',
    emptyBorder: '#343434',
  } : {
    rowBg: '#ffffff',
    rowBorder: '#e5e7eb',
    rowHoverBorder: '#4d43e6',
    rowShadow: '0 1px 3px rgba(0,0,0,0.02)',
    rowHoverShadow: '0 10px 20px rgba(77, 67, 230, 0.05)',
    title: '#111827',
    text: '#6b7280',
    muted: '#9ca3af',
    codeBg: '#f3f4f6',
    codeText: '#4b5563',
    methodBg: '#f3f4f6',
    methodText: '#374151',
    buttonBg: '#ffffff',
    buttonBorder: '#dfe4ef',
    buttonText: '#374151',
    accentBg: 'rgba(77, 67, 230, 0.08)',
    accentBorder: 'rgba(77, 67, 230, 0.15)',
    accentText: '#4d43e6',
    successBg: 'rgba(16, 185, 129, 0.08)',
    successText: '#10b981',
    warnText: '#d97706',
    dangerBg: '#fee2e2',
    dangerBorder: '#fecaca',
    dangerText: '#ef4444',
    typeNeutral: { background: '#f3f4f6', color: '#374151', border: '1px solid #e5e7eb' },
    typeSearch: { background: '#e0f2fe', color: '#0369a1', border: '1px solid #bae6fd' },
    typeBuiltin: { background: '#f3e8ff', color: '#6b21a8', border: '1px solid #e9d5ff' },
    typeHttp: { background: '#dcfce7', color: '#15803d', border: '1px solid #bbf7d0' },
    typeMcp: { background: '#fff7ed', color: '#c2410c', border: '1px solid #fed7aa' },
    emptyBg: '#f9fafb',
    emptyBorder: '#e5e7eb',
  };
  const toolButtonStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    padding: '6px 12px',
    borderRadius: '8px',
    background: toolTheme.buttonBg,
    border: `1px solid ${toolTheme.buttonBorder}`,
    color: toolTheme.buttonText,
    fontWeight: 600,
    cursor: 'pointer',
    fontSize: '12px',
  };
  const toolAccentButtonStyle = {
    ...toolButtonStyle,
    background: toolTheme.accentBg,
    border: `1px solid ${toolTheme.accentBorder}`,
    color: toolTheme.accentText,
  };

  function switchType(type) {
    const preset = createToolForm(type);
    setForm((current) => ({
      ...preset,
      name: current.name && current.type === type ? current.name : preset.name,
      auth_type: type === 'builtin_search' ? 'none' : current.auth_type || preset.auth_type,
    }));
    setHeadersParams(schemaToParams(preset.headers_schema));
    setQueryParams(schemaToParams(preset.query_schema));
    setBodyParams(schemaToParams(preset.body_schema));
    setMcpDiscoveries([]);
  }

  function updateToolForm(patch) {
    setForm((current) => {
      const next = { ...current, ...patch };
      if (Object.prototype.hasOwnProperty.call(patch, 'method') && ['GET', 'DELETE'].includes(String(patch.method).toUpperCase())) {
        next.body_schema = '{}';
        setBodyParams([]);
      }
      if (Object.prototype.hasOwnProperty.call(patch, 'auth_type')) {
        if (patch.auth_type === 'none') {
          next.auth_secret = '';
        }
        if (patch.auth_type !== 'query') {
          next.auth_query_name = '';
        }
        if (!['bearer', 'header'].includes(patch.auth_type)) {
          next.auth_header_name = 'Authorization';
        }
      }
      return next;
    });
  }

  function openToolForm(type = 'http') {
    const defaultForm = createToolForm(type);
    setForm(defaultForm);
    setHeadersParams(schemaToParams(defaultForm.headers_schema));
    setQueryParams(schemaToParams(defaultForm.query_schema));
    setBodyParams(schemaToParams(defaultForm.body_schema));
    setMcpDiscoveries([]);
    setEditingTool(null);
    setEditingMcpGroup(null);
    setNotice('');
    setProfileError('');
    setFormOpen(true);
  }

  function openEditTool(tool) {
    const editForm = formFromTool(tool);
    setForm(editForm);
    setHeadersParams(schemaToParams(editForm.headers_schema));
    setQueryParams(schemaToParams(editForm.query_schema));
    setBodyParams(schemaToParams(editForm.body_schema));
    setMcpDiscoveries([]);
    setEditingTool(tool);
    setEditingMcpGroup(null);
    setNotice('');
    setProfileError('');
    setFormOpen(true);
  }

  function openEditMcpGroup(entry) {
    const firstTool = entry.tools[0];
    const editForm = formFromTool(firstTool, {
      enabled: entry.enabled,
      label: entry.title,
      description: '',
      server_label: firstTool?.server_label || '',
    });
    setForm(editForm);
    setHeadersParams(schemaToParams(editForm.headers_schema));
    setQueryParams(schemaToParams(editForm.query_schema));
    setBodyParams(schemaToParams(editForm.body_schema));
    setMcpDiscoveries([]);
    setEditingTool(null);
    setEditingMcpGroup(entry);
    setNotice('');
    setProfileError('');
    setFormOpen(true);
  }

  function openCopyTool(tool) {
    const copyForm = formFromTool(tool, { name: `${tool.name || 'tool'}_copy`, label: `${tool.label || tool.name} 副本`, enabled: true });
    setForm(copyForm);
    setHeadersParams(schemaToParams(copyForm.headers_schema));
    setQueryParams(schemaToParams(copyForm.query_schema));
    setBodyParams(schemaToParams(copyForm.body_schema));
    setMcpDiscoveries([]);
    setEditingTool(null);
    setEditingMcpGroup(null);
    setNotice('');
    setProfileError('');
    setFormOpen(true);
  }

  function closeToolForm() {
    if (saving) return;
    setFormOpen(false);
    setEditingTool(null);
    setEditingMcpGroup(null);
    setMcpDiscoveries([]);
  }

  function showToolError(value) {
    const message = errorMessage(value);
    setProfileError(message);
    setNotice('');
    if (notify) notify(message, 'error');
  }

  async function submitTool(event) {
    event.preventDefault();
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      let successMessage = editingTool ? '工具已更新。' : '工具已保存，密钥不会在页面或接口响应中回显。';
      if (isEditingMcpGroup) {
        await updateMcpGroupConfig(editingMcpGroup);
        successMessage = `已更新「${editingMcpGroup.title}」下 ${editingMcpGroup.tools.length} 个 MCP 小工具。`;
      } else if (editingTool?.id) {
        const payload = toolFormPayload(form, { includeSecret: false });
        await updateToolConfig(editingTool.id, payload);
      } else if (form.type === 'mcp') {
        const discoveries = mcpDiscoveries.length > 0 ? mcpDiscoveries : await fetchRemoteMcpItems();
        setMcpDiscoveries(discoveries);
        if (!discoveries.length) {
          throw new Error('MCP 服务器已连接，但没有返回可用工具。');
        }
        const { payloads, skippedCount } = buildMcpDiscoveryPayloads(form, discoveries, tools);
        if (!payloads.length) {
          setNotice('该 MCP 服务下的工具已全部存在，无需重复添加。');
          if (notify) notify('MCP 工具已存在。');
          return;
        }
        await createToolConfig(payloads);
        successMessage = `已保存 ${payloads.length} 个 MCP 工具${skippedCount ? `，跳过 ${skippedCount} 个已存在工具` : ''}。`;
      } else {
        const payload = toolFormPayload(form, { includeSecret: true });
        await createToolConfig(payload);
      }
      setForm(createToolForm(form.type));
      setEditingTool(null);
      setEditingMcpGroup(null);
      setFormOpen(false);
      setNotice(successMessage);
      if (notify) notify((editingTool || isEditingMcpGroup) ? '工具保存成功。' : '工具创建成功。');
    } catch (err) {
      showToolError(err);
    } finally {
      setSaving(false);
    }
  }

  function mcpGroupConfigPatch(tool) {
    const timeout = Number(form.timeout_seconds);
    const transport = mcpTransportValue(form.mcp_transport, form.url);
    const isStdio = transport === 'stdio';
    const authType = isStdio ? 'none' : (form.auth_type || 'none');
    const mcpConfig = {
      transport,
      tool_name: tool?.mcp?.tool_name || tool?.name || '',
      input_schema: tool?.mcp?.input_schema || { type: 'object', properties: {} },
    };
    if (isStdio) {
      mcpConfig.command = String(form.mcp_command || '').trim();
      const argsStr = String(form.mcp_args || '').trim();
      mcpConfig.args = argsStr ? argsStr.split(/\s+/) : [];
      if (String(form.mcp_cwd || '').trim()) mcpConfig.cwd = String(form.mcp_cwd).trim();
      if (String(form.mcp_env || '').trim()) {
        try { mcpConfig.env = JSON.parse(String(form.mcp_env).trim()); } catch { /* ignore */ }
      }
    }
    return {
      url: isStdio ? '' : String(form.url || '').trim(),
      server_label: String(form.server_label || '').trim(),
      auth: {
        type: authType,
        header_name: ['bearer', 'header'].includes(authType) ? form.auth_header_name || 'Authorization' : null,
        query_name: authType === 'query' ? form.auth_query_name || null : null,
      },
      timeout_seconds: clampToolTimeout('mcp', timeout),
      enabled: Boolean(form.enabled),
      mcp: mcpConfig,
    };
  }

  async function updateMcpGroupConfig(entry) {
    await updateToolConfig(entry.tools.map((tool) => ({ id: tool.id, patch: mcpGroupConfigPatch(tool) })));
  }

  async function patchTool(tool, patch) {
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      await updateToolConfig(tool.id, patch);
    } catch (err) {
      showToolError(err);
    } finally {
      setSaving(false);
    }
  }

  async function patchMcpGroup(entry, patchForTool, successMessage) {
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      await updateToolConfig(entry.tools.map((tool) => ({
        id: tool.id,
        patch: typeof patchForTool === 'function' ? patchForTool(tool) : patchForTool,
      })));
      if (successMessage) setNotice(successMessage);
    } catch (err) {
      showToolError(err);
    } finally {
      setSaving(false);
    }
  }

  async function toggleMcpGroupEnabled(entry) {
    const nextEnabled = !entry.allEnabled;
    await patchMcpGroup(
      entry,
      { enabled: nextEnabled },
      `已${nextEnabled ? '启用' : '禁用'}「${entry.title}」下 ${entry.tools.length} 个 MCP 小工具。`
    );
  }

  async function replaceToolSecret(tool, nextSecret) {
    if (!String(nextSecret || '').trim()) {
      showToolError('Secret cannot be empty');
      return;
    }
    if (tool?.kind === 'mcp_group') {
      await patchMcpGroup(
        tool,
        (item) => ({
          auth: {
            type: item.auth?.type || item.auth_type || 'bearer',
            header_name: item.auth?.header_name || item.auth_header_name || 'Authorization',
            query_name: item.auth?.query_name || item.auth_query_name || null,
            secret: String(nextSecret).trim(),
          },
        }),
        `已替换「${tool.title}」下 ${tool.tools.length} 个 MCP 小工具的密钥。`
      );
      setSecretDialogTool(null);
      return;
    }
    await patchTool(tool, {
      auth: {
        type: tool.auth?.type || tool.auth_type || 'bearer',
        header_name: tool.auth?.header_name || tool.auth_header_name || 'Authorization',
        query_name: tool.auth?.query_name || tool.auth_query_name || null,
        secret: String(nextSecret).trim(),
      },
    });
    setSecretDialogTool(null);
    setNotice('工具密钥已替换，页面仅保留 has_secret 状态。');
  }

  function applyMcpDiscovery(item) {
    const nextMcp = item?.mcp || {};
    const patch = {
      name: item?.name || form.name,
      label: item?.label || form.label,
      description: item?.description || form.description,
      mcp_transport: nextMcp.transport || form.mcp_transport,
      mcp_tool_name: nextMcp.tool_name || item?.name || form.mcp_tool_name,
      mcp_input_schema: JSON.stringify(nextMcp.input_schema || { type: 'object', properties: {} }, null, 2),
      server_label: form.server_label || item?.server_label,
    };
    if (nextMcp.transport === 'stdio') {
      patch.mcp_command = nextMcp.command || form.mcp_command || '';
      patch.mcp_args = Array.isArray(nextMcp.args) ? nextMcp.args.join(' ') : (nextMcp.args || form.mcp_args || '');
      patch.mcp_cwd = nextMcp.cwd || form.mcp_cwd || '';
      patch.mcp_env = nextMcp.env && typeof nextMcp.env === 'object' ? JSON.stringify(nextMcp.env, null, 2) : (form.mcp_env || '');
    }
    updateToolForm(patch);
  }

  async function fetchRemoteMcpItems() {
    const transport = mcpTransportValue(form.mcp_transport, form.url);
    const payload = {
      tool_id: editingTool?.id || null,
      server_label: String(form.server_label || '').trim(),
      transport,
      timeout_seconds: clampToolTimeout('mcp', Number(form.timeout_seconds)),
    };
    if (transport === 'stdio') {
      const argsStr = String(form.mcp_args || '').trim();
      payload.mcp = {
        transport: 'stdio',
        command: String(form.mcp_command || '').trim(),
        args: argsStr ? argsStr.split(/\s+/) : [],
      };
      if (String(form.mcp_cwd || '').trim()) {
        payload.mcp.cwd = String(form.mcp_cwd).trim();
      }
      if (String(form.mcp_env || '').trim()) {
        try {
          payload.mcp.env = JSON.parse(String(form.mcp_env).trim());
        } catch {
          // ignore invalid JSON
        }
      }
    } else {
      const auth = {
        type: form.auth_type || 'none',
        header_name: ['bearer', 'header'].includes(form.auth_type) ? form.auth_header_name || 'Authorization' : null,
        query_name: form.auth_type === 'query' ? form.auth_query_name || null : null,
      };
      if (needsAuthSecret && String(form.auth_secret || '').trim()) {
        auth.secret = String(form.auth_secret).trim();
      }
      payload.url = String(form.url || '').trim();
      payload.auth = auth;
    }
    const data = await discoverMcpTools(payload);
    return data.items || [];
  }

  async function discoverRemoteMcpTools() {
    const transport = mcpTransportValue(form.mcp_transport, form.url);
    if (transport === 'stdio') {
      if (!String(form.mcp_command || '').trim()) {
        showToolError('请先填写 MCP 命令 (mcp.command)');
        return;
      }
    } else if (!form.url.trim()) {
      showToolError('请先填写 MCP 服务器 URL');
      return;
    }
    setDiscoveringMcp(true);
    setNotice('');
    setProfileError('');
    try {
      const items = await fetchRemoteMcpItems();
      setMcpDiscoveries(items);
      if (items.length > 0) {
        if (editingTool) {
          applyMcpDiscovery(items[0]);
          setNotice(`已读取 ${items.length} 个 MCP 工具定义。`);
        } else {
          const serverLabel = form.server_label || items[0]?.server_label;
          const transport = items[0]?.mcp?.transport || mcpTransportValue(form.mcp_transport, form.url);
          updateToolForm({
            server_label: serverLabel,
            mcp_transport: transport,
            label: form.label === MCP_TOOL_PRESET.label && serverLabel ? serverLabel : form.label,
          });
          setNotice(`已读取 ${items.length} 个 MCP 工具定义，保存时将默认添加全部。`);
        }
      } else {
        setNotice('MCP 服务器已连接，但没有返回可用工具。');
      }
    } catch (err) {
      showToolError(err);
    } finally {
      setDiscoveringMcp(false);
    }
  }

  function openToolTest(tool) {
    setTestingTool(tool);
    setProfileError('');
    setNotice('');
    setTestInputById((items) => ({ ...items, [tool.id]: items[tool.id] || defaultToolTestInput(tool) }));
  }

  function closeToolTest() {
    if (testingId) return;
    setTestingTool(null);
  }

  async function deleteTool(tool) {
    const confirmed = await requestDeleteConfirm({
      title: '删除工具',
      message: `删除「${tool.label || tool.name}」？`,
      detail: '如果已有智能体绑定，后端会按约束拒绝。',
      confirmLabel: '删除工具',
    });
    if (!confirmed) return;
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      await deleteToolConfig(tool.id);
    } catch (err) {
      showToolError(err);
    } finally {
      setSaving(false);
    }
  }

  async function deleteMcpGroup(entry) {
    const confirmed = await requestDeleteConfirm({
      title: '删除 MCP 工具组',
      message: `删除「${entry.title}」下 ${entry.tools.length} 个 MCP 小工具？`,
      detail: '如果已有智能体绑定其中任一工具，后端会按约束拒绝删除。',
      confirmLabel: '删除全部',
    });
    if (!confirmed) return;
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      await deleteToolConfig(entry.toolIds);
      setNotice(`已删除「${entry.title}」下 ${entry.tools.length} 个 MCP 小工具。`);
    } catch (err) {
      showToolError(err);
    } finally {
      setSaving(false);
    }
  }

  async function testTool(tool) {
    setTestingId(tool.id);
    setNotice('');
    setProfileError('');
    try {
      const payload = {
        input: parseJsonField(testInputById[tool.id] || '{}', 'test input'),
        body: parseOptionalJsonField(testBodyById[tool.id] || '', 'test body'),
      };
      const result = await testToolConfig(tool.id, payload);
      setTestResults((items) => ({ ...items, [tool.id]: result }));
    } catch (err) {
      showToolError(err);
    } finally {
      setTestingId(null);
    }
  }

  return (
    <section className="plain-panel tools-panel">
      <div className="panel-title-row">
        <div>
          <h3>工具</h3>
          <p>HTTP 工具按 Day2 契约提交 method、url、schema、auth、response_path 和 timeout_seconds；MCP 工具可通过 Streamable HTTP 或 SSE 连接远端 MCP Server。</p>
        </div>
        <div className="panel-actions">
          <span className="soft-pill">{toolDisplayEntries.length} 个工具项</span>
          <button className="primary-model-action" type="button" onClick={() => openToolForm('http')}><Plus size={15} />新增工具</button>
        </div>
      </div>

      {formOpen && (
        <div className="profile-dialog-backdrop">
          <section className="resource-form-dialog tool-config-dialog" role="dialog" aria-modal="true" aria-label={editingTool ? '编辑工具' : '新增工具'} onClick={(event) => event.stopPropagation()}>
            <button className="profile-dialog-close" type="button" title="关闭" aria-label="关闭工具表单" onClick={closeToolForm} disabled={saving}>
              <X size={16} />
            </button>
            <header className="model-dialog-heading">
              <h3>{isEditingMcpGroup ? '编辑 MCP 工具组' : editingTool ? '编辑工具' : '新增工具'}</h3>
              <p>{isEditingMcpGroup ? '修改该 MCP 服务下所有小工具共享的 URL、超时、鉴权和启用状态。' : editingTool ? '修改工具基础配置、Schema、超时和启用状态。已保存的密钥不会回显，需要单独替换。' : '配置可绑定到智能体的 HTTP、MCP 或内置联网搜索工具。密钥只提交一次，保存后不回显。'}</p>
            </header>
            <form className="tool-form dialog-form" onSubmit={submitTool}>
              <div className="tool-modal-body">
                {/* ── Left: type selector ── */}
                <div className="tool-type-panel">
                  <div className="tool-type-switch">
                    <button type="button" disabled={!!editingTool || isEditingMcpGroup} className={form.type === 'http' ? 'active' : ''} onClick={() => switchType('http')}>HTTP</button>
                    <button type="button" disabled={!!editingTool || isEditingMcpGroup} className={form.type === 'mcp' ? 'active' : ''} onClick={() => switchType('mcp')}>MCP</button>
                    <button type="button" disabled={!!editingTool || isEditingMcpGroup} className={form.type === 'builtin_search' ? 'active' : ''} onClick={() => switchType('builtin_search')}>builtin_search</button>
                  </div>
                </div>

                {/* ── Right: config form grid ── */}
                <div className="tool-config-form">
                  {/* ── Non-MCP: name + label ── */}
                  {!isMcpForm && !isEditingMcpGroup && (
                    <>
                      <div className="form-field">
                        <label>name</label>
                        <input value={form.name} onChange={(event) => updateToolForm({ name: event.target.value })} placeholder="weather_lookup" autoFocus />
                      </div>
                      <div className="form-field">
                        <label>label</label>
                        <input value={form.label} onChange={(event) => updateToolForm({ label: event.target.value })} placeholder="Weather lookup" />
                      </div>
                    </>
                  )}

                  {/* ── MCP: server_label + transport ── */}
                  {isMcpForm && (
                    <>
                      <div className="form-field">
                        <label>server_label</label>
                        <input value={form.server_label} onChange={(event) => updateToolForm({ server_label: event.target.value })} placeholder="请输入服务标签" autoFocus />
                      </div>
                      <div className="form-field">
                        <label>mcp.transport</label>
                        <select value={mcpTransportValue(form.mcp_transport, form.url)} onChange={(event) => updateToolForm({ mcp_transport: event.target.value })}>
                          <option value="streamable_http">streamable_http</option>
                          <option value="sse">sse</option>
                          <option value="stdio">stdio</option>
                        </select>
                      </div>
                    </>
                  )}

                  {/* ── HTTP: method + response_path ── */}
                  {isHttpForm && (
                    <>
                      <div className="form-field">
                        <label>method</label>
                        <select value={form.method} onChange={(event) => updateToolForm({ method: event.target.value })}>
                          {['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].map((method) => <option key={method} value={method}>{method}</option>)}
                        </select>
                      </div>
                      <div className="form-field">
                        <label>response_path</label>
                        <input value={form.response_path} onChange={(event) => updateToolForm({ response_path: event.target.value })} placeholder="$" />
                      </div>
                    </>
                  )}

                  {/* ── Remote MCP / HTTP: url (full width) ── */}
                  {(isHttpForm || isMcpRemote) && (
                    <div className="form-field full">
                      <label>url</label>
                      <input
                        value={form.url}
                        onChange={(event) => {
                          const nextUrl = event.target.value;
                          const inferredTransport = mcpTransportValue('', nextUrl);
                          updateToolForm({
                            url: nextUrl,
                            ...(isMcpForm && inferredTransport === 'sse' ? { mcp_transport: inferredTransport } : {}),
                          });
                        }}
                        placeholder={isMcpForm ? 'https://dashscope.aliyuncs.com/api/v1/mcps/WebParser/sse' : 'https://api.example.com/weather'}
                      />
                    </div>
                  )}

                  {/* ── stdio: command + timeout_seconds ── */}
                  {isMcpStdio && (
                    <>
                      <div className="form-field">
                        <label>mcp.command</label>
                        <input
                          value={form.mcp_command || ''}
                          onChange={(event) => updateToolForm({ mcp_command: event.target.value })}
                          placeholder="npx"
                          autoFocus
                        />
                        <p className="field-help">启动 MCP 服务器的可执行命令，例如 npx、node、python</p>
                      </div>
                      <div className="form-field">
                        <label>timeout_seconds</label>
                        <input type="number" min="1" max={toolTimeoutMax('mcp')} value={form.timeout_seconds} onChange={(event) => updateToolForm({ timeout_seconds: event.target.value })} />
                        <p className="field-help">MCP 工具默认 30s，慢速网页解析可调高到 120s</p>
                      </div>
                    </>
                  )}

                  {/* ── stdio: mcp.args (full width) ── */}
                  {isMcpStdio && (
                    <div className="form-field full">
                      <label>mcp.args</label>
                      <textarea
                        value={form.mcp_args || ''}
                        onChange={(event) => updateToolForm({ mcp_args: event.target.value })}
                        placeholder="@playwright/mcp"
                        rows={2}
                      />
                      <p className="field-help">参数以空格分隔，例如：@playwright/mcp --allowed-hosts=example.com</p>
                    </div>
                  )}

                  {/* ── stdio: mcp.cwd (full width) ── */}
                  {isMcpStdio && (
                    <div className="form-field full">
                      <label>mcp.cwd (可选)</label>
                      <input
                        value={form.mcp_cwd || ''}
                        onChange={(event) => updateToolForm({ mcp_cwd: event.target.value })}
                        placeholder="C:\Users\Ivy\project"
                      />
                      <p className="field-help">MCP 进程的工作目录，留空使用后端当前工作目录</p>
                    </div>
                  )}

                  {/* ── stdio: mcp.env (full width) ── */}
                  {isMcpStdio && (
                    <div className="form-field full">
                      <label>mcp.env (可选)</label>
                      <textarea
                        value={form.mcp_env || ''}
                        onChange={(event) => updateToolForm({ mcp_env: event.target.value })}
                        placeholder='{"KEY":"value","DEBUG":"pw:api"}'
                        rows={3}
                      />
                      <p className="field-help">JSON 格式的环境变量，例如 {'{'}"DEBUG": "pw:api"{'}'}</p>
                    </div>
                  )}

                  {/* ── timeout_seconds for non-stdio ── */}
                  {!isMcpStdio && (isHttpForm || isMcpForm) && (
                    <div className="form-field">
                      <label>timeout_seconds</label>
                      <input type="number" min="1" max={toolTimeoutMax(isMcpForm ? 'mcp' : 'http')} value={form.timeout_seconds} onChange={(event) => updateToolForm({ timeout_seconds: event.target.value })} />
                      {isMcpForm && <p className="field-help">MCP 工具默认 30s</p>}
                    </div>
                  )}

                  {/* ── MCP editing: tool_name + input_schema ── */}
                  {isMcpForm && editingTool && !isEditingMcpGroup && (
                    <>
                      <div className="form-field">
                        <label>mcp.tool_name</label>
                        <input value={form.mcp_tool_name} onChange={(event) => updateToolForm({ mcp_tool_name: event.target.value })} placeholder="get_weather" />
                      </div>
                      <div className="form-field full">
                        <label>mcp.input_schema</label>
                        <textarea value={form.mcp_input_schema} onChange={(event) => updateToolForm({ mcp_input_schema: event.target.value })} placeholder='{"type":"object","properties":{}}' rows={3} />
                      </div>
                    </>
                  )}

                  {/* ── Non-MCP: description (full width) ── */}
                  {!isEditingMcpGroup && !isMcpForm && (
                    <div className="form-field full">
                      <label>description</label>
                      <textarea value={form.description} onChange={(event) => updateToolForm({ description: event.target.value })} placeholder="工具能力说明" rows={3} />
                    </div>
                  )}

                  {/* ── Auth section for remote MCP / HTTP ── */}
                  {(isHttpForm || isMcpRemote) && (
                    <>
                      <div className="form-field">
                        <label>auth.type</label>
                        <select value={form.auth_type} onChange={(event) => updateToolForm({ auth_type: event.target.value })}>
                          <option value="none">none</option>
                          <option value="bearer">bearer</option>
                          <option value="header">header</option>
                          <option value="query">query</option>
                        </select>
                      </div>
                      {needsAuthHeader && (
                        <div className="form-field">
                          <label>auth.header_name</label>
                          <input value={form.auth_header_name} onChange={(event) => updateToolForm({ auth_header_name: event.target.value })} placeholder="Authorization" />
                        </div>
                      )}
                      {needsAuthQuery && (
                        <div className="form-field">
                          <label>auth.query_name</label>
                          <input value={form.auth_query_name} onChange={(event) => updateToolForm({ auth_query_name: event.target.value })} placeholder="api_key" />
                        </div>
                      )}
                      {!editingTool && !isEditingMcpGroup && needsAuthSecret && (
                        <div className="form-field">
                          <label>auth.secret</label>
                          <input type="password" value={form.auth_secret} onChange={(event) => updateToolForm({ auth_secret: event.target.value })} placeholder="只提交一次，不回显" autoComplete="off" />
                        </div>
                      )}
                      {editingTool && needsAuthSecret && (
                        <div className="form-field full">
                          <div className="tool-edit-secret-note">
                            <strong>密钥不在编辑表单中回显</strong>
                            <span>需要换密钥时，在列表里点击"替换 Secret"。</span>
                          </div>
                        </div>
                      )}
                      {isEditingMcpGroup && needsAuthSecret && (
                        <div className="form-field full">
                          <div className="tool-edit-secret-note">
                            <strong>密钥不在编辑表单中回显</strong>
                            <span>需要换密钥时，在 MCP 父项点击"更新密钥"。</span>
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {/* ── MCP discovery section ── */}
                  {isMcpForm && !isEditingMcpGroup && (
                    <div className="tool-mcp-discovery">
                      <div className="tool-mcp-discovery-header">
                        <strong>MCP 工具定义</strong>
                        <button
                          type="button"
                          onClick={() => discoverRemoteMcpTools()}
                          disabled={discoveringMcp || (isMcpStdio ? !String(form.mcp_command || '').trim() : !form.url.trim())}
                          style={toolAccentButtonStyle}
                        >
                          <Sparkles size={13} />
                          {discoveringMcp ? '读取中...' : isMcpStdio ? '读取本地工具' : '读取远端工具'}
                        </button>
                      </div>
                      {mcpDiscoveries.length > 0 && (
                        <div style={{ display: 'grid', gap: '8px' }}>
                          {mcpDiscoveries.map((item, index) => (
                            <div
                              key={`${item.name || item.label}-${index}`}
                              role={editingTool ? 'button' : undefined}
                              tabIndex={editingTool ? 0 : undefined}
                              onClick={editingTool ? () => applyMcpDiscovery(item) : undefined}
                              onKeyDown={editingTool ? (event) => {
                                if (event.key === 'Enter' || event.key === ' ') {
                                  event.preventDefault();
                                  applyMcpDiscovery(item);
                                }
                              } : undefined}
                              style={{
                                ...toolButtonStyle,
                                justifyContent: 'space-between',
                                textAlign: 'left',
                                cursor: editingTool ? 'pointer' : 'default',
                                background: editingTool ? toolButtonStyle.background : toolTheme.rowBg,
                              }}
                            >
                              <span style={{ display: 'grid', gap: '2px' }}>
                                <strong>{item.label || item.name}</strong>
                                <small style={{ color: toolTheme.text }}>{item.description || '无描述'}</small>
                              </span>
                              <span style={{ color: toolTheme.accentText }}>{editingTool ? '使用' : '将保存'}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── Footer: enable + security note + actions ── */}
                  <div className="tool-config-footer">
                    <div className="model-checks" style={{ flex: '0 0 auto' }}>
                      <label><input type="checkbox" checked={form.enabled} onChange={(event) => updateToolForm({ enabled: event.target.checked })} />启用</label>
                    </div>
                    <div className="tool-security-note" style={{ fontSize: '12px', color: '#6b7280', minWidth: 0 }}>
                      {isHttpForm && 'HTTP 工具必须使用 https://，后端负责阻断 localhost、私网和 metadata 地址。'}
                      {isMcpRemote && 'MCP 支持 Streamable HTTP 和 SSE；http:// 仅允许本机 localhost，公网地址请使用 https://。'}
                      {isMcpStdio && 'stdio 传输在本地启动子进程通信（如 npx @playwright/mcp），默认无网络鉴权。'}
                    </div>
                  </div>

                  {/* ── Non-MCP schema editors (full width) ── */}
                  {!isMcpForm && (
                    <div className="form-field full" style={{ gap: '10px' }}>
                      {isHttpForm && (
                        <ParamTableEditor
                          isDarkTheme={isDarkTheme}
                          label="Headers 参数结构定义 (headers_schema)"
                          params={headersParams}
                          onChange={(next) => {
                            setHeadersParams(next);
                            updateToolForm({ headers_schema: paramsToSchema(next) });
                          }}
                        />
                      )}
                      <ParamTableEditor
                        isDarkTheme={isDarkTheme}
                        label={isHttpForm ? "Query 请求参数定义 (query_schema)" : "联网搜索参数定义 (search_query_schema)"}
                        params={queryParams}
                        onChange={(next) => {
                          setQueryParams(next);
                          updateToolForm({ query_schema: paramsToSchema(next) });
                        }}
                      />
                      {needsBodySchema && (
                        <ParamTableEditor
                          isDarkTheme={isDarkTheme}
                          label="Body 请求体定义 (body_schema)"
                          params={bodyParams}
                          onChange={(next) => {
                            setBodyParams(next);
                            updateToolForm({ body_schema: paramsToSchema(next) });
                          }}
                        />
                      )}
                    </div>
                  )}
                </div>
              </div>

              <footer className="dialog-actions form-actions-right">
                <button type="button" onClick={closeToolForm} disabled={saving}>取消</button>
                <button
                  className="primary-model-action"
                  type="submit"
                  disabled={
                    saving
                    || (!isEditingMcpGroup && !isMcpForm && (!form.name.trim() || !form.label.trim()))
                    || (isMcpStdio && !String(form.mcp_command || '').trim())
                    || (isMcpRemote && !form.url.trim())
                    || (isHttpForm && !form.url.trim())
                    || (isMcpForm && editingTool && !form.mcp_tool_name.trim())
                  }
                >
                  <Plus size={15} />{saving ? '保存中...' : isEditingMcpGroup ? '保存 MCP 工具组' : editingTool ? '保存修改' : isCreatingMcpGroup ? '保存全部 MCP 工具' : '保存工具'}
                </button>
              </footer>
            </form>
          </section>
        </div>
      )}

      {notice && <p className="model-row-warning">{notice}</p>}

      <div className="tool-list" style={{ gap: '12px' }}>
        {toolDisplayEntries.map((entry) => {
          if (entry.kind === 'mcp_group') {
            const expanded = Boolean(expandedMcpGroups[entry.key]);
            return (
              <article
                className="tool-list-row mcp-tool-group-row"
                key={entry.id}
                style={{
                  transition: 'all 0.2s ease',
                  border: `1px solid ${toolTheme.rowBorder}`,
                  borderRadius: '12px',
                  padding: '16px',
                  background: toolTheme.rowBg,
                  boxShadow: toolTheme.rowShadow,
                  cursor: 'pointer',
                }}
                onClick={() => setExpandedMcpGroups((items) => ({ ...items, [entry.key]: !items[entry.key] }))}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = toolTheme.rowHoverBorder;
                  e.currentTarget.style.boxShadow = toolTheme.rowHoverShadow;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = toolTheme.rowBorder;
                  e.currentTarget.style.boxShadow = toolTheme.rowShadow;
                }}
              >
                <div className="tool-row-main" style={{ display: 'flex', gap: '14px', alignItems: 'center', minWidth: 0, flex: 1 }}>
                  <span
                    className="tool-kind mcp"
                    style={{
                      ...toolTheme.typeMcp,
                      padding: '4px 10px',
                      borderRadius: '20px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}
                  >
                    MCP
                  </span>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <strong style={{ fontSize: '14px', fontWeight: 700, color: toolTheme.title }}>{entry.title}</strong>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          padding: '2px 8px',
                          borderRadius: '12px',
                          fontSize: '11px',
                          background: entry.enabled ? toolTheme.successBg : toolTheme.codeBg,
                          color: entry.enabled ? toolTheme.successText : toolTheme.muted,
                          fontWeight: 600,
                        }}
                      >
                        <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: entry.enabled ? toolTheme.successText : toolTheme.muted }} />
                        {entry.enabled ? '已启用' : '已禁用'}
                      </span>
                    </div>
                    <small style={{ display: 'block', marginTop: '4px', fontSize: '12px', color: toolTheme.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={entry.url}>
                      <code style={{ background: toolTheme.codeBg, padding: '2px 6px', borderRadius: '4px', marginRight: '6px', fontSize: '11px', fontFamily: 'monospace', color: toolTheme.codeText }}>
                        MCP Server
                      </code>
                      {entry.description}
                    </small>
                  </div>
                </div>

                <div className="tool-row-meta" style={{ display: 'flex', gap: '12px', alignItems: 'center', color: toolTheme.text, fontSize: '12px' }}>
                  <span style={{ background: toolTheme.methodBg, padding: '3px 8px', borderRadius: '6px', fontWeight: 'bold', color: toolTheme.methodText }}>
                    MCP
                  </span>
                  <span style={{ color: toolTheme.muted }}>|</span>
                  <span>{entry.tools.length} 个小工具</span>
                  <span style={{ color: toolTheme.muted }}>|</span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: entry.hasSecret ? toolTheme.warnText : toolTheme.text }}>
                    <KeyRound size={12} />
                    {entry.hasSecret ? '已配密钥' : '免鉴权'}
                  </span>
                </div>

                <div className="tool-row-actions" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  {entry.tools.every((tool) => isUserTool(tool)) && (
                    <>
                      <button
                        type="button"
                        disabled={saving}
                        onClick={(event) => {
                          event.stopPropagation();
                          openEditMcpGroup(entry);
                        }}
                        style={toolButtonStyle}
                      >
                        <SquarePen size={13} />
                        编辑
                      </button>
                      <button
                        type="button"
                        disabled={saving}
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleMcpGroupEnabled(entry);
                        }}
                        style={{
                          ...toolButtonStyle,
                          color: entry.allEnabled ? toolTheme.warnText : toolTheme.successText,
                        }}
                      >
                        <Shield size={13} />
                        {entry.allEnabled ? '禁用' : '启用'}
                      </button>
                      <button
                        type="button"
                        disabled={saving}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSecretDialogTool(entry);
                        }}
                        style={toolButtonStyle}
                      >
                        <KeyRound size={13} />
                        更新密钥
                      </button>
                      <button
                        className="model-delete-button"
                        type="button"
                        disabled={saving}
                        onClick={(event) => {
                          event.stopPropagation();
                          deleteMcpGroup(entry);
                        }}
                        style={{
                          ...toolButtonStyle,
                          background: toolTheme.dangerBg,
                          border: `1px solid ${toolTheme.dangerBorder}`,
                          color: toolTheme.dangerText,
                        }}
                      >
                        <Trash2 size={13} />
                        删除
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      setExpandedMcpGroups((items) => ({ ...items, [entry.key]: !items[entry.key] }));
                    }}
                    style={toolAccentButtonStyle}
                  >
                    <ChevronRight size={13} style={{ transform: expanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s ease' }} />
                    {expanded ? '收起小工具' : '查看小工具'}
                  </button>
                </div>

                {expanded && (
                  <div className="mcp-subtool-list" onClick={(event) => event.stopPropagation()}>
                    {entry.tools.map((tool) => {
                      const enabled = tool.enabled !== false;
                      return (
                        <div className="mcp-subtool-row" key={tool.id}>
                          <div className="mcp-subtool-main">
                            <strong>{tool.label || tool.name}</strong>
                            <small title={tool.description}>
                              <code>{tool.name}</code>
                              {tool.description || '暂无详细说明'}
                            </small>
                          </div>
                          <span className="mcp-subtool-remote" title={tool.url || ''}>{tool.mcp?.tool_name || 'remote tool'}</span>
                          <div className="mcp-subtool-actions">
                            <button type="button" disabled={testingId === tool.id} onClick={() => openToolTest(tool)} style={toolAccentButtonStyle}>
                              <Sparkles size={13} />
                              {testingId === tool.id ? '测试中...' : '测试'}
                            </button>
                            {!isUserTool(tool) && (
                              <button type="button" disabled={saving} onClick={() => openCopyTool(tool)} style={toolAccentButtonStyle}>
                                <Layers size={13} />
                                复制为自定义
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </article>
            );
          }
          const tool = entry.tool;
          const type = toolType(tool);
          const isHttp = type === 'http';
          const isMcp = type === 'mcp';
          const isSearch = type === 'builtin_search';
          const isBuiltin = type === 'builtin';

          let typeStyle = toolTheme.typeNeutral;
          if (isSearch) typeStyle = toolTheme.typeSearch;
          else if (isBuiltin) typeStyle = toolTheme.typeBuiltin;
          else if (isHttp) typeStyle = toolTheme.typeHttp;
          else if (isMcp) typeStyle = toolTheme.typeMcp;

          const enabled = tool.enabled !== false;
          
          return (
            <article 
              className="tool-list-row" 
              key={tool.id}
              style={{
                transition: 'all 0.2s ease',
                border: `1px solid ${toolTheme.rowBorder}`,
                borderRadius: '12px',
                padding: '16px',
                background: toolTheme.rowBg,
                boxShadow: toolTheme.rowShadow,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = toolTheme.rowHoverBorder;
                e.currentTarget.style.boxShadow = toolTheme.rowHoverShadow;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = toolTheme.rowBorder;
                e.currentTarget.style.boxShadow = toolTheme.rowShadow;
              }}
            >
              <div className="tool-row-main" style={{ display: 'flex', gap: '14px', alignItems: 'center', minWidth: 0, flex: 1 }}>
                <span 
                  className={`tool-kind ${type}`} 
                  style={{
                    ...typeStyle,
                    padding: '4px 10px',
                    borderRadius: '20px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}
                >
                  {type}
                </span>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <strong style={{ fontSize: '14px', fontWeight: 700, color: toolTheme.title }}>{tool.label || tool.name}</strong>
                    <span 
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '2px 8px',
                        borderRadius: '12px',
                        fontSize: '11px',
                        background: enabled ? toolTheme.successBg : toolTheme.codeBg,
                        color: enabled ? toolTheme.successText : toolTheme.muted,
                        fontWeight: 600,
                      }}
                    >
                      <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: enabled ? toolTheme.successText : toolTheme.muted }} />
                      {enabled ? '已启用' : '已禁用'}
                    </span>
                  </div>
                  <small style={{ display: 'block', marginTop: '4px', fontSize: '12px', color: toolTheme.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={tool.description}>
                    <code style={{ background: toolTheme.codeBg, padding: '2px 6px', borderRadius: '4px', marginRight: '6px', fontSize: '11px', fontFamily: 'monospace', color: toolTheme.codeText }}>
                      {tool.name}
                    </code>
                    {tool.description || '暂无详细说明'}
                  </small>
                </div>
              </div>

              <div className="tool-row-meta" style={{ display: 'flex', gap: '12px', alignItems: 'center', color: toolTheme.text, fontSize: '12px' }}>
                <span style={{ background: toolTheme.methodBg, padding: '3px 8px', borderRadius: '6px', fontWeight: 'bold', color: toolTheme.methodText }}>
                  {isMcp ? 'MCP' : (tool.method || 'GET')}
                </span>
                <span style={{ color: toolTheme.muted }}>|</span>
                <span>超时 {tool.timeout_seconds || (isMcp ? 30 : 10)}s</span>
                {isMcp && (
                  <>
                    <span style={{ color: toolTheme.muted }}>|</span>
                    <span title={tool.url || ''}>{tool.mcp?.tool_name || 'remote tool'}</span>
                  </>
                )}
                <span style={{ color: toolTheme.muted }}>|</span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: toolHasSecret(tool) ? toolTheme.warnText : toolTheme.text }}>
                  <KeyRound size={12} />
                  {toolHasSecret(tool) ? '已配密钥' : '免鉴权'}
                </span>
              </div>

              <div className="tool-row-actions" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <button 
                  type="button" 
                  disabled={testingId === tool.id} 
                  onClick={() => openToolTest(tool)}
                  style={toolAccentButtonStyle}
                >
                  <Sparkles size={13} />
                  {testingId === tool.id ? '测试中...' : '测试'}
                </button>
                {isUserTool(tool) ? (
                  <>
                    <button 
                      type="button" 
                      disabled={saving} 
                      onClick={() => openEditTool(tool)}
                      style={toolButtonStyle}
                    >
                      <SquarePen size={13} />
                      编辑
                    </button>
                    <button 
                      type="button" 
                      disabled={saving} 
                      onClick={() => patchTool(tool, { enabled: tool.enabled === false })}
                      style={{
                        ...toolButtonStyle,
                        color: enabled ? toolTheme.warnText : toolTheme.successText,
                      }}
                    >
                      <Shield size={13} />
                      {enabled ? '禁用' : '启用'}
                    </button>
                    {(isHttp || isMcp) && (
                      <button 
                        type="button" 
                        disabled={saving} 
                        onClick={() => setSecretDialogTool(tool)}
                        style={toolButtonStyle}
                      >
                        <KeyRound size={13} />
                        更新密钥
                      </button>
                    )}
                    <button 
                      className="model-delete-button" 
                      type="button" 
                      disabled={saving} 
                      onClick={() => deleteTool(tool)}
                      style={{
                        ...toolButtonStyle,
                        background: toolTheme.dangerBg,
                        border: `1px solid ${toolTheme.dangerBorder}`,
                        color: toolTheme.dangerText,
                      }}
                    >
                      <Trash2 size={13} />
                      删除
                    </button>
                  </>
                ) : (
                  <button 
                    type="button" 
                    disabled={saving} 
                    onClick={() => openCopyTool(tool)}
                    style={toolAccentButtonStyle}
                  >
                    <Layers size={13} />
                    复制为自定义
                  </button>
                )}
              </div>
            </article>
          );
        })}
        {toolDisplayEntries.length === 0 && (
          <p className="muted" style={{ padding: '24px', textAlign: 'center', background: toolTheme.emptyBg, borderRadius: '12px', border: `1px dashed ${toolTheme.emptyBorder}`, color: toolTheme.text }}>
            当前没有可用工具。保存 builtin_search、HTTP 或 MCP 工具后即可在 Builder 中绑定。
          </p>
        )}
      </div>
      {testingTool && (
        <div className="profile-dialog-backdrop">
          <section className="resource-form-dialog tool-test-dialog" role="dialog" aria-modal="true" aria-label="测试工具" onClick={(event) => event.stopPropagation()}>
            <button className="profile-dialog-close" type="button" title="关闭" aria-label="关闭工具测试" onClick={closeToolTest} disabled={!!testingId}>
              <X size={16} />
            </button>
            <header className="model-dialog-heading">
              <h3>测试工具</h3>
              <p>{testingTool.label || testingTool.name}</p>
            </header>
            <div className="tool-test-box">
              <label className="field-stack">
                <span>test input</span>
                <textarea value={testInputById[testingTool.id] || defaultToolTestInput(testingTool)} onChange={(event) => setTestInputById((items) => ({ ...items, [testingTool.id]: event.target.value }))} />
              </label>
              <label className="field-stack">
                <span>test body</span>
                 <textarea value={testBodyById[testingTool.id] || ''} onChange={(event) => setTestBodyById((items) => ({ ...items, [testingTool.id]: event.target.value }))} placeholder="可空，JSON body" />
              </label>
              {testResults[testingTool.id] && <ToolTestResult result={testResults[testingTool.id]} />}
              <footer className="dialog-actions">
                <button type="button" onClick={closeToolTest} disabled={!!testingId}>关闭</button>
                <button className="primary-model-action" type="button" disabled={testingId === testingTool.id} onClick={() => testTool(testingTool)}>
                  <Check size={15} />{testingId === testingTool.id ? '测试中...' : '运行测试'}
                </button>
              </footer>
            </div>
          </section>
        </div>
      )}
      {secretDialogTool && (
        <SecretInputDialog
          label="Secret"
          message={`替换「${secretDialogTool.label || secretDialogTool.name}」的密钥。新密钥只提交一次，保存后不回显。`}
          onCancel={() => !saving && setSecretDialogTool(null)}
          onSubmit={(value) => replaceToolSecret(secretDialogTool, value).catch((err) => showToolError(err))}
          saving={saving}
          submitLabel="替换 Secret"
          title={secretDialogTool.kind === 'mcp_group' ? '替换 MCP 工具组 Secret' : '替换工具 Secret'}
        />
      )}
    </section>
  );
}

function ToolTestResult({ result }) {
  return (
    <div className={result.ok ? 'tool-test-result ok' : 'tool-test-result'}>
      <strong>{result.ok ? '测试成功' : '测试失败'}</strong>
      <span>{result.tool_type || 'tool'} · {result.status_code || '-'} · {result.latency_ms ?? '-'}ms · {result.content_type || '-'}</span>
      {!result.ok && result.hint && <small className="tool-test-hint">{result.hint}</small>}
      <pre>{result.result_preview || result.error || result.message || JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}

const USER_MODEL_PRESETS = [
  {
    id: 'qwen',
    label: 'DashScope / Qwen',
    modelHint: 'qwen-plus',
    description: '阿里云百炼，推荐默认入口，支持通义千问；图片能力由测试自动检测，RAG/Embedding 由后端默认配置提供。',
    values: {
      display_name: 'Qwen Plus',
      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      chat_model: 'qwen-plus',
      supports_image: false,
      supports_document: true,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '131072',
      default_temperature: '0.4',
    },
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    modelHint: 'deepseek-v4-flash',
    description: 'DeepSeek 官方兼容接口。这里只配置主聊天模型；知识库检索使用后端默认 Embedding。',
    values: {
      display_name: 'DeepSeek V4 Flash',
      base_url: 'https://api.deepseek.com',
      chat_model: 'deepseek-v4-flash',
      supports_image: false,
      supports_document: false,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '64000',
      default_temperature: '0.6',
    },
  },
  {
    id: 'kimi',
    label: 'Kimi / Moonshot',
    modelHint: 'moonshot-v1-8k',
    description: '月之暗面兼容接口。模型名可按控制台改成最新 Kimi 模型。',
    values: {
      display_name: 'Kimi Moonshot',
      base_url: 'https://api.moonshot.cn/v1',
      chat_model: 'moonshot-v1-8k',
      supports_image: false,
      supports_document: false,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '32768',
      default_temperature: '0.4',
    },
  },
  {
    id: 'zhipu',
    label: '智谱 GLM',
    modelHint: 'glm-4-flash',
    description: '智谱 BigModel 兼容接口。适合 GLM 系列文本模型。',
    values: {
      display_name: 'GLM',
      base_url: 'https://open.bigmodel.cn/api/paas/v4',
      chat_model: 'glm-4-flash',
      supports_image: false,
      supports_document: true,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '128000',
      default_temperature: '0.4',
    },
  },
  {
    id: 'volcengine',
    label: '火山方舟 / 豆包',
    modelHint: 'ep-xxxxxxxx',
    description: '火山方舟兼容接口，model 通常填写控制台创建的 endpoint id。',
    values: {
      display_name: 'Doubao Ark',
      base_url: 'https://ark.cn-beijing.volces.com/api/v3',
      chat_model: 'ep-xxxxxxxx',
      supports_image: false,
      supports_document: true,
      supports_reasoning: false,
      reasoning_type: 'none',
      reasoning_label: '不支持',
      max_context: '128000',
      default_temperature: '0.4',
    },
  },
  {
    id: 'qianfan',
    label: '百度千帆 / ERNIE',
    modelHint: 'ernie-4.5-turbo-128k',
    description: '百度智能云千帆兼容接口。保存前按控制台可用模型名调整。',
    values: {
      display_name: 'ERNIE',
      base_url: 'https://qianfan.baidubce.com/v2',
      chat_model: 'ernie-4.5-turbo-128k',
      supports_image: false,
      supports_document: true,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '128000',
      default_temperature: '0.4',
    },
  },
  {
    id: 'siliconflow',
    label: '硅基流动',
    modelHint: 'Qwen/Qwen3-32B',
    description: '聚合国产和开源模型，模型名建议从控制台复制。',
    values: {
      display_name: 'SiliconFlow Qwen',
      base_url: 'https://api.siliconflow.cn/v1',
      chat_model: 'Qwen/Qwen3-32B',
      supports_image: false,
      supports_document: true,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '32768',
      default_temperature: '0.5',
    },
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    modelHint: 'deepseek/deepseek-chat',
    description: '海外聚合网关。建议使用非 GPT 模型作为默认配置。',
    values: {
      display_name: 'OpenRouter DeepSeek',
      base_url: 'https://openrouter.ai/api/v1',
      chat_model: 'deepseek/deepseek-chat',
      supports_image: false,
      supports_document: false,
      supports_reasoning: true,
      reasoning_type: 'prompt',
      reasoning_label: '提示词增强',
      max_context: '64000',
      default_temperature: '0.6',
    },
  },
  {
    id: 'ollama',
    label: 'Ollama 本机',
    modelHint: 'qwen2.5:7b',
    description: '本机开发可用。Docker 内运行 API 时，通常要把 127.0.0.1 改成 host.docker.internal。',
    values: {
      display_name: 'Local Ollama',
      base_url: 'http://127.0.0.1:11434/v1',
      chat_model: 'qwen2.5:7b',
      supports_image: false,
      supports_document: true,
      supports_reasoning: false,
      reasoning_type: 'none',
      reasoning_label: '不支持',
      max_context: '32768',
      default_temperature: '0.4',
    },
  },
  {
    id: 'custom',
    label: '自定义兼容网关',
    modelHint: '填写控制台模型名',
    description: '适用于私有部署、代理网关、One API、LiteLLM、New API 或 OpenAI-compatible 服务。',
    values: {
      display_name: 'Custom Model',
      base_url: '',
      chat_model: '',
      supports_image: false,
      supports_document: false,
      supports_reasoning: false,
      reasoning_type: 'none',
      reasoning_label: '不支持',
      max_context: '32768',
      default_temperature: '0.4',
    },
  },
];

const USER_MODEL_PRESET_MAP = Object.fromEntries(USER_MODEL_PRESETS.map((preset) => [preset.id, preset]));

function createUserModelForm(presetId = 'qwen') {
  const preset = USER_MODEL_PRESET_MAP[presetId] || USER_MODEL_PRESET_MAP.qwen;
  return {
    provider: 'openai-compatible',
    api_key: '',
    enabled: true,
    is_default: true,
    preset_id: preset.id,
    ...preset.values,
  };
}

function userModelEditForm(config) {
  return {
    display_name: config.display_name || '',
    provider: config.provider || 'openai-compatible',
    base_url: config.base_url || '',
    chat_model: config.chat_model || '',
    supports_document: config.supports_document !== false,
    supports_reasoning: Boolean(config.supports_reasoning),
    reasoning_type: config.reasoning_type || (config.supports_reasoning ? 'prompt' : 'none'),
    reasoning_label: config.reasoning_label || reasoningLabel(config.reasoning_type || 'none'),
    max_context: String(config.max_context || 131072),
    default_temperature: String(config.default_temperature ?? 0.4),
    enabled: Boolean(config.enabled),
    is_default: Boolean(config.is_default),
  };
}

function UserModelsPanel({
  createUserModelConfig,
  deleteUserModelConfig,
  requestDeleteConfirm,
  setProfileError,
  testUserModelDraft,
  testUserModelConfig,
  updateUserModelConfig,
  userModels,
}) {
  const [form, setForm] = useState(createUserModelForm);
  const [formOpen, setFormOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');
  const [draftTesting, setDraftTesting] = useState(false);
  const [draftTestResult, setDraftTestResult] = useState(null);
  const [testingId, setTestingId] = useState(null);
  const [testResults, setTestResults] = useState({});
  const [keyDialogConfig, setKeyDialogConfig] = useState(null);
  const [editConfig, setEditConfig] = useState(null);
  const [editForm, setEditForm] = useState(null);
  const activePreset = USER_MODEL_PRESET_MAP[form.preset_id] || USER_MODEL_PRESET_MAP.custom;
  const formReady = Boolean(form.display_name.trim() && form.base_url.trim() && form.chat_model.trim() && form.api_key.trim());
  const canSaveForm = formReady && draftTestResult?.ok;
  const imageProbeStatus = imageCapabilityFromTest(form, draftTestResult);

  function updateForm(patch) {
    setDraftTestResult(null);
    setForm((current) => ({ ...current, ...patch }));
  }

  function applyPreset(presetId) {
    const preset = USER_MODEL_PRESET_MAP[presetId] || USER_MODEL_PRESET_MAP.custom;
    setDraftTestResult(null);
    setForm((current) => ({
      ...current,
      ...preset.values,
      provider: 'openai-compatible',
      preset_id: preset.id,
      api_key: current.api_key || '',
    }));
  }

  function openCreateForm() {
    setForm(createUserModelForm());
    setDraftTestResult(null);
    setNotice('');
    setProfileError('');
    setFormOpen(true);
  }

  function closeCreateForm() {
    if (saving || draftTesting) return;
    setFormOpen(false);
    setDraftTestResult(null);
  }

  function openEditForm(config) {
    setEditConfig(config);
    setEditForm(userModelEditForm(config));
    setNotice('');
    setProfileError('');
  }

  function closeEditForm() {
    if (saving) return;
    setEditConfig(null);
    setEditForm(null);
  }

  function updateEditForm(patch) {
    setEditForm((current) => ({ ...current, ...patch }));
  }

  async function submitUserModel(event) {
    event.preventDefault();
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      const saved = await createUserModelConfig(userModelFormPayload(form, { includeApiKey: true }));
      setForm(createUserModelForm());
      setFormOpen(false);
      setDraftTestResult(null);
      setNotice(saved?.supports_image ? '模型连接已保存，图片探测通过。' : '模型连接已保存；图片探测未通过，图片附件发送会提示切换视觉模型。');
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function testDraftModel() {
    setDraftTesting(true);
    setNotice('');
    setProfileError('');
    setDraftTestResult(null);
    try {
      const result = await testUserModelDraft({ ...userModelFormPayload(form, { includeApiKey: true }), detect_image: true });
      setDraftTestResult(result);
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setDraftTesting(false);
    }
  }

  async function patchUserModel(config, patch) {
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      const updated = await updateUserModelConfig(config.id, patch);
      if (updated?.image_detection?.tested) {
        setNotice(updated.supports_image ? '图片探测通过。' : '图片探测未通过；图片附件发送会提示切换视觉模型。');
      }
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function submitEditUserModel(event) {
    event.preventDefault();
    if (!editConfig || !editForm) return;
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      const updated = await updateUserModelConfig(editConfig.id, userModelEditPayload(editForm));
      setEditConfig(null);
      setEditForm(null);
      setNotice(updated?.supports_image ? '模型已保存，图片探测通过。' : '模型已保存；图片探测未通过，图片附件发送会提示切换视觉模型。');
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function replaceKey(config, apiKey) {
    if (!String(apiKey || '').trim()) {
      setProfileError('API key cannot be empty');
      return;
    }
    await patchUserModel(config, { api_key: String(apiKey).trim() });
    setKeyDialogConfig(null);
    setNotice('API Key 已替换，页面不会显示已保存的密钥。');
  }

  async function deleteUserModel(config) {
    const confirmed = await requestDeleteConfirm({
      title: '删除私有模型',
      message: `删除「${config.display_name || config.chat_model}」？`,
      detail: '如果已有智能体使用它，后端会保留该配置。',
      confirmLabel: '删除模型',
    });
    if (!confirmed) return;
    setSaving(true);
    setNotice('');
    setProfileError('');
    try {
      await deleteUserModelConfig(config.id);
    } catch (err) {
      const message = errorMessage(err);
      if (message.toLowerCase().includes('model config is in use')) {
        setNotice('该模型配置正在被智能体使用，已保留。请先切换相关智能体模型，或改为停用。');
      } else {
        setProfileError(message);
      }
    } finally {
      setSaving(false);
    }
  }

  async function testUserModel(config) {
    setTestingId(config.id);
    setNotice('');
    setProfileError('');
    try {
      const result = await testUserModelConfig(config.id);
      if (result?.detected_capabilities) {
        const patch = {};
        const detectedReasoningSupport = result.detected_capabilities.supports_reasoning;
        if (typeof detectedReasoningSupport === 'boolean' && detectedReasoningSupport !== Boolean(config.supports_reasoning)) {
          patch.supports_reasoning = detectedReasoningSupport;
          patch.reasoning_type = detectedReasoningSupport ? (result.detected_capabilities.reasoning_type || 'prompt') : 'none';
          patch.reasoning_label = reasoningLabel(patch.reasoning_type);
        }
        if (Object.keys(patch).length) {
          await updateUserModelConfig(config.id, patch);
        }
      }
      setTestResults((items) => ({ ...items, [config.id]: result }));
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setTestingId(null);
    }
  }

  return (
    <section className="plain-panel user-models-panel">
      <div className="panel-title-row">
        <div>
          <h3>我的模型</h3>
          <p>选择常见兼容网关预设，填入 API Key 和模型名，保存后即可在智能体配置里使用。</p>
        </div>
        <div className="panel-actions">
          <span className="soft-pill">{userModels.length} 个私有模型</span>
          <button className="primary-model-action" type="button" onClick={openCreateForm}><Plus size={15} />新增模型</button>
        </div>
      </div>
      {formOpen && (
        <div className="profile-dialog-backdrop">
          <section className="user-model-dialog" role="dialog" aria-modal="true" aria-label="新增模型" onClick={(event) => event.stopPropagation()}>
            <button className="profile-dialog-close" type="button" title="关闭" aria-label="关闭新增模型" onClick={closeCreateForm} disabled={saving || draftTesting}>
              <X size={16} />
            </button>
            <header className="model-dialog-heading">
              <h3>新增模型</h3>
              <p>选择厂商预设，填入 API Key 和模型名；图片测试只作为诊断信息。</p>
            </header>
            <form className="user-model-form" onSubmit={submitUserModel}>
              <div className="provider-preset-panel">
                <div className="provider-preset-header">
                  <ServerCog size={16} />
                  <span>
                    <strong>模型厂商预设</strong>
                    <small>预设填地址、常用模型名和显式能力；图片测试只作为诊断结果。</small>
                  </span>
                </div>
                <div className="provider-preset-grid">
                  {USER_MODEL_PRESETS.map((preset) => (
                    <button
                      type="button"
                      key={preset.id}
                      className={form.preset_id === preset.id ? 'active' : ''}
                      disabled={saving}
                      onClick={() => applyPreset(preset.id)}
                    >
                      <strong>{preset.label}</strong>
                      <small>{preset.modelHint}</small>
                    </button>
                  ))}
                </div>
                <p className="provider-preset-note">{activePreset.description}</p>
              </div>
              <div className="model-channel-grid">
                <div className="model-channel-card">
                  <div className="model-channel-heading">
                    <strong>聊天模型</strong>
                    <small>用于 Agent 对话、工具推理；图片会直接交给所选模型处理。</small>
                  </div>
                  <div className="user-model-grid compact">
                    <label className="field-stack">
                      <span>显示名称</span>
                       <input value={form.display_name} onChange={(event) => updateForm({ display_name: event.target.value, preset_id: form.preset_id || 'custom' })} placeholder="Qwen Plus" />
                    </label>
                    <label className="field-stack">
                      <span>chat_base_url</span>
                      <input value={form.base_url} onChange={(event) => updateForm({ base_url: event.target.value, preset_id: 'custom' })} placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1" />
                    </label>
                    <label className="field-stack">
                      <span>chat_api_key</span>
                      <input type="password" value={form.api_key} onChange={(event) => updateForm({ api_key: event.target.value })} placeholder="只提交一次，不会回显" autoComplete="off" />
                    </label>
                    <label className="field-stack">
                      <span>chat_model</span>
                      <input value={form.chat_model} onChange={(event) => updateForm({ chat_model: event.target.value, preset_id: form.preset_id || 'custom' })} placeholder="qwen-plus" />
                    </label>
                    <label className="field-stack">
                      <span>默认温度</span>
                      <input type="number" min="0" max="2" step="0.1" value={form.default_temperature} onChange={(event) => updateForm({ default_temperature: event.target.value, preset_id: form.preset_id || 'custom' })} />
                    </label>
                    <label className="field-stack">
                      <span>最大上下文</span>
                      <input type="number" min="1" value={form.max_context} onChange={(event) => updateForm({ max_context: event.target.value, preset_id: form.preset_id || 'custom' })} />
                    </label>
                    <label className="field-stack">
                      <span>深度思考能力</span>
                      <select
                        value={form.reasoning_type || 'none'}
                        onChange={(event) => updateForm({
                          reasoning_type: event.target.value,
                          supports_reasoning: event.target.value !== 'none',
                          reasoning_label: reasoningLabel(event.target.value),
                          preset_id: form.preset_id || 'custom',
                        })}
                      >
                        <option value="none">不支持</option>
                        <option value="prompt">提示词增强</option>
                        <option value="native">原生推理</option>
                      </select>
                    </label>
                  </div>
                </div>
              </div>
              <label className="model-enabled-check inline">
                <input type="checkbox" checked disabled />
                <span>
                  <strong>图片测试仅用于诊断</strong>
                  <small>保存和测试会向模型发送最小图片请求；图片探测未通过时，聊天会阻止图片附件发送。</small>
                </span>
              </label>
              <div className="model-capability-summary">
                <span className="enabled">文本</span>
                <span className={imageProbeStatus.className}>{imageProbeStatus.label}</span>
                <span className="enabled">文档附件后端解析</span>
                <span className={reasoningCapabilityForModel(form).supported ? 'enabled' : ''}>{reasoningCapabilityForModel(form).label}</span>
                <small>图片探测通过后可以发送图片附件；未通过时请切换视觉模型。</small>
              </div>
              <div className="model-checks state-checks">
                <label><input type="checkbox" checked={form.enabled} onChange={(event) => updateForm({ enabled: event.target.checked })} />启用</label>
                <label><input type="checkbox" checked={form.is_default} onChange={(event) => updateForm({ is_default: event.target.checked })} />设为默认</label>
              </div>
              {formReady && !draftTestResult?.ok && <p className="model-row-warning">保存前请先点击"测试当前配置"。图片探测用于判断是否可发送图片附件；文档附件由后端解析成文本，RAG/Embedding 使用后端默认配置。</p>}
              {draftTestResult && <UserModelTestResult result={draftTestResult} />}
              <div className="model-form-actions">
                <button className="preset-action" type="button" disabled={draftTesting || saving || !formReady} onClick={testDraftModel}>{draftTesting ? '检测中' : '测试当前配置'}</button>
                <button className="primary-model-action" type="submit" disabled={saving || !canSaveForm}><Plus size={15} />保存私有模型</button>
              </div>
            </form>
          </section>
        </div>
      )}
      {notice && <p className="model-row-warning">{notice}</p>}
      <div className="user-model-list">
        {userModels.map((config) => (
            <div className="user-model-row" key={config.id}>
              <div className="model-admin-main">
                <strong>{config.display_name || config.chat_model}</strong>
                <small>{config.chat_model} · {config.base_url}</small>
                <div className="model-row-tags">
                  <span className={config.enabled ? 'enabled' : ''}>{config.enabled ? '启用' : '停用'}</span>
                  <span className={config.is_default ? 'enabled' : ''}>{config.is_default ? '默认' : '非默认'}</span>
                  <span>{config.has_api_key ? 'Key 已保存' : '缺少 Key'}</span>
                  {modelCapabilityChips(config).map((label) => (
                    <span className="enabled" key={label}>{label}</span>
                  ))}
                </div>
              </div>
              <div className="user-model-actions">
                <button type="button" disabled={saving} onClick={() => patchUserModel(config, { enabled: !config.enabled })}>{config.enabled ? '停用' : '启用'}</button>
                <button type="button" disabled={saving || config.is_default} onClick={() => patchUserModel(config, { is_default: true })}>设默认</button>
                <button type="button" disabled={saving} onClick={() => openEditForm(config)}>编辑</button>
                <button type="button" disabled={saving} onClick={() => setKeyDialogConfig(config)}>替换 Key</button>
                <button className="model-delete-button" type="button" disabled={saving} onClick={() => deleteUserModel(config)}><Trash2 size={14} />删除</button>
              </div>
            </div>
        ))}
        {userModels.length === 0 && <p className="muted">还没有私有模型配置。选择一个厂商预设，填入 API Key，保存后就可以在智能体配置里选择它。</p>}
      </div>
      {keyDialogConfig && (
        <SecretInputDialog
          label="API Key"
          message={`替换「${keyDialogConfig.display_name || keyDialogConfig.chat_model}」的 API Key。新 key 只提交一次，保存后不回显。`}
          onCancel={() => !saving && setKeyDialogConfig(null)}
          onSubmit={(value) => replaceKey(keyDialogConfig, value).catch((err) => setProfileError(errorMessage(err)))}
          saving={saving}
          submitLabel="替换 Key"
          title="替换模型 API Key"
        />
      )}
      {editConfig && editForm && (
        <div className="profile-dialog-backdrop">
          <section className="user-model-dialog" role="dialog" aria-modal="true" aria-label="编辑模型" onClick={(event) => event.stopPropagation()}>
            <button className="profile-dialog-close" type="button" title="关闭" aria-label="关闭编辑模型" onClick={closeEditForm} disabled={saving}>
              <X size={16} />
            </button>
            <header className="model-dialog-heading">
              <h3>编辑模型</h3>
              <p>修改模型地址、名称和运行参数。API Key 不回显，需要用"替换 Key"单独更新。</p>
            </header>
            <form className="user-model-form" onSubmit={submitEditUserModel}>
              <div className="model-channel-card">
                <div className="model-channel-heading">
                  <strong>{editConfig.display_name || editConfig.chat_model}</strong>
                  <small>测试连接会检查 chat 和图片请求；图片附件会按探测结果校验。</small>
                </div>
                <div className="user-model-grid compact">
                  <label className="field-stack">
                    <span>显示名称</span>
                    <input value={editForm.display_name} onChange={(event) => updateEditForm({ display_name: event.target.value })} />
                  </label>
                  <label className="field-stack">
                    <span>chat_base_url</span>
                    <input value={editForm.base_url} onChange={(event) => updateEditForm({ base_url: event.target.value })} />
                  </label>
                  <label className="field-stack">
                    <span>chat_model</span>
                    <input value={editForm.chat_model} onChange={(event) => updateEditForm({ chat_model: event.target.value })} />
                  </label>
                  <label className="field-stack">
                    <span>默认温度</span>
                    <input type="number" min="0" max="2" step="0.1" value={editForm.default_temperature} onChange={(event) => updateEditForm({ default_temperature: event.target.value })} />
                  </label>
                  <label className="field-stack">
                    <span>最大上下文</span>
                    <input type="number" min="1" value={editForm.max_context} onChange={(event) => updateEditForm({ max_context: event.target.value })} />
                  </label>
                  <label className="field-stack">
                    <span>深度思考能力</span>
                    <select
                      value={editForm.reasoning_type || 'none'}
                      onChange={(event) => updateEditForm({
                        reasoning_type: event.target.value,
                        supports_reasoning: event.target.value !== 'none',
                        reasoning_label: reasoningLabel(event.target.value),
                      })}
                    >
                      <option value="none">不支持</option>
                      <option value="prompt">提示词增强</option>
                      <option value="native">原生推理</option>
                    </select>
                  </label>
                </div>
                <div className="model-checks state-checks">
                  <label><input type="checkbox" checked={editForm.supports_document} onChange={(event) => updateEditForm({ supports_document: event.target.checked })} />文档附件后端解析</label>
                  <label><input type="checkbox" checked={editForm.enabled} onChange={(event) => updateEditForm({ enabled: event.target.checked })} />启用</label>
                  <label><input type="checkbox" checked={editForm.is_default} onChange={(event) => updateEditForm({ is_default: event.target.checked })} />设为默认</label>
                </div>
                <div className="model-capability-summary">
                  <span className="enabled">文本</span>
                  <span className={editForm.supports_image ? 'enabled' : ''}>{editForm.supports_image ? '图片可发送' : '图片未通过'}</span>
                  <span className={editForm.supports_document ? 'enabled' : ''}>文档附件后端解析</span>
                  <span className={reasoningCapabilityForModel(editForm).supported ? 'enabled' : ''}>{reasoningCapabilityForModel(editForm).label}</span>
                </div>
              </div>
              <div className="model-form-actions">
                <button type="button" disabled={saving} onClick={closeEditForm}>取消</button>
                <button className="preset-action" type="button" disabled={saving || testingId === editConfig.id} onClick={() => testUserModel(editConfig)}>
                  {testingId === editConfig.id ? '测试中...' : '测试连接'}
                </button>
                <button className="primary-model-action" type="submit" disabled={saving || !editForm.display_name.trim() || !editForm.base_url.trim() || !editForm.chat_model.trim()}>
                  <Check size={15} />保存修改
                </button>
              </div>
              {testResults[editConfig.id] && <UserModelTestResult result={testResults[editConfig.id]} />}
            </form>
          </section>
        </div>
      )}
    </section>
  );
}

function UserModelTestResult({ result }) {
  const checks = Object.entries(result.checks || {}).filter(([, check]) => check.required);
  const imageCapability = imageCapabilityFromTest(null, result);
  const chatError = result.detected_capabilities?.chat_error || result.checks?.chat?.message || '';
  const chatErrorCode = result.detected_capabilities?.chat_error_code || result.checks?.chat?.error_code || '';
  const imageError = result.detected_capabilities?.image_error || result.checks?.image?.message || '';
  const imageErrorCode = result.detected_capabilities?.image_error_code || result.checks?.image?.error_code || '';
  return (
    <div className={result.ok ? 'user-model-test ok' : 'user-model-test'}>
      <strong>{result.ok ? '能力检查通过' : '能力检查失败'} · {result.model} · {result.latency_ms}ms</strong>
      <div className="capability-checks">
        {result.detected_capabilities && (
          <span className={imageCapability.className}>
            图片: {imageCapability.label}
          </span>
        )}
        {result.detected_capabilities && (
          <span className={result.detected_capabilities.supports_reasoning ? 'ok' : ''}>
            深度思考: {result.detected_capabilities.supports_reasoning ? reasoningLabel(result.detected_capabilities.reasoning_type || 'prompt') : '不支持'}
          </span>
        )}
        {checks.map(([name, check]) => (
          <span key={name} className={check.ok ? 'ok' : 'fail'}>
            {capabilityCheckLabel(name)}: {check.ok ? '通过' : '失败'}
          </span>
        ))}
      </div>
      <small>{result.message}</small>
      {chatError && <small>chat 测试失败：{chatErrorCode ? `${chatErrorCode} · ` : ''}{chatError}</small>}
      {imageError && <small>图片探测失败：{imageErrorCode ? `${imageErrorCode} · ` : ''}{imageError}</small>}
    </div>
  );
}

const QWEN_MODEL_PRESET = {
  display_name: 'Qwen Plus',
  model_name: 'qwen-plus',
  provider: 'openai-compatible',
  max_context: '131072',
  default_temperature: 0.4,
  supports_text: true,
  supports_image: false,
  supports_document: true,
  supports_reasoning: true,
  reasoning_type: 'prompt',
  reasoning_label: '提示词增强',
  enabled: true,
};

const MODEL_CAPABILITY_PRESETS = {
  qwen_plus: {
    label: 'Qwen 文本',
    values: QWEN_MODEL_PRESET,
  },
  text_document: {
    label: '文本模型',
    values: { supports_text: true, supports_image: false, supports_document: true, supports_reasoning: true, reasoning_type: 'prompt', reasoning_label: 'prompt' },
  },
  vision_document: {
    label: '视觉模型',
    values: { supports_text: true, supports_image: true, supports_document: true, supports_reasoning: true, reasoning_type: 'prompt', reasoning_label: 'prompt' },
  },
  text_only: {
    label: '\u7eaf\u6587\u672c',
    values: { supports_text: true, supports_image: false, supports_document: true, supports_reasoning: false, reasoning_type: 'none', reasoning_label: 'none' },
  },
  custom: {
    label: '\u81ea\u5b9a\u4e49',
    values: {},
  },
};

function createModelForm() {
  return { ...QWEN_MODEL_PRESET, preset: 'qwen_plus' };
}

function ModelAdminPanel({ createModelConfig, deleteModelConfig, models, requestDeleteConfirm, setProfileError, updateModelConfig }) {
  const [form, setForm] = useState(createModelForm);
  const [formOpen, setFormOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [rowNotice, setRowNotice] = useState({ modelId: null, message: '' });

  function applyPreset(preset) {
    const values = MODEL_CAPABILITY_PRESETS[preset]?.values || {};
    setForm((current) => ({
      ...current,
      ...values,
      preset,
    }));
  }

  async function createModel(event) {
    event.preventDefault();
    setSaving(true);
    setProfileError('');
    setRowNotice({ modelId: null, message: '' });
    try {
      await createModelConfig(modelFormPayload(form));
      setForm(createModelForm());
      setFormOpen(false);
      setAdvancedOpen(false);
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  function openCreateModel() {
    setForm(createModelForm());
    setAdvancedOpen(false);
    setRowNotice({ modelId: null, message: '' });
    setProfileError('');
    setFormOpen(true);
  }

  function closeCreateModel() {
    if (saving) return;
    setFormOpen(false);
    setAdvancedOpen(false);
  }

  async function patchModel(model, patch) {
    setSaving(true);
    setProfileError('');
    setRowNotice({ modelId: null, message: '' });
    try {
      await updateModelConfig(model.id, patch);
    } catch (err) {
      setProfileError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function deleteModel(model) {
    const label = model.display_name || model.model_name;
    const confirmed = await requestDeleteConfirm({
      title: '删除系统模型',
      message: `确定删除「${label}」？`,
      detail: '删除只适用于未被引用的自定义模型；默认模型或已被智能体使用的模型会被后端保留，请改用停用。',
      confirmLabel: '删除模型',
    });
    if (!confirmed) return;
    setSaving(true);
    setProfileError('');
    setRowNotice({ modelId: null, message: '' });
    try {
      await deleteModelConfig(model.id);
    } catch (err) {
      const message = errorMessage(err);
      const normalizedMessage = message.toLowerCase();
      setRowNotice({ modelId: model.id, message: modelDeleteGuidance(message) });
      if (!normalizedMessage.includes('model is protected') && !normalizedMessage.includes('model is in use')) {
        setProfileError(message);
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="plain-panel model-admin-panel">
      <div className="panel-title-row">
        <div>
          <h3>模型管理</h3>
          <p>管理员维护可被智能体选择的模型，以及文本、图片等对话输入能力。文档附件由后端解析成文本。</p>
        </div>
        <div className="panel-actions">
          <span className="soft-pill">{models.length} 个模型</span>
          <button className="primary-model-action" type="button" onClick={openCreateModel}><Plus size={15} />新增系统模型</button>
        </div>
      </div>
      {formOpen && (
        <div className="profile-dialog-backdrop">
          <section className="user-model-dialog" role="dialog" aria-modal="true" aria-label="新增系统模型" onClick={(event) => event.stopPropagation()}>
            <button className="profile-dialog-close" type="button" title="关闭" aria-label="关闭新增系统模型" onClick={closeCreateModel} disabled={saving}>
              <X size={16} />
            </button>
            <header className="model-dialog-heading">
              <h3>新增系统模型</h3>
              <p>管理员维护平台内置模型能力，普通用户仍优先使用自己的模型配置。</p>
            </header>
            <form className="model-admin-form" onSubmit={createModel}>
              <div className="model-form-section">
                <div className="model-section-title">
                  <strong>基础字段</strong>
                  <span>普通管理员只需要确认名称、能力预设和启用状态。</span>
                </div>
                <div className="model-basic-grid">
                  <label className="field-stack">
                    <span>显示名称</span>
                    <input value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} placeholder="Qwen Plus" />
                  </label>
                  <label className="field-stack">
                    <span>模型名</span>
                    <input value={form.model_name} onChange={(event) => setForm({ ...form, model_name: event.target.value })} placeholder="qwen-plus" />
                  </label>
                  <label className="field-stack">
                    <span>能力预设</span>
                    <select value={form.preset} onChange={(event) => applyPreset(event.target.value)}>
                      {Object.entries(MODEL_CAPABILITY_PRESETS).map(([value, preset]) => (
                        <option key={value} value={value}>{preset.label}</option>
                      ))}
                    </select>
                  </label>
                  <div className="model-capability-summary compact">
                    <span className={form.supports_text ? 'enabled' : ''}>文本</span>
                    <span className={form.supports_image ? 'enabled' : ''}>{form.supports_image ? '图片声明' : '图片未声明'}</span>
                    <span className="enabled">文档附件后端解析</span>
                    <span className={reasoningCapabilityForModel(form).supported ? 'enabled' : ''}>{reasoningCapabilityForModel(form).label}</span>
                  </div>
                  <label className="model-enabled-check">
                    <input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} />
                    <span>
                      <strong>启用</strong>
                      <small>出现在智能体模型选择中。</small>
                    </span>
                  </label>
                </div>
                <div className="model-preset-banner">
                  <Wand2 size={16} />
                  <span>
                    <strong>Qwen 快捷预设</strong>
                    <small>填入 qwen-plus、OpenAI-compatible、文本能力、131072 上下文和 0.4 温度。</small>
                  </span>
                  <button type="button" className="preset-action" disabled={saving} onClick={() => applyPreset('qwen_plus')}>填入 qwen-plus</button>
                </div>
              </div>
              <div className="model-form-section">
                <button className="model-advanced-toggle" type="button" onClick={() => setAdvancedOpen(!advancedOpen)}>
                  <Settings2 size={15} />
                  高级字段
                  <span>{advancedOpen ? '收起' : '展开'}</span>
                </button>
                {advancedOpen && (
                  <div className="model-advanced-grid">
                    <label className="field-stack">
                      <span>provider</span>
                      <input value={form.provider} onChange={(event) => setForm({ ...form, provider: event.target.value, preset: 'custom' })} placeholder="openai-compatible" />
                    </label>
                    <label className="field-stack">
                      <span>最大上下文</span>
                      <input type="number" min="1" value={form.max_context} onChange={(event) => setForm({ ...form, max_context: event.target.value, preset: 'custom' })} placeholder="131072" />
                    </label>
                    <label className="field-stack">
                      <span>默认温度</span>
                      <input type="number" min="0" max="2" step="0.1" value={form.default_temperature} onChange={(event) => setForm({ ...form, default_temperature: event.target.value, preset: 'custom' })} placeholder="0.4" />
                    </label>
                    <label className="field-stack">
                      <span>深度思考能力</span>
                      <select
                        value={form.reasoning_type || 'none'}
                        onChange={(event) => setForm({
                          ...form,
                          reasoning_type: event.target.value,
                          supports_reasoning: event.target.value !== 'none',
                          reasoning_label: reasoningLabel(event.target.value),
                          preset: 'custom',
                        })}
                      >
                        <option value="none">不支持</option>
                        <option value="prompt">提示词增强</option>
                        <option value="native">原生推理</option>
                      </select>
                    </label>
                    <div className="model-capability-summary">
                      <span className={form.supports_text ? 'enabled' : ''}>{form.supports_text ? '文本' : '文本未声明'}</span>
                      <span className={form.supports_image ? 'enabled' : ''}>{form.supports_image ? '图片声明' : '图片未声明'}</span>
                      <span className="enabled">文档附件后端解析</span>
                      <span className={reasoningCapabilityForModel(form).supported ? 'enabled' : ''}>{reasoningCapabilityForModel(form).label}</span>
                          </div>
                  </div>
                )}
              </div>
              <div className="model-form-actions">
                <button className="primary-model-action" type="submit" disabled={saving || !form.display_name.trim() || !form.model_name.trim()}><Plus size={15} />保存系统模型</button>
              </div>
            </form>
          </section>
        </div>
      )}
      <div className="model-admin-list">
        {models.map((model) => (
          <div className="model-admin-row" key={model.id}>
            <div className="model-admin-main">
              <strong>{model.display_name || model.model_name}</strong>
              <small>{model.model_name} · {model.provider} · 上下文 {model.max_context}</small>
              <div className="model-row-tags">
                <span className={model.enabled ? 'enabled' : ''}>{model.enabled ? '启用' : '已停用'}</span>
                {modelCapabilityChips(model).map((label) => (
                  <span className="enabled" key={label}>{label}</span>
                ))}
              </div>
              {rowNotice.modelId === model.id && <p className="model-row-warning">{rowNotice.message}</p>}
            </div>
            <div className="model-admin-actions">
              <button type="button" className={model.enabled ? 'model-state-toggle enabled' : 'model-state-toggle'} disabled={saving} onClick={() => patchModel(model, { enabled: !model.enabled })}>{model.enabled ? '停用' : '启用'}</button>
              <button className="model-delete-button" type="button" title="删除模型配置" disabled={saving} onClick={() => deleteModel(model)}><Trash2 size={14} />删除</button>
            </div>
          </div>
        ))}
        {models.length === 0 && <p className="muted">还没有模型配置。</p>}
      </div>
    </section>
  );
}

function modelFormPayload(form) {
  const maxContext = Number(form.max_context);
  const temperature = Number(form.default_temperature);
  return {
    display_name: form.display_name.trim(),
    model_name: form.model_name.trim(),
    provider: form.provider.trim() || 'openai-compatible',
    supports_text: Boolean(form.supports_text),
    supports_image: Boolean(form.supports_image),
    supports_document: Boolean(form.supports_document),
    supports_reasoning: Boolean(form.supports_reasoning) && (form.reasoning_type || 'none') !== 'none',
    reasoning_type: form.reasoning_type || 'none',
    reasoning_label: reasoningLabel(form.reasoning_type || 'none'),
    max_context: Number.isFinite(maxContext) && maxContext > 0 ? maxContext : 8192,
    default_temperature: Number.isFinite(temperature) ? temperature : 0.4,
    enabled: Boolean(form.enabled),
  };
}

function toolFormPayload(form, { includeSecret = false } = {}) {
  const timeout = Number(form.timeout_seconds);
  const isHttp = form.type === 'http';
  const isMcp = form.type === 'mcp';
  const transport = isMcp ? mcpTransportValue(form.mcp_transport, form.url) : 'streamable_http';
  const isMcpStdio = isMcp && transport === 'stdio';
  const authType = (isHttp || (isMcp && !isMcpStdio)) ? form.auth_type || 'none' : 'none';
  const auth = {
    type: authType,
    header_name: ['bearer', 'header'].includes(authType) ? form.auth_header_name || 'Authorization' : null,
    query_name: authType === 'query' ? form.auth_query_name || null : null,
  };
  if (includeSecret && authType !== 'none' && String(form.auth_secret || '').trim()) {
    auth.secret = String(form.auth_secret).trim();
  }
  const method = String(form.method || 'GET').toUpperCase();
  const hasBodySchema = isHttp && !['GET', 'DELETE'].includes(method);
  const mcpConfig = isMcp ? {
    transport,
    tool_name: String(form.mcp_tool_name || '').trim(),
    input_schema: parseJsonField(form.mcp_input_schema || '{}', 'mcp.input_schema'),
  } : {};
  if (isMcpStdio) {
    const argsStr = String(form.mcp_args || '').trim();
    mcpConfig.command = String(form.mcp_command || '').trim();
    mcpConfig.args = argsStr ? argsStr.split(/\s+/) : [];
    if (String(form.mcp_cwd || '').trim()) {
      mcpConfig.cwd = String(form.mcp_cwd).trim();
    }
    if (String(form.mcp_env || '').trim()) {
      mcpConfig.env = parseJsonField(form.mcp_env, 'mcp.env');
    }
  }
  return {
    type: isHttp ? 'http' : isMcp ? 'mcp' : form.type,
    name: String(form.name || '').trim(),
    label: String(form.label || '').trim(),
    description: String(form.description || '').trim(),
    server_label: String(form.server_label || '').trim(),
    enabled: Boolean(form.enabled),
    method: isHttp ? method : isMcp ? 'POST' : method,
    url: isMcpStdio ? '' : (isHttp || isMcp) ? String(form.url || '').trim() : '',
    headers_schema: isHttp ? parseJsonField(form.headers_schema, 'headers_schema') : {},
    query_schema: isHttp || form.type === 'builtin_search' ? parseJsonField(form.query_schema, 'query_schema') : {},
    body_schema: hasBodySchema ? parseJsonField(form.body_schema, 'body_schema') : {},
    auth,
    response_path: isHttp ? String(form.response_path || '$').trim() || '$' : '$',
    timeout_seconds: (isHttp || isMcp) ? clampToolTimeout(isMcp ? 'mcp' : 'http', timeout) : 10,
    mcp: mcpConfig,
  };
}

function toolTimeoutMax(type) {
  return type === 'mcp' ? 120 : 30;
}

function toolTimeoutDefault(type) {
  return type === 'mcp' ? 30 : 10;
}

function clampToolTimeout(type, value) {
  const numeric = Number(value);
  const fallback = toolTimeoutDefault(type);
  const max = toolTimeoutMax(type);
  return Number.isFinite(numeric) ? Math.min(max, Math.max(1, numeric)) : fallback;
}

function mcpTransportValue(value, url = '') {
  const raw = String(value || '').trim().toLowerCase();
  if (raw === 'sse' || raw === 'streamable_http' || raw === 'stdio') return raw;
  if (raw === 'streamable-http' || raw === 'streamablehttp') return 'streamable_http';
  const path = (() => {
    try {
      return new URL(String(url || ''), window.location.origin).pathname.toLowerCase().replace(/\/+$/, '');
    } catch {
      return String(url || '').split('?')[0].split('#')[0].toLowerCase().replace(/\/+$/, '');
    }
  })();
  return path.endsWith('/sse') || path === 'sse' ? 'sse' : 'streamable_http';
}

function uniqueToolName(name, usedNames) {
  const raw = String(name || 'mcp_tool').trim().replace(/[^0-9A-Za-z_]+/g, '_').replace(/^_+|_+$/g, '') || 'mcp_tool';
  const base = raw.slice(0, 120) || 'mcp_tool';
  if (!usedNames.has(base)) {
    usedNames.add(base);
    return base;
  }
  for (let index = 2; index < 10000; index += 1) {
    const suffix = `_${index}`;
    const candidate = `${base.slice(0, Math.max(1, 120 - suffix.length))}${suffix}`;
    if (!usedNames.has(candidate)) {
      usedNames.add(candidate);
      return candidate;
    }
  }
  throw new Error('无法生成唯一的 MCP 工具名称');
}

function formWithMcpDiscovery(form, item, name) {
  const nextMcp = item?.mcp || {};
  const result = {
    ...form,
    type: 'mcp',
    name: name || item?.name || form.name,
    label: item?.label || item?.name || form.label,
    description: item?.description || form.description,
    mcp_transport: nextMcp.transport || form.mcp_transport,
    mcp_tool_name: nextMcp.tool_name || item?.label || item?.name || form.mcp_tool_name,
    mcp_input_schema: JSON.stringify(nextMcp.input_schema || { type: 'object', properties: {} }, null, 2),
    server_label: form.server_label || item?.server_label,
  };
  if (nextMcp.transport === 'stdio') {
    result.mcp_command = nextMcp.command || form.mcp_command || '';
    result.mcp_args = Array.isArray(nextMcp.args) ? nextMcp.args.join(' ') : (nextMcp.args || form.mcp_args || '');
    result.mcp_cwd = nextMcp.cwd || form.mcp_cwd || '';
    result.mcp_env = nextMcp.env && typeof nextMcp.env === 'object' ? JSON.stringify(nextMcp.env, null, 2) : (form.mcp_env || '');
  }
  return result;
}

function buildMcpDiscoveryPayloads(form, discoveries, existingTools = []) {
  const usedNames = new Set(existingTools.map((tool) => tool?.name).filter(Boolean));
  const isStdio = mcpTransportValue(form.mcp_transport, form.url) === 'stdio';
  const existingRemoteKeys = new Set(
    existingTools
      .filter((tool) => toolType(tool) === 'mcp')
      .map((tool) => {
        const keyUrl = isStdio
          ? `${String(tool?.mcp?.command || '').trim()}\n${(tool?.mcp?.args || []).join(' ')}`
          : String(tool?.url || '').trim();
        return `${keyUrl}\n${String(tool?.mcp?.tool_name || '').trim()}`;
      })
  );
  const identifier = isStdio
    ? `${String(form.mcp_command || '').trim()}\n${String(form.mcp_args || '').trim()}`
    : String(form.url || '').trim();
  const payloads = [];
  let skippedCount = 0;

  for (const item of discoveries || []) {
    const remoteName = String(item?.mcp?.tool_name || item?.label || item?.name || '').trim();
    if (!remoteName) continue;
    const remoteKey = `${identifier}\n${remoteName}`;
    if (existingRemoteKeys.has(remoteKey)) {
      skippedCount += 1;
      continue;
    }
    const name = uniqueToolName(item?.name || remoteName, usedNames);
    payloads.push(toolFormPayload(formWithMcpDiscovery(form, item, name), { includeSecret: true }));
  }

  return { payloads, skippedCount };
}

function parseJsonField(value, label) {
  const text = String(value ?? '').trim();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`${label} must be valid JSON`);
  }
}

function parseOptionalJsonField(value, label) {
  const text = String(value ?? '').trim();
  return text ? parseJsonField(text, label) : null;
}

function toolType(tool) {
  return tool?.type || (tool?.mcp ? 'mcp' : tool?.name === 'builtin_search' ? 'builtin_search' : 'http');
}

function toolHasSecret(tool) {
  return Boolean(tool?.auth?.has_secret || tool?.has_secret || tool?.auth_has_secret);
}

function defaultToolTestInput(tool) {
  if (toolType(tool) === 'builtin_search') return '{\n  "query": "AgentBase"\n}';
  if (toolType(tool) === 'mcp') {
    return JSON.stringify(sampleInputFromSchema(tool?.mcp?.input_schema), null, 2);
  }
  return '{\n  "city": "Hangzhou"\n}';
}

function sampleInputFromSchema(schema) {
  const normalized = schema && typeof schema === 'object' ? schema : {};
  const properties = normalized.properties && typeof normalized.properties === 'object' ? normalized.properties : {};
  const sample = {};
  Object.entries(properties).forEach(([key, spec]) => {
    sample[key] = sampleValueForSchema(spec, key);
  });
  return sample;
}

function sampleValueForSchema(spec, key = '') {
  if (!spec || typeof spec !== 'object') return '';
  if (Array.isArray(spec.enum) && spec.enum.length > 0) return spec.enum[0];
  const type = Array.isArray(spec.type) ? spec.type.find((item) => item !== 'null') : spec.type;
  const hint = `${key} ${spec.description || ''}`.toLowerCase();
  if (type === 'number') return Number.isFinite(spec.minimum) ? spec.minimum : 1.23;
  if (type === 'integer') return Number.isFinite(spec.minimum) ? spec.minimum : 1;
  if (type === 'boolean') return true;
  if (type === 'array') return [sampleValueForSchema(spec.items, `${key}_item`)];
  if (type === 'object') return sampleInputFromSchema(spec);
  if (/(origin|destination|location|lnglat|lonlat|coordinate|coord|经度|纬度|坐标)/i.test(hint)) return '116.397428,39.90923';
  if (/(city|adcode|citycode|行政区|城市编码)/i.test(hint)) return '310000';
  if (/(keyword|keywords|query|关键字)/i.test(hint)) return '咖啡';
  if (/(address|addr|地址)/i.test(hint)) return '北京市朝阳区阜通东大街6号';
  if (/\bip\b/i.test(hint)) return '8.8.8.8';
  if (/(phone|mobile|tel|电话)/i.test(hint)) return '13800138000';
  if (spec.format === 'uri' || spec.format === 'url') return 'https://example.com';
  return 'example';
}

function removeTextSuffix(value, suffix) {
  const text = String(value || '');
  const tail = String(suffix || '');
  return tail && text.endsWith(tail) ? text.slice(0, -tail.length) : text;
}

function appendReasoningTimelineItem(message, chunk) {
  const content = String(chunk || '');
  if (!content) return message;
  const timeline = Array.isArray(message?.reasoningTimeline) ? [...message.reasoningTimeline] : [];
  const last = timeline[timeline.length - 1];
  if (last?.type === 'reasoning') {
    timeline[timeline.length - 1] = { ...last, content: `${last.content || ''}${content}` };
  } else {
    timeline.push({
      id: `reasoning-${Date.now()}-${timeline.length}`,
      type: 'reasoning',
      content,
    });
  }
  return { ...message, reasoningTimeline: timeline };
}

function appendToolTimelineItem(message, eventData, eventType = 'tool_call') {
  const item = formatToolTimelineLabel(eventData, eventType);
  if (!item) return message;
  const timeline = Array.isArray(message?.reasoningTimeline) ? [...message.reasoningTimeline] : [];
  const existingIndex = findToolTimelineMergeIndex(timeline, item);
  if (existingIndex >= 0) {
    const existing = timeline[existingIndex];
    timeline[existingIndex] = {
      ...existing,
      ...item,
      id: existing.id || item.id,
      inputLabel: item.inputLabel || existing.inputLabel || '参数',
      inputPreview: item.inputPreview || existing.inputPreview || '',
      rawInput: item.rawInput || existing.rawInput || '',
      rawResult: item.rawResult || existing.rawResult || '',
    };
    return { ...message, reasoningTimeline: timeline };
  }
  const uniqueItem = ensureUniqueTimelineItemId(item, timeline);
  timeline.push(uniqueItem);
  return { ...message, reasoningTimeline: timeline };
}

function findToolTimelineMergeIndex(timeline, item) {
  const toolCallId = item.toolCallId || '';
  if (toolCallId) {
    return timeline.findIndex((entry) => entry?.toolCallId === toolCallId);
  }
  const toolRunKey = item.toolRunKey || '';
  if (!toolRunKey) return -1;
  for (let index = timeline.length - 1; index >= 0; index -= 1) {
    const entry = timeline[index];
    if (entry?.toolRunKey === toolRunKey && entry?.status === 'running') {
      return index;
    }
  }
  return -1;
}

function ensureUniqueTimelineItemId(item, timeline) {
  const baseId = item.id || `${item.type || 'tool'}-${Date.now()}`;
  if (!timeline.some((entry) => entry?.id === baseId)) {
    return { ...item, id: baseId };
  }
  let suffix = timeline.length + 1;
  let nextId = `${baseId}-${suffix}`;
  while (timeline.some((entry) => entry?.id === nextId)) {
    suffix += 1;
    nextId = `${baseId}-${suffix}`;
  }
  return { ...item, id: nextId };
}

function formatToolTimelineLabel(eventData = {}, eventType = 'tool_call') {
  if (eventType === 'search_status') {
    // Only show search timeline item if search actually happened — not just
    // the initial "tool_available" status emitted at the start of every run.
    const count = Number(eventData.matched_results || 0);
    const reason = String(eventData.reason || '');
    if (reason === 'tool_available' && count === 0) return null;
    if (eventData.requested === false && !eventData.matched_results) return null;
    const status = reason === 'web_search_unavailable' ? 'error' : count > 0 || reason === 'no_results' ? 'success' : 'running';
    const title = count > 0 ? `搜索到 ${count} 个网页` : status === 'error' ? '搜索工具不可用' : '正在搜索网页';
    const provider = eventData.provider ? String(eventData.provider) : '';
    const query = eventData.query ? `关键词：${String(eventData.query)}` : '';
    const summary = searchStatusSummary(eventData);
    const toolCallId = eventData.tool_call_id || eventData.call_id || '';
    const toolRunKey = timelineToolRunKey({
      toolCallId,
      toolType: 'builtin_search',
      toolName: 'web_search',
      rawInput: eventData.query || '',
    });
    return {
      id: `search-${Date.now()}`,
      type: 'search',
      status,
      toolCallId,
      toolRunKey,
      title,
      meta: [provider, query].filter(Boolean).join(' · '),
      latency: formatTimelineLatency(eventData.latency_ms),
      summary,
    };
  }

  const toolName = eventData.tool_name || eventData.tool || eventData.name || eventData.tool_id || 'tool';
  const toolType = eventData.tool_type || eventData.type || 'tool';
  const isSearchTool = toolType === 'builtin_search' || toolName === 'web_search';
  const isStart = eventType === 'tool_call_start' || eventData.type === 'tool_call_start';
  const status = eventData.status === 'error' || eventData.error_code ? 'error' : isStart || eventData.status === 'running' ? 'running' : 'success';
  const toolCallId = eventData.tool_call_id || eventData.call_id || '';
  const rawInput = eventData.input_preview || '';
  const rawResult = eventData.result_preview || eventData.error || eventData.error_code || '';
  const inputSummary = summarizeToolInput(rawInput);
  const toolRunKey = timelineToolRunKey({ toolCallId, toolType, toolName, rawInput });
  const title = status === 'running'
    ? `正在调用 ${toolName}`
    : isSearchTool
      ? '调用联网搜索'
      : `调用 ${toolName}`;
  return {
    id: `${isSearchTool ? 'search' : 'tool'}-${toolCallId || Date.now()}-${toolName}`,
    type: isSearchTool ? 'search' : 'tool',
    status,
    toolCallId,
    toolRunKey,
    title,
    meta: [toolType, status === 'error' ? '失败' : status === 'running' ? '运行中' : '完成'].filter(Boolean).join(' · '),
    latency: formatTimelineLatency(eventData.latency_ms),
    inputLabel: inputSummary.label,
    inputPreview: inputSummary.text,
    summary: summarizeToolResult(rawResult, { status, toolType, toolName }),
    rawInput,
    rawResult,
  };
}

function timelineToolRunKey({ toolCallId = '', toolType = 'tool', toolName = 'tool', rawInput = '' } = {}) {
  if (toolCallId) return `call:${toolCallId}`;
  return [
    'fallback',
    String(toolType || 'tool'),
    String(toolName || 'tool'),
    normalizeToolRunInput(rawInput),
  ].join(':');
}

function normalizeToolRunInput(value) {
  const data = parseJsonPreview(value);
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const query = findFirstString(data, ['query', 'q', 'keyword', 'keywords', 'search', 'text', 'input']);
    if (query) return compactTimelineText(query, 160);
    return compactTimelineText(JSON.stringify(data), 180);
  }
  return compactTimelineText(value || '', 180);
}

function searchStatusSummary(eventData = {}) {
  const items = Array.isArray(eventData.items) ? eventData.items : [];
  const titles = items
    .slice(0, 3)
    .map((item) => item?.title || item?.url || '')
    .filter(Boolean);
  if (titles.length) return `结果：${titles.join('、')}`;
  if (eventData.reason === 'no_results') return '未返回可用搜索结果。';
  if (eventData.reason === 'web_search_unavailable') return '当前搜索运行时不可用。';
  if (eventData.query) return `正在围绕"${eventData.query}"获取结果。`;
  return '';
}

function formatTimelineLatency(value) {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms <= 0) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s`;
}

function compactTimelineText(value, limit = 180) {
  const text = typeof value === 'string' ? value : JSON.stringify(value || '', null, 0);
  const compact = String(text || '').replace(/\s+/g, ' ').trim();
  return compact.length > limit ? `${compact.slice(0, limit)}...` : compact;
}

function summarizeToolInput(rawInput) {
  const data = parseJsonPreview(rawInput);
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const query = findFirstString(data, ['query', 'q', 'keyword', 'keywords', 'search', 'text', 'input']);
    const count = findFirstScalar(data, ['count', 'limit', 'top_k', 'max_results', 'num_results']);
    if (query) {
      return {
        label: '查询',
        text: compactTimelineText([query, count ? `数量 ${count}` : ''].filter(Boolean).join(' · '), 120),
      };
    }
    const keys = Object.keys(data).filter((key) => data[key] !== undefined && data[key] !== null && data[key] !== '');
    if (keys.length) {
      const preview = keys.length <= 3
        ? keys.map((key) => `${key}: ${compactParamValue(data[key])}`).join(' · ')
        : `已传入 ${keys.length} 个参数：${keys.slice(0, 3).join('、')}`;
      return { label: '参数', text: compactTimelineText(preview, 140) };
    }
  }
  const text = looksLikeJsonText(rawInput) ? '已传入结构化参数。' : compactTimelineText(rawInput || '', 120);
  return { label: '参数', text };
}

function summarizeToolResult(rawResult, context = {}) {
  if (context.status === 'running') return '';
  if (context.status === 'error') {
    return rawResult ? `调用失败：${compactTimelineText(rawResult, 140)}` : '调用失败。';
  }
  const data = parseJsonPreview(rawResult);
  const items = collectResultItems(data);
  if (items.length) return summarizeResultItems(items, rawResult);
  const fallbackSearchCount = countJsonField(rawResult, 'snippet') || countJsonField(rawResult, 'title');
  if (fallbackSearchCount) {
    const dates = extractDateSignals(rawResult);
    return compactTimelineText([
      `搜索到约 ${fallbackSearchCount} 条结果`,
      dates.length ? `结果中出现 ${dates.slice(0, 3).join('、')} 等日期` : '',
    ].filter(Boolean).join('；'), 180);
  }
  if (data && typeof data === 'object') {
    if (!Array.isArray(data)) {
      const error = findFirstString(data, ['error', 'message', 'detail']);
      if (error) return compactTimelineText(error, 160);
      const keys = Object.keys(data).filter(Boolean);
      if (keys.length) return `工具返回 ${keys.length} 个字段：${keys.slice(0, 4).join('、')}`;
    }
    if (Array.isArray(data)) return `工具返回 ${data.length} 条结构化结果。`;
  }
  if (looksLikeJsonText(rawResult)) return '工具返回了结构化结果，原始内容可展开查看。';
  return compactTimelineText(rawResult || '', 160);
}

function summarizeResultItems(items, rawResult) {
  const names = items
    .slice(0, 3)
    .map((item) => item?.title || item?.name || item?.hostname || item?.source || item?.url || '')
    .filter(Boolean);
  const dates = extractDateSignals(rawResult || JSON.stringify(items));
  return compactTimelineText([
    `搜索到 ${items.length} 条结果`,
    names.length ? `包括 ${names.join('、')}` : '',
    dates.length ? `结果中出现 ${dates.slice(0, 3).join('、')} 等日期` : '',
  ].filter(Boolean).join('；'), 180);
}

function parseJsonPreview(value) {
  if (!value || typeof value !== 'string') return value || null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function looksLikeJsonText(value) {
  const text = String(value || '').trim();
  return text.startsWith('{') || text.startsWith('[');
}

function collectResultItems(data) {
  if (!data) return [];
  if (Array.isArray(data)) return data.filter((item) => item && typeof item === 'object');
  if (typeof data !== 'object') return [];
  for (const key of ['pages', 'items', 'results', 'data', 'documents']) {
    const value = data[key];
    if (Array.isArray(value)) return value.filter((item) => item && typeof item === 'object');
  }
  return [];
}

function findFirstString(data, keys) {
  for (const key of keys) {
    const value = data?.[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return '';
}

function findFirstScalar(data, keys) {
  for (const key of keys) {
    const value = data?.[key];
    if (typeof value === 'string' || typeof value === 'number') return value;
  }
  return '';
}

function compactParamValue(value) {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return compactTimelineText(String(value), 50);
  if (Array.isArray(value)) return `${value.length} 项`;
  if (value && typeof value === 'object') return '对象';
  return '';
}

function countJsonField(value, fieldName) {
  if (!value || typeof value !== 'string') return 0;
  const escaped = fieldName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return (value.match(new RegExp(`"${escaped}"\\s*:`, 'g')) || []).length;
}

function extractDateSignals(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value || '');
  const matches = text.match(/20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?|20\d{2}年\d{1,2}月?/g) || [];
  return [...new Set(matches)].slice(0, 5);
}

function debugEventSummary(event) {
  if (['tool_call', 'tool_call_start', 'tool_call_result'].includes(event.event)) {
    const status = event.status || 'unknown';
    const name = event.tool_name || event.tool_id || 'tool';
    return `${name} · ${event.tool_type || 'tool'} · ${status} · ${event.latency_ms ?? '-'}ms`;
  }
  if (event.event === 'rag_status') {
    const enabled = event.enabled === false ? 'disabled' : 'enabled';
    const source = event.effective_source || 'default';
    const dense = event.dense?.matched ?? 0;
    const bm25 = event.bm25?.matched ?? 0;
    const rrf = event.rrf?.matched ?? event.matched_chunks ?? 0;
    const rerank = event.rerank?.applied ? 'rerank on' : event.rerank?.enabled ? 'rerank fallback' : 'rerank off';
    const cache = event.cache?.hit ? 'cache hit' : 'cache miss';
    const evidence = event.no_evidence ? 'no evidence' : 'evidence ok';
    return `${enabled} · ${source} · dense ${dense} · bm25 ${bm25} · rrf ${rrf} · ${rerank} · ${cache} · ${evidence}`;
  }
  if (event.event === 'search_status') {
    if (event.enabled) {
      return `enabled · ${event.provider || 'web'} · results ${event.matched_results ?? 0} · ${event.latency_ms ?? '-'}ms`;
    }
    return event.requested ? `disabled · ${event.reason || 'search unavailable'}` : 'disabled · not requested';
  }
  if (event.event === 'memory_used') {
    const enabled = event.enabled === false ? 'disabled' : 'enabled';
    const profile = event.profile_found ? 'profile found' : 'profile missing';
    const summary = event.summary_used ? 'summary used' : 'summary skipped';
    const session = event.session_summary_used ? 'session summary used' : 'session summary skipped';
    return `${enabled} · ${profile} · ${summary} · facts ${event.facts_count ?? 0} · prefs ${(event.preferences_keys || []).length} · ${session}`;
  }
  if (event.event === 'thinking_status') {
    if (event.enabled) {
      return event.type === 'prompt' ? 'enabled · prompt enhanced · not native reasoning' : `enabled · ${event.type || 'native'}`;
    }
    return event.reason === 'model_not_supported' ? 'disabled · model not supported' : `disabled · ${event.reason || 'not requested'}`;
  }
  return JSON.stringify(event);
}

function documentStatusLabel(status) {
  const labels = {
    uploaded: '已上传',
    indexing: '索引中',
    indexed: '已索引',
    failed: '失败',
  };
  return labels[status] || status || '未知';
}

function modelDeleteGuidance(message) {
  const normalizedMessage = message.toLowerCase();
  if (normalizedMessage.includes('model is protected')) {
    return '该模型是默认模型或最后一个可用文本模型，后端已保留该行。请先新增可用模型，或对不再使用的模型执行停用。';
  }
  if (normalizedMessage.includes('model is in use')) {
    return '该模型已被智能体或已发布版本引用，后端已保留该行。请改用停用，让它从新建选择列表隐藏，同时保留历史记录。';
  }
  return message;
}



















function Panel({ title, icon, children }) {
  return (
    <section className="panel">
      <h3>{icon}{title}</h3>
      {children}
    </section>
  );
}



function NavButton({ active, icon, label, onClick }) {
  return (
    <button type="button" className={active ? 'active' : ''} onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function ConfigRow({ label, children }) {
  return (
    <div className="config-row">
      <span>{label}</span>
      <div>{children}</div>
    </div>
  );
}

function Segmented({ options, value, onChange }) {
  return (
    <div className="segmented">
      {options.map((option) => (
        <button key={option} type="button" className={value === option ? 'active' : ''} onClick={() => onChange(option)}>
          {option}
        </button>
      ))}
    </div>
  );
}

function Toggle({ checked, disabled = false, label, onChange }) {
  return (
    <button type="button" className={`toggle-switch ${checked ? 'on' : ''}`} disabled={disabled} onClick={() => onChange(!checked)}>
      <span />
      {label}
    </button>
  );
}











function parseMarkdownBlocks(content) {
  const blocks = [];
  const lines = String(content || '').split('\n');
  let buffer = [];
  let code = [];
  let language = '';
  let inCode = false;
  for (const line of lines) {
    const match = line.match(/^```(\w+)?\s*$/);
    if (match) {
      if (inCode) {
        blocks.push({ type: 'code', language, content: code.join('\n') });
        code = [];
        language = '';
        inCode = false;
      } else {
        if (buffer.length) {
          blocks.push({ type: 'text', content: buffer.join('\n') });
          buffer = [];
        }
        language = match[1] || '';
        inCode = true;
      }
      continue;
    }
    if (inCode) code.push(line);
    else buffer.push(line);
  }
  if (inCode) blocks.push({ type: 'code', language, content: code.join('\n') });
  if (buffer.length) blocks.push({ type: 'text', content: buffer.join('\n') });
  return blocks.length ? blocks : [{ type: 'text', content: '' }];
}

function renderMarkdownLines(content, keyPrefix) {
  const lines = String(content || '').split('\n');
  const elements = [];
  let listItems = [];
  function flushList() {
    if (!listItems.length) return;
    elements.push(<ul key={`${keyPrefix}-ul-${elements.length}`}>{listItems.map((item, index) => <li key={index}>{renderInlineMarkdown(item)}</li>)}</ul>);
    listItems = [];
  }
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      elements.push(<br key={`${keyPrefix}-br-${index}`} />);
      return;
    }
    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushList();
      const Tag = `h${heading[1].length + 2}`;
      elements.push(<Tag key={`${keyPrefix}-h-${index}`}>{renderInlineMarkdown(heading[2])}</Tag>);
      return;
    }
    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      listItems.push(bullet[1]);
      return;
    }
    flushList();
    elements.push(<p key={`${keyPrefix}-p-${index}`}>{renderInlineMarkdown(trimmed)}</p>);
  });
  flushList();
  return <React.Fragment key={keyPrefix}>{elements}</React.Fragment>;
}

function renderInlineMarkdown(text) {
  const parts = String(text).split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>;
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index}>{part.slice(2, -2)}</strong>;
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="fatal-error" style={{ padding: '20px', maxWidth: '800px', margin: '50px auto' }}>
          <h1>页面渲染失败</h1>
          <p style={{ color: '#d9383a', fontWeight: 'bold' }}>{this.state.error.message}</p>
          <pre style={{ textAlign: 'left', background: '#fafafa', border: '1px solid #eaeaea', borderRadius: '6px', padding: '15px', overflow: 'auto', fontSize: '11px', lineHeight: '1.5', fontFamily: 'monospace', color: '#333' }}>
            {this.state.error.stack}
          </pre>
          <button type="button" onClick={() => window.location.reload()} style={{ marginTop: '15px', padding: '8px 16px', background: '#181b25', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>刷新页面</button>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(<AppErrorBoundary><App /></AppErrorBoundary>);
