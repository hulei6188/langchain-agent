export const THEME_STORAGE_KEY = 'agentbase_theme';
export const SIDEBAR_COLLAPSED_STORAGE_KEY = 'agentbase_sidebar_collapsed';
export const CHAT_WELCOME_SNAPSHOT_STORAGE_KEY = 'agentbase_chat_welcome_snapshot';

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

export function initialThemeMode() {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  return THEME_MODES.includes(stored) ? stored : 'light';
}

export function initialSidebarHidden() {
  if (typeof window === 'undefined') return false;
  return localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === 'true';
}

export function readChatWelcomeSnapshot() {
  if (typeof window === 'undefined') return null;
  try {
    const value = localStorage.getItem(CHAT_WELCOME_SNAPSHOT_STORAGE_KEY);
    if (!value) return null;
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

export function sameRouteId(left, right) {
  if (left === null || left === undefined || right === null || right === undefined) return false;
  return String(left) === String(right);
}

export function normalizeAppView(value) {
  return value === 'builder' ? 'builder' : 'home';
}

export function normalizeAppNav(value, view = 'home') {
  if (view === 'builder') return 'agents';
  return APP_NAV_VALUES.has(value) ? value : 'chat';
}

export function isChatHomeRoute(route) {
  return route?.view === 'home' && route?.nav === 'chat';
}

export function readChatRoute() {
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

export function writeChatRoute(agentId, sessionId = null, options = {}) {
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

export function clearChatRoute(options = {}) {
  writeChatRoute(null, null, options);
}

export function writePageRoute(view, nav, options = {}) {
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

export function systemPrefersDark() {
  return Boolean(typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
}
