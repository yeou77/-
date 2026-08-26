// 앱 셸(정적 파일)만 캐시한다. API 호출(POST, 외부 도메인)은 절대 건드리지 않는다.
// v1의 캐시 우선(cache-first) 전략은 이 앱이 하루에도 몇 번씩 업데이트되는 상황과 상극이었다 —
// 방문할 때마다 "예전에 캐시해둔 index.html"을 먼저 보여주고 새 버전은 백그라운드에서만 받아놨다가
// 그 다음 방문에야 반영되는 식이라, 새로고침 한 번으론 최신 수정사항이 안 보이는 문제가 계속 생겼다.
// v2부터는 네트워크 우선(network-first)으로 바꾼다: 온라인이면 항상 최신 버전을 받아오고,
// 오프라인일 때만 캐시로 대체한다. CACHE_NAME을 바꿔서 v1의 오래된 캐시도 폐기시킨다.
const CACHE_NAME = 'persona-chat-v2';
const APP_SHELL = ['./', './index.html', './manifest.json', './icons/icon-192.png', './icons/icon-512.png', './icons/apple-touch-icon.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' || !req.url.startsWith(self.location.origin)) return;
  e.respondWith(
    fetch(req).then(res => {
      if (res && res.ok) {
        const copy = res.clone();
        caches.open(CACHE_NAME).then(c => c.put(req, copy));
      }
      return res;
    }).catch(() => caches.match(req))
  );
});
