// 앱 셸(정적 파일)만 캐시한다. API 호출(POST, 외부 도메인)은 건드리지 않는다.
// persona/sw.js와 같은 네트워크 우선 전략: 온라인이면 항상 최신 버전, 오프라인일 때만 캐시.
const CACHE_NAME = 'letters-v2';
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
  // GitHub Pages는 파일을 최대 10분 동안 HTTP 캐시에 두게 해서, 그냥 fetch하면 배포 직후에도 예전 파일이 온다.
  // 'no-cache'로 매번 서버에 새 버전이 있는지 확인하게 한다(바뀐 게 없으면 304라 가볍다).
  e.respondWith(
    fetch(req.url, { cache: 'no-cache', credentials: 'same-origin' }).then(res => {
      if (res && res.ok) {
        const copy = res.clone();
        caches.open(CACHE_NAME).then(c => c.put(req, copy));
      }
      return res;
    }).catch(() => caches.match(req))
  );
});

// 알림을 누르면 열려 있는 앱 창으로 돌아간다.
self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
      for (const c of list) { if ('focus' in c) return c.focus(); }
      return self.clients.openWindow('./');
    })
  );
});
