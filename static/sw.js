const CACHE_NAME = 'smarthealth-v1';
const urlsToCache = [
  '/',
  '/dashboard',
  '/translator',
  '/triage',
  '/diagnosis',
  '/referral',
  '/static/manifest.json'
];

// 1. Install: Cache the core pages
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
});

// 2. Fetch: Serve from cache if offline, otherwise fetch from network
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version if found
        if (response) {
          return response;
        }
        // Otherwise, fetch from network
        return fetch(event.request);
      }
    )
  );
});

// 3. Activate: Clean up old caches if we update the app
self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
