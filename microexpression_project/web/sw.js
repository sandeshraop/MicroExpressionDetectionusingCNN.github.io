// Service Worker for Micro-Expression Recognition Web App
// Provides offline functionality and PWA features

const CACHE_NAME = 'micro-expression-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/real_main.js',
  '/model-report-graphs/per_subject_accuracy.png',
  '/model-report-graphs/confusion_matrix_counts.png',
  '/model-report-graphs/confusion_matrix_row_normalized.png',
  '/model-report-graphs/confusion_matrix_col_normalized.png',
  '/model-report-graphs/roc_ovr.png',
  '/model-report-graphs/pr_ovr.png',
  '/model-report-graphs/calibration_top1.png',
  '/model-report-graphs/per_class_f1.png',
  '/model-report-graphs/per_class_precision.png',
  '/model-report-graphs/per_class_recall.png'
];

// Install service worker
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Service Worker: Caching files');
        return cache.addAll(urlsToCache);
      })
      .catch(error => {
        console.log('Service Worker: Failed to cache files', error);
      })
  );
});

// Activate service worker
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cache => {
          if (cache !== CACHE_NAME) {
            console.log('Service Worker: Clearing old cache');
            return caches.delete(cache);
          }
        })
      );
    })
  );
});

// Fetch event - serve cached content when offline
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        if (response) {
          return response;
        }
        
        return fetch(event.request)
          .catch(() => {
            // Return a custom offline page for HTML requests
            if (event.request.headers.get('accept').includes('text/html')) {
              return new Response(`
                <html>
                  <body>
                    <h1>Offline</h1>
                    <p>The micro-expression recognition web app is currently offline.</p>
                    <p>Please check your internet connection.</p>
                  </body>
                </html>
              `, {
                headers: { 'Content-Type': 'text/html' }
              });
            }
          });
      })
  );
});

// Background sync for offline uploads (if needed)
self.addEventListener('sync', event => {
  if (event.tag === 'background-sync') {
    event.waitUntil(
      // Handle background sync operations
      console.log('Service Worker: Background sync triggered')
    );
  }
});

// Push notifications (if needed)
self.addEventListener('push', event => {
  const options = {
    body: event.data ? event.data.text() : 'New notification from Micro-Expression Recognition',
    icon: '/favicon.ico',
    badge: '/favicon.ico',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    }
  };

  event.waitUntil(
    self.registration.showNotification('Micro-Expression Recognition', options)
  );
});

console.log('Service Worker: Loaded successfully');
