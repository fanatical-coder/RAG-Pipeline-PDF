
export default {
  bootstrap: () => import('./main.server.mjs').then(m => m.default),
  inlineCriticalCss: true,
  baseHref: '/',
  locale: undefined,
  routes: [
  {
    "renderMode": 2,
    "route": "/"
  }
],
  entryPointToBrowserMapping: undefined,
  assets: {
    'index.csr.html': {size: 544, hash: '11cbbfa43d636db4d0e251a406108394a175e42774c7dacedc786b11efb3f79b', text: () => import('./assets-chunks/index_csr_html.mjs').then(m => m.default)},
    'index.server.html': {size: 946, hash: 'd26456df412d660408a9773e2fb5233fe29ed0b31d9173aa399a13dd7bd4017f', text: () => import('./assets-chunks/index_server_html.mjs').then(m => m.default)},
    'index.html': {size: 7742, hash: '902169f05e1a6462b0f26155f3832f0bc06dca8f139efbef3d7ce16bb7320fbd', text: () => import('./assets-chunks/index_html.mjs').then(m => m.default)},
    'styles-G6XLB6NH.css': {size: 96, hash: 'AN8aoBtR9ZA', text: () => import('./assets-chunks/styles-G6XLB6NH_css.mjs').then(m => m.default)}
  },
};
