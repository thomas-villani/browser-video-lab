// Files served from public/ by path ('/models/x.onnx', '/ort/'), below the site's base: '/' in
// development and the CLI, '/<repo>/' on GitHub Pages (vite's `base`).
export const asset = (path: string) => (path.startsWith('/') ? import.meta.env.BASE_URL + path.slice(1) : path);
