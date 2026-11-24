import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import {defineConfig} from 'vite';

const BASIC_PITCH_MODEL_DIR = path.resolve(
  __dirname,
  'node_modules/@spotify/basic-pitch/model',
);

export default defineConfig(() => {
  return {
    base: './',
    plugins: [
      react(),
      tailwindcss(),
      // Serves @spotify/basic-pitch TFJS model files in dev and bundles them into
      // dist/basic-pitch-model/ for GitHub Pages & production builds (zero CDN needed).
      {
        name: 'basic-pitch-model-assets',
        configureServer(server) {
          server.middlewares.use('/basic-pitch-model', (req, res, next) => {
            const rawUrl = (req.url || '/').split('?')[0].replace(/^\/+/, '');
            if (!rawUrl || rawUrl.includes('..')) {
              next();
              return;
            }
            const filePath = path.join(BASIC_PITCH_MODEL_DIR, rawUrl);
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              res.setHeader(
                'Content-Type',
                rawUrl.endsWith('.json')
                  ? 'application/json; charset=utf-8'
                  : 'application/octet-stream',
              );
              res.setHeader('Cache-Control', 'public, max-age=3600');
              fs.createReadStream(filePath).pipe(res);
              return;
            }
            next();
          });
        },
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(__dirname, 'dist');
          if (!fs.existsSync(distDir)) {
            return;
          }
          if (fs.existsSync(BASIC_PITCH_MODEL_DIR)) {
            fs.cpSync(
              BASIC_PITCH_MODEL_DIR,
              path.join(distDir, 'basic-pitch-model'),
              {recursive: true, force: true},
            );
          }
        },
      },
      // Copies src/assets into dist/ and prepares GitHub Pages static files (.nojekyll, 404.html).
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(__dirname, 'dist');
          const srcAssets = path.resolve(__dirname, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, {recursive: true, force: true});
          }
          if (fs.existsSync(distDir)) {
            fs.writeFileSync(path.join(distDir, '.nojekyll'), '');
            const indexHtml = path.join(distDir, 'index.html');
            if (fs.existsSync(indexHtml)) {
              fs.copyFileSync(indexHtml, path.join(distDir, '404.html'));
            }
          }
        },
      },
    ],
    resolve: {
      dedupe: ['@tensorflow/tfjs', '@tonejs/midi'],
      alias: {
        '@': path.resolve(__dirname, '.'),
        // Deduplicate nested TFJS inside @spotify/basic-pitch to single root @tensorflow/tfjs
        '@tensorflow/tfjs': path.resolve(__dirname, 'node_modules/@tensorflow/tfjs'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
