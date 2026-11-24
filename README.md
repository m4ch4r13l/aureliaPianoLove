# Aurelia — Piano de Cola 3D

Aplicación web 100% estática hecha con React, TypeScript, Three.js y Web Audio API. Permite subir archivos MP3 para calcular automáticamente las notas del piano y la melodía vocal y reproducirlas en un piano de cola 3D interactivo.

## Publicar en GitHub Pages

1. Sube este proyecto a tu repositorio de GitHub en la rama `main`.
2. En tu repositorio ve a **Settings → Pages**.
3. En **Build and deployment → Source**, selecciona **GitHub Actions**.
4. Cada `git push` a `main` compilará y publicará automáticamente el sitio usando `.github/workflows/deploy.yml`.

## Desarrollo local

```bash
npm install
npm run dev
```

## Compilar versión estática

```bash
npm run build
```

Los archivos listos para publicación se generan en la carpeta `dist/` con rutas relativas (`./`), `.nojekyll` y `404.html`.
