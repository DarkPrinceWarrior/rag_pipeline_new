UserGuide RAG — Frontend (React + TypeScript)

Commands
- Dev: `npm install && npm run dev` → http://localhost:5173/
- Build: `npm install && npm run build` → outputs to `../app/static/`
- Preview build: `npm run preview`

Notes
- During dev, Vite proxies `/ask`, `/healthz`, and `/docs` to `http://localhost:8000`.
- For production, serve the built UI from FastAPI by running `uvicorn app.main:app --port 8000`.
- PDF is available under `/docs/<file>.pdf` (default: `/docs/User_Guide.pdf`).

