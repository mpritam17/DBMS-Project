# Week 4 MERN Integration

This folder contains a MERN scaffold that integrates your C++ query engine.

## Backend (Express + optional MongoDB)

Path: `mern/backend`

1. Install dependencies:

```bash
cd mern/backend
npm install
```

2. Configure env:

```bash
cp .env.example .env
# edit MONGODB_URI if you want persistence
```

3. Run backend:

```bash
npm run dev
```

Endpoints:

- `GET /api/health`
- `POST /api/query`
- `GET /api/query-logs` (returns empty when MongoDB is not connected)

Sample payload:

```json
{
  "queryId": 0,
  "k": 10,
  "dbPath": "/home/quantumec/Documents/DBMS_term_project/sample.db"
}
```

## Frontend (React + Vite)

Path: `mern/frontend`

1. Install dependencies:

```bash
cd mern/frontend
npm install
```

2. Start UI:

```bash
npm run dev
```

Vite proxies `/api` to `http://localhost:5000`.

## Notes

- The backend executes `build/week4_query_benchmark` for each query.
- MongoDB is optional. If `MONGODB_URI` is unset, backend still works and skips persistence.
