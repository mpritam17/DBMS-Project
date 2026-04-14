import { useCallback, useEffect, useRef, useState } from "react";

function ImageSearchTab() {
  const [k, setK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Check service status on mount
  useEffect(() => {
    fetch("/api/image-search/status")
      .then((res) => res.json())
      .then((data) => setServiceStatus(data))
      .catch(() => setServiceStatus({ ok: false, service: "unavailable" }));
  }, []);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError("");
    setResults(null);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("k", String(k));

      const response = await fetch("/api/image-search", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Search failed");
      }

      setResults(data);
    } catch (err) {
      setError(err.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h3 style={{ margin: 0 }}>Upload Query Image</h3>
          {serviceStatus && (
            <span className={`service-status ${serviceStatus.ok ? "available" : "unavailable"}`}>
              {serviceStatus.ok ? "✓ Service Online" : "✗ Service Offline"}
            </span>
          )}
        </div>

        {!serviceStatus?.ok && (
          <p style={{ color: "#856404", background: "#fff3cd", padding: 12, borderRadius: 8 }}>
            Start the image search service:
            <br />
            <code style={{ fontSize: 12 }}>
              python scripts/image_search_api.py --vec-file data/sample_vecs.bin --image-dir data/sample_images
            </code>
          </p>
        )}

        <form onSubmit={submit}>
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => handleFileSelect(e.target.files[0])}
            />
            {previewUrl ? (
              <div>
                <img src={previewUrl} alt="Preview" className="preview-image" />
                <p style={{ margin: "8px 0 0", color: "#666" }}>{selectedFile?.name}</p>
              </div>
            ) : (
              <div>
                <p style={{ margin: 0, fontSize: 18 }}>📷 Drop an image here or click to browse</p>
                <p style={{ margin: "8px 0 0", color: "#666", fontSize: 14 }}>Supports JPG, PNG, BMP</p>
              </div>
            )}
          </div>

          <div style={{ display: "flex", gap: 12, marginTop: 12, alignItems: "end" }}>
            <label style={{ flex: "0 0 100px" }}>
              k (neighbors)
              <input type="number" min="1" max="100" value={k} onChange={(e) => setK(Number(e.target.value))} />
            </label>
            <button type="submit" disabled={loading || !selectedFile || !serviceStatus?.ok}>
              {loading ? "Searching..." : "Find Similar Images"}
            </button>
          </div>
        </form>
      </div>

      {error && <p className="error">{error}</p>}

      {results && (
        <>
          <section className="card">
            <h3 style={{ marginTop: 0 }}>⏱️ Performance Timing</h3>
            <div className="timing-grid">
              <div className="timing-item">
                <div className="label">Image Load</div>
                <div className="value">{results.timing.imageLoad_ms?.toFixed(1)} ms</div>
              </div>
              <div className="timing-item">
                <div className="label">Embedding</div>
                <div className="value">{results.timing.embedding_ms?.toFixed(1)} ms</div>
              </div>
              {results.timing.pca_ms !== undefined && (
                <div className="timing-item">
                  <div className="label">PCA Transform</div>
                  <div className="value">{results.timing.pca_ms?.toFixed(2)} ms</div>
                </div>
              )}
              <div className="timing-item">
                <div className="label">KNN Search</div>
                <div className="value">{results.timing.knnSearch_ms?.toFixed(2)} ms</div>
              </div>
              <div className="timing-item">
                <div className="label">Total</div>
                <div className="value">{results.timing.total_ms?.toFixed(1)} ms</div>
              </div>
            </div>
            <p style={{ marginBottom: 0, color: "#666", fontSize: 13 }}>
              Dimensions: {results.dims} {results.pcaEnabled && "(PCA enabled)"}
            </p>
          </section>

          <section className="card">
            <h3 style={{ marginTop: 0 }}>🔍 Top {results.k} Similar Images</h3>
            <div className="results-grid">
              {results.results.map((result, idx) => (
                <div key={result.id} className="result-card">
                  <img
                    src={result.imageUrl}
                    alt={`Result ${idx + 1}`}
                    onError={(e) => {
                      e.target.style.display = "none";
                    }}
                  />
                  <div className="meta">
                    <strong>#{idx + 1}</strong> ID: {result.id}
                    <br />
                    Dist: {result.distance.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </>
  );
}

function QueryBenchmarkTab() {
  const [queryId, setQueryId] = useState("0");
  const [imageFile, setImageFile] = useState(null);
  const [k, setK] = useState(10);
  const [dbPath, setDbPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState(null);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    let timeoutId;

    try {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes

      let response;
      if (imageFile) {
        const formData = new FormData();
        formData.append("image", imageFile);
        formData.append("k", String(k));
        if (dbPath.trim().length > 0) {
          formData.append("dbPath", dbPath.trim());
        }
        
        response = await fetch("/api/query-image", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      } else {
        const normalizedQuery = queryId.trim().toLowerCase();
        const requestBody = {
          queryId: normalizedQuery === "all" ? "all" : Number(queryId),
          k: Number(k),
        };
        if (dbPath.trim().length > 0) {
          requestBody.dbPath = dbPath.trim();
        }

        response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
      }
      clearTimeout(timeoutId);

      const rawText = await response.text();
      let data = null;
      if (rawText.trim().length > 0) {
        try {
          data = JSON.parse(rawText);
        } catch {
          throw new Error(`Server returned non-JSON response (status ${response.status})`);
        }
      }

      if (!data) {
        throw new Error(`Server returned empty response (status ${response.status})`);
      }

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Query failed");
      }
      setPayload(data);
    } catch (err) {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (err?.name === "AbortError") {
        setError("Request timed out in UI. Try a smaller DB (for example large_3000.db) or single query id.");
      } else {
        setError(err.message);
      }
      setPayload(null);
    } finally {
      setLoading(false);
    }
  };

  const metrics = payload?.data?.metrics || {};
  const rows = payload?.data?.results || [];

  return (
    <>
      <form className="card" onSubmit={submit}>
        <label>
          Custom Image Query (overrides Query ID)
          <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files[0])} />
        </label>

        <label>
          Query ID (number or <code>all</code>)
          <input value={queryId} disabled={!!imageFile} onChange={(e) => setQueryId(e.target.value)} />
        </label>

        <label>
          k
          <input type="number" min="1" value={k} onChange={(e) => setK(e.target.value)} />
        </label>

        <label>
          DB Path
          <input value={dbPath} placeholder="(optional) e.g. sample.db" onChange={(e) => setDbPath(e.target.value)} />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Running..." : "Run Query"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {payload && (
        <section className="card">
          <h2>Metrics</h2>
          <div className="grid">
            <div>
              <strong>R-tree us:</strong> {metrics.rtreeUs}
            </div>
            <div>
              <strong>Brute us:</strong> {metrics.bruteUs}
            </div>
            <div>
              <strong>Recall:</strong> {metrics.recall}
            </div>
            <div>
              <strong>Dims:</strong> {metrics.dims}
            </div>
            <div>
              <strong>Unique vectors:</strong> {metrics.vectorsUnique}
            </div>
            <div>
              <strong>Disk writes:</strong> {metrics.diskWrites}
            </div>
          </div>
          {Number(metrics.vectorsUnique || 0) < 1000 && (
            <p>
              Dataset is small ({metrics.vectorsUnique} vectors). Brute-force can be faster at this scale due to lower
              overhead.
            </p>
          )}
        </section>
      )}

      {rows.length > 0 && (
        <section className="card">
          <h2>Per-query Rows</h2>
          <table>
            <thead>
              <tr>
                <th>Query ID</th>
                <th>R-tree (us)</th>
                <th>Brute (us)</th>
                <th>Recall</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={`${row.queryId}-${idx}`}>
                  <td>{row.queryId}</td>
                  <td>{row.rtreeUs}</td>
                  <td>{row.bruteUs}</td>
                  <td>{row.recall}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}

function ExactPointTab() {
  const [mode, setMode] = useState("single");
  const [queryId, setQueryId] = useState("0");
  const [allLimit, setAllLimit] = useState("200");
  const [dbPath, setDbPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState(null);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      let pointSelector;
      if (mode === "single") {
        const numericId = Number(queryId);
        if (!Number.isInteger(numericId) || numericId < 0) {
          throw new Error("Point Query ID must be a non-negative integer");
        }
        pointSelector = numericId;
      } else {
        const trimmed = allLimit.trim();
        if (trimmed.length === 0) {
          pointSelector = "all";
        } else {
          const limit = Number(trimmed);
          if (!Number.isInteger(limit) || limit < 1) {
            throw new Error("All-limit must be >= 1");
          }
          pointSelector = `all:${limit}`;
        }
      }

      const requestBody = { queryId: pointSelector };
      if (dbPath.trim().length > 0) {
        requestBody.dbPath = dbPath.trim();
      }

      const response = await fetch("/api/point-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Exact-point query failed");
      }

      setPayload(data);
    } catch (err) {
      setError(err.message);
      setPayload(null);
    } finally {
      setLoading(false);
    }
  };

  const metrics = payload?.data?.metrics || {};
  const rows = payload?.data?.results || [];

  return (
    <>
      <form className="card" onSubmit={submit}>
        <label>
          Mode
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="single">Single Point ID</option>
            <option value="all">Benchmark All / All:N</option>
          </select>
        </label>

        {mode === "single" ? (
          <label>
            Point Query ID
            <input value={queryId} onChange={(e) => setQueryId(e.target.value)} />
          </label>
        ) : (
          <label>
            All Limit (optional)
            <input
              value={allLimit}
              placeholder="Leave blank for all"
              onChange={(e) => setAllLimit(e.target.value)}
            />
          </label>
        )}

        <label>
          DB Path
          <input value={dbPath} placeholder="(optional) e.g. sample.db" onChange={(e) => setDbPath(e.target.value)} />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Running..." : "Run Exact Point Query"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {payload && (
        <section className="card">
          <h2>Exact-Point Metrics</h2>
          <div className="grid">
            <div>
              <strong>Exact point us:</strong> {metrics.pointUs}
            </div>
            <div>
              <strong>Brute exact us:</strong> {metrics.pointBruteUs}
            </div>
            <div>
              <strong>Match rate:</strong> {metrics.pointMatchRate}
            </div>
            <div>
              <strong>Exact matches:</strong> {metrics.pointExactMatches}/{metrics.pointQueries}
            </div>
            <div>
              <strong>BPM hit rate %:</strong> {metrics.bpmHitRatePercent}
            </div>
            <div>
              <strong>Dims:</strong> {metrics.dims}
            </div>
          </div>
        </section>
      )}

      {rows.length > 0 && (
        <section className="card">
          <h2>Per-query Exact Rows</h2>
          <table>
            <thead>
              <tr>
                <th>Query ID</th>
                <th>Point (us)</th>
                <th>Brute (us)</th>
                <th>R-tree hits</th>
                <th>Brute hits</th>
                <th>Exact Match</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={`${row.queryId}-${idx}`}>
                  <td>{row.queryId}</td>
                  <td>{row.pointUs}</td>
                  <td>{row.bruteUs}</td>
                  <td>{row.rtreeHits}</td>
                  <td>{row.bruteHits}</td>
                  <td>{row.exactMatch ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("image-search");

  return (
    <div className="page">
      <header>
        <h1>Image Similarity Search</h1>
        <p>Upload an image to find similar images using KNN with R-tree indexing and PCA dimensionality reduction.</p>
      </header>

      <div className="tabs">
        <button className={activeTab === "image-search" ? "active" : ""} onClick={() => setActiveTab("image-search")}>
          🖼️ Image Search
        </button>
        <button className={activeTab === "benchmark" ? "active" : ""} onClick={() => setActiveTab("benchmark")}>
          📊 Query Benchmark
        </button>
        <button className={activeTab === "point" ? "active" : ""} onClick={() => setActiveTab("point")}>
          🎯 Exact Point
        </button>
      </div>

      {activeTab === "image-search" && <ImageSearchTab />}
      {activeTab === "benchmark" && <QueryBenchmarkTab />}
      {activeTab === "point" && <ExactPointTab />}
    </div>
  );
}
