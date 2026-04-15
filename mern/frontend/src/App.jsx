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
              {results.timing.pointSearch_ms !== undefined && (
                <div className="timing-item">
                  <div className="label">R-tree Point Search</div>
                  <div className="value">{results.timing.pointSearch_ms?.toFixed(2)} ms</div>
                </div>
              )}
              <div className="timing-item">
                <div className="label">R-tree KNN</div>
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
            {results.searchDiagnostics && (
              <div style={{ marginTop: 8, fontSize: 13, color: "#455a64" }}>
                <p style={{ margin: 0 }}>
                  {results.searchDiagnostics.pointMatchCount > 0
                    ? `Exact point match found in R-tree (${results.searchDiagnostics.pointMatchCount}). R-tree KNN fills remaining neighbors.`
                    : "No exact point match in R-tree. Returning nearest neighbors from R-tree KNN."}
                </p>
                <p style={{ margin: "4px 0 0" }}>
                  Point nodes: {results.searchDiagnostics.pointNodesVisited}, entries: {results.searchDiagnostics.pointEntriesExamined},
                  method: {results.searchDiagnostics.method}
                </p>
                {results.searchDiagnostics.pointSearchError && (
                  <p style={{ margin: "4px 0 0", color: "#b00020" }}>
                    Point-search feedback unavailable: {results.searchDiagnostics.pointSearchError}
                  </p>
                )}
              </div>
            )}
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
                    {result.source && (
                      <>
                        Source: {result.source}
                        <br />
                      </>
                    )}
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
  const [insertImageFile, setInsertImageFile] = useState(null);
  const [insertId, setInsertId] = useState("auto");
  const [insertLoading, setInsertLoading] = useState(false);
  const [insertError, setInsertError] = useState("");
  const [insertPayload, setInsertPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState(null);

  const submitInsert = async (event) => {
    event.preventDefault();
    setInsertLoading(true);
    setInsertError("");

    try {
      if (!insertImageFile) {
        throw new Error("Please select an image for insertion");
      }

      const rawId = insertId.trim();
      let idValue = "auto";
      if (rawId && rawId.toLowerCase() !== "auto") {
        const parsed = Number(rawId);
        if (!Number.isInteger(parsed) || parsed < 0) {
          throw new Error("ID must be 'auto' or a non-negative integer");
        }
        idValue = parsed;
      }

      const formData = new FormData();
      formData.append("image", insertImageFile);
      formData.append("id", String(idValue));
      if (dbPath.trim().length > 0) {
        formData.append("dbPath", dbPath.trim());
      }

      const response = await fetch("/api/incremental-insert", {
        method: "POST",
        body: formData,
      });

      const rawText = await response.text();
      let data = null;
      if (rawText.trim().length > 0) {
        data = JSON.parse(rawText);
      }

      if (!response.ok || !data?.ok) {
        throw new Error(data?.error || "Incremental insert failed");
      }

      setInsertPayload(data);
    } catch (err) {
      setInsertPayload(null);
      setInsertError(err.message || "Incremental insert failed");
    } finally {
      setInsertLoading(false);
    }
  };

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
  const rtreeUs = Number(metrics.rtreeUs);
  const kdUs = Number(metrics.kdUs);
  const bruteUs = Number(metrics.bruteUs);
  const latencyMethods = [
    { name: "R-tree", value: rtreeUs },
    { name: "KD-tree", value: kdUs },
    { name: "Brute-force", value: bruteUs },
  ].filter((method) => Number.isFinite(method.value) && method.value > 0);
  const sortedMethods = [...latencyMethods].sort((left, right) => left.value - right.value);
  const hasLatencyComparison = sortedMethods.length >= 2;
  const winner = hasLatencyComparison ? sortedMethods[0].name : "N/A";
  const runnerUp = hasLatencyComparison ? sortedMethods[1] : null;
  const deltaUs = hasLatencyComparison ? (runnerUp.value - sortedMethods[0].value) : 0;
  const speedup = hasLatencyComparison ? (runnerUp.value / sortedMethods[0].value) : 0;
  const fasterPct = hasLatencyComparison ? ((deltaUs / runnerUp.value) * 100) : 0;

  return (
    <>
      <form className="card" onSubmit={submitInsert}>
        <h2 style={{ marginTop: 0 }}>Incremental Insert</h2>
        <p style={{ marginTop: 0, color: "#666" }}>
          Insert one image into the current DB: the backend extracts the embedding, appends it to the store, updates
          the persistent R-tree index, and keeps query caches in sync.
        </p>

        <label>
          Image File
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setInsertImageFile(e.target.files?.[0] || null)}
          />
        </label>

        <label>
          Insert ID ("auto" or integer)
          <input value={insertId} onChange={(e) => setInsertId(e.target.value)} />
        </label>

        <label>
          DB Path (shared with query form below)
          <input value={dbPath} placeholder="(optional) e.g. sample.db" onChange={(e) => setDbPath(e.target.value)} />
        </label>

        <button type="submit" disabled={insertLoading || !insertImageFile}>
          {insertLoading ? "Inserting..." : "Insert Image"}
        </button>
      </form>

      {insertError && <p className="error">{insertError}</p>}

      {insertPayload && (
        <section className="card">
          <h2 style={{ marginTop: 0 }}>Incremental Insert Result</h2>
          <div className="grid">
            <div>
              <strong>DB:</strong> {insertPayload.data?.dbFile || insertPayload.request?.dbPath}
            </div>
            <div>
              <strong>Index DB:</strong> {insertPayload.data?.indexDbFile || "N/A"}
            </div>
            <div>
              <strong>Inserted ID:</strong> {insertPayload.data?.id ?? "N/A"}
            </div>
            <div>
              <strong>Dims:</strong> {insertPayload.data?.dims ?? insertPayload.request?.vectorDims}
            </div>
            <div>
              <strong>Vector source:</strong> {insertPayload.request?.vectorSource || "N/A"}
            </div>
            <div>
              <strong>Total vectors:</strong> {insertPayload.data?.vectorsTotal ?? "N/A"}
            </div>
            <div>
              <strong>Index rebuilt:</strong> {insertPayload.data?.rebuilt ? "yes" : "no"}
            </div>
            <div>
              <strong>Cache sync:</strong> {insertPayload.cacheSync?.updated ? "yes" : "no"}
            </div>
            <div>
              <strong>Image saved:</strong> {insertPayload.imageStoredAs || "N/A"}
            </div>
          </div>
        </section>
      )}

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
              <strong>KD-tree us:</strong> {metrics.kdUs ?? "N/A"}
            </div>
            <div>
              <strong>Brute us:</strong> {metrics.bruteUs}
            </div>
            <div>
              <strong>Recall:</strong> {metrics.recall}
            </div>
            <div>
              <strong>KD recall:</strong> {metrics.kdRecall ?? "N/A"}
            </div>
            <div>
              <strong>Point search us:</strong> {metrics.pointSearchUs ?? "N/A"}
            </div>
            <div>
              <strong>Point hit rate:</strong> {metrics.pointHitRate ?? "N/A"}
            </div>
            <div>
              <strong>Point matches/query:</strong> {metrics.pointMatchesPerQuery ?? "N/A"}
            </div>
            <div>
              <strong>Point nodes visited:</strong> {metrics.pointNodesVisited ?? "N/A"}
            </div>
            <div>
              <strong>Point entries examined:</strong> {metrics.pointEntriesExamined ?? "N/A"}
            </div>
            <div>
              <strong>Dims:</strong> {metrics.dims}
            </div>
            <div>
              <strong>Unique vectors:</strong> {metrics.vectorsUnique}
            </div>
            <div>
              <strong>KD build us:</strong> {metrics.kdBuildUs ?? "N/A"}
            </div>
            <div>
              <strong>BPM fetch requests:</strong> {metrics.bpmFetchRequests ?? "N/A"}
            </div>
            <div>
              <strong>BPM hits:</strong> {metrics.bpmFetchHits ?? "N/A"}
            </div>
            <div>
              <strong>BPM misses:</strong> {metrics.bpmFetchMisses ?? "N/A"}
            </div>
            <div>
              <strong>BPM hit-rate:</strong> {metrics.bpmHitRate ?? "N/A"}
            </div>
            <div>
              <strong>Disk writes:</strong> {metrics.diskWrites}
            </div>
            {hasLatencyComparison && (
              <>
                <div>
                  <strong>Winner:</strong> {winner}
                </div>
                <div>
                  <strong>Speedup:</strong> {speedup.toFixed(2)}x
                </div>
                <div>
                  <strong>Latency delta:</strong> {deltaUs.toFixed(2)} us
                </div>
                <div>
                  <strong>Faster by:</strong> {fasterPct.toFixed(2)}%
                </div>
              </>
            )}
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
                <th>KD-tree (us)</th>
                <th>Brute (us)</th>
                <th>Point (us)</th>
                <th>Point Matches</th>
                <th>Best Method</th>
                <th>Delta vs 2nd (us)</th>
                <th>KD Recall</th>
                <th>Recall</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const rowRtree = Number(row.rtreeUs);
                const rowKd = Number(row.kdUs);
                const rowBrute = Number(row.bruteUs);
                const rowMethods = [
                  { name: "R-tree", value: rowRtree },
                  { name: "KD-tree", value: rowKd },
                  { name: "Brute-force", value: rowBrute },
                ].filter((method) => Number.isFinite(method.value) && method.value > 0)
                 .sort((left, right) => left.value - right.value);
                const rowWinner = rowMethods.length >= 2 ? rowMethods[0].name : "N/A";
                const rowDelta = rowMethods.length >= 2 ? (rowMethods[1].value - rowMethods[0].value) : null;

                return (
                  <tr key={`${row.queryId}-${idx}`}>
                    <td>{row.queryId}</td>
                    <td>{row.rtreeUs}</td>
                    <td>{row.kdUs ?? "N/A"}</td>
                    <td>{row.bruteUs}</td>
                    <td>{row.pointUs ?? "N/A"}</td>
                    <td>{row.pointMatches ?? "N/A"}</td>
                    <td>{rowWinner}</td>
                    <td>{rowDelta === null ? "N/A" : rowDelta.toFixed(2)}</td>
                    <td>{row.kdRecall ?? "N/A"}</td>
                    <td>{row.recall}</td>
                  </tr>
                );
              })}
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
      </div>

      {activeTab === "image-search" && <ImageSearchTab />}
      {activeTab === "benchmark" && <QueryBenchmarkTab />}
    </div>
  );
}
