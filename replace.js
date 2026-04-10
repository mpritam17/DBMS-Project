const fs = require('fs');
const file = 'mern/backend/server.js';
let content = fs.readFileSync(file, 'utf8');

const regex = /return res\.json\(\{[\s\S]*?ok: true,[\s\S]*?k,[\s\S]*?results: parsed\.neighbors\.map[\s\S]*?\}\);/m;

const replacement = `return res.json({
      ok: true,
      k,
      results: parsed.neighbors.map((n) => ({
        id: n.id,
        distance: n.distance,
        imageUrl: \`/api/images/\${n.id}\`,
      })),
      timing: {
        imageLoad_ms: 0,
        embedding_ms: 0,
        pca_ms: 0,
        knnSearch_ms: parsed.metrics.rtreeUs / 1000.0,
        total_ms: parsed.metrics.rtreeUs / 1000.0,
        rtreeUs: parsed.metrics.rtreeUs,
        bruteUs: parsed.metrics.bruteUs,
      },
      dims: dbDims,
      pcaEnabled: extractData.pca_dims === dbDims,
    });`;

content = content.replace(regex, replacement);
fs.writeFileSync(file, content);
