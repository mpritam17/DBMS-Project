const fs = require('fs');
const file = 'mern/backend/server.js';
let content = fs.readFileSync(file, 'utf8');

content = content.replace(/pull\("diskWrites".*?\\n/g, `pull("diskWrites", /writes=(\\d+)/);\n\n  const neighbors = [];\n  const neighborLines = output.match(/\\{ "id": (\\d+), "distance": ([0-9.]+) \\}/g);\n  if (neighborLines) {\n    for (const matchLine of neighborLines) {\n      const match = matchLine.match(/\\{ "id": (\\d+), "distance": ([0-9.]+) \\}/);\n      if (match) {\n        neighbors.push({\n          id: Number(match[1]),\n          distance: Number(match[2]),\n        });\n      }\n    }\n  }\n`);
content = content.replace(/return \{ metrics, results, raw: output \}/g, `return { metrics, results, neighbors, raw: output }`);

fs.writeFileSync(file, content);
