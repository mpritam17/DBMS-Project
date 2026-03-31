# End-to-End Image Search and MERN Stack

This milestone introduces a complete end-to-end image search application, combining our custom R-Tree and storage engine with a modern web stack.

## Architecture

1. **Storage & Indexing Core**: The C++ engine handles binary R-Tree indexing and fixed-size 4KB slotted pages.
2. **Python Image Search API**: A robust Python Flask API (`scripts/image_search_api.py`) exposing the machine learning logic, taking image files, applying ResNet-18 embeddings, reducing via PCA, and performing nearest neighbor retrieval.
3. **MERN Backend**: An Express/Node.js server orchestrating requests and tracking the query benchmarks.
4. **React Frontend**: A Vite-based React application allowing users to upload prompt queried images directly to match against the dataset visually with computed distance scoring and timing.

## Recent Fixes & Scalability Improvements

### 1. 60,000 Image Support
The dataset initialization script (`scripts/populate_database.py`) was updated to synthesize both the CIFAR-10 `train` and `test` splits, extending the dataset support to a massive **60,000 images**.

- Lexicographical sorting flaws for file names (`img_10000.png` bypassing `img_1000.png`) were resolved by forcefully extracting unique integer IDs from the file names within the pipeline, ensuring `sample.db` IDs correctly align 1:1 with the queried image rendering.

### 2. High Accuracy 64D Embeddings
Previous iterations used a heavy 12-Dimensional PCA compression causing visual feature loss (e.g. crossing deer with horses). The PCA threshold was increased to **64-Dimensions**, retaining `>85%` dataset variance resolving cross-class contamination while still achieving massive runtime enhancements versus brute-force 512D ResNet-18 logic.

### 3. Deterministic Network Initialization
A bug in PyTorch where the `torch.nn.Linear` projection layers were instantiating random unseeded weights across the CLI Extractor and the API server was squashed using unified initialization seeding (`torch.manual_seed(42)`).

## Usage

### Populate the Dataset & Database
Downloads and parses 60,000 CIFAR-10 photos into a 64D database.
```bash
source .venv/bin/activate
python scripts/populate_database.py --source cifar10 --count 60000 --pca-dims 64
./build/bulk_load data/sample_vecs.bin sample.db
```

### Start the Python ML API
Launch the Flask model processor (default port 5001):
```bash
source .venv/bin/activate
python scripts/image_search_api.py --vec-file data/sample_vecs.bin --image-dir data/sample_images --pca-model data/pca_model.npz
```

### Start the MERN Environment
Run sequentially in individual terminals:
```bash
cd mern/backend
npm install
npm run dev
```

```bash
cd mern/frontend
npm install
npm run dev
```

Visit the displayed local host (usually `http://localhost:5173/`) to access the Image Search UI.
