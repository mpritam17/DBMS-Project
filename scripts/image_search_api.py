#!/usr/bin/env python3
"""
Image Search API - Handles image upload, embedding extraction, PCA, and KNN search.

Endpoints:
- POST /search - Upload image and get k nearest neighbors
- GET /health - Health check
- GET /image/<id> - Serve image by ID
"""

import argparse
import io
import os
import re
import struct
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

DUP_HASH_SIZE = 16
DUP_MAX_DISTANCE = 72
DUP_MIN_GAP = 12

# Global state (initialized in main)
_state = {
    "backbone": None,
    "projector": None,
    "pca": None,
    "device": "cpu",
    "vectors": None,       # numpy array of shape (n, dims)
    "ids": None,           # numpy array of IDs
    "dims": 0,
    "image_dir": None,
    "transform": None,
    "image_hash_index": None,
}


def load_pca_model(pca_path: Path):
    """Load PCA model from .npz file."""
    if not pca_path.exists():
        return None
    data = np.load(pca_path)
    return {
        "components": data["components"].astype(np.float32),
        "mean": data["mean"].astype(np.float32),
    }


def apply_pca(vectors: np.ndarray, pca: dict, normalize: bool = True) -> np.ndarray:
    """Apply PCA transformation to vectors."""
    centered = vectors.astype(np.float32) - pca["mean"]
    transformed = centered @ pca["components"].T
    if normalize:
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        transformed = transformed / norms
    return transformed.astype(np.float32)


def load_vec1_file(vec1_path: Path):
    """Load vectors from VEC1 binary file."""
    with open(vec1_path, "rb") as f:
        magic = f.read(4)
        if magic != b"VEC1":
            raise ValueError(f"Invalid VEC1 file: {vec1_path}")
        
        version = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        dims = struct.unpack("<I", f.read(4))[0]
        _flags = struct.unpack("<I", f.read(4))[0]
        
        ids = []
        vectors = []
        
        for _ in range(count):
            vec_id = struct.unpack("<Q", f.read(8))[0]
            vec_data = np.frombuffer(f.read(dims * 4), dtype=np.float32).copy()
            ids.append(vec_id)
            vectors.append(vec_data)
        
        return np.array(ids, dtype=np.uint64), np.vstack(vectors).astype(np.float32), dims


def extract_embedding(image: Image.Image) -> np.ndarray:
    """Extract embedding from a single image."""
    img_tensor = _state["transform"](image).unsqueeze(0).to(_state["device"])
    
    with torch.no_grad():
        features = _state["backbone"](img_tensor)
        vector = _state["projector"](features)
        vector = torch.nn.functional.normalize(vector, p=2, dim=1)
    
    return vector.cpu().numpy().astype(np.float32)


def knn_search(query_vector: np.ndarray, k: int):
    """
    Perform brute-force KNN search.
    Returns list of (id, distance) tuples sorted by distance.
    """
    # Ensure query is 2D
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # L2 distance
    diff = _state["vectors"] - query_vector
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Get top-k indices
    k = min(k, len(distances))
    top_indices = np.argpartition(distances, k)[:k]
    top_indices = top_indices[np.argsort(distances[top_indices])]
    
    results = []
    for idx in top_indices:
        results.append({
            "id": int(_state["ids"][idx]),
            "distance": float(distances[idx]),
        })
    
    return results


def get_image_path(image_id: int) -> Path | None:
    """Get image path by ID."""
    if _state["image_dir"] is None:
        return None
    
    # Try common naming patterns
    patterns = [
        f"img_{image_id:04d}.png",
        f"img_{image_id:04d}.jpg",
        f"img_{image_id}.png",
        f"img_{image_id}.jpg",
        f"{image_id}.png",
        f"{image_id}.jpg",
    ]
    
    for pattern in patterns:
        path = _state["image_dir"] / pattern
        if path.exists():
            return path
    
    return None


def parse_image_id_from_name(filename: str) -> int | None:
    """Parse numeric image id from common dataset naming patterns."""
    stem = Path(filename).stem

    match = re.match(r"^img_(\d+)$", stem)
    if match:
        return int(match.group(1))

    match = re.match(r"^(\d+)$", stem)
    if match:
        return int(match.group(1))

    return None


def compute_dhash(image: Image.Image, hash_size: int = DUP_HASH_SIZE) -> int:
    """Compute a difference hash (dHash) as a Python int."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.uint8)
    diff = pixels[:, :-1] > pixels[:, 1:]

    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def build_image_hash_index(image_dir: Path) -> list[tuple[int, int]]:
    """Build an in-memory (image_id, dHash) index for near-duplicate lookup."""
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(image_dir.glob(pattern))

    index: list[tuple[int, int]] = []
    for path in sorted(paths):
        image_id = parse_image_id_from_name(path.name)
        if image_id is None:
            continue
        try:
            with Image.open(path) as img:
                image_hash = compute_dhash(img.convert("RGB"))
            index.append((image_id, image_hash))
        except Exception:
            # Skip unreadable/corrupt files and continue building the index.
            continue

    return index


def find_near_duplicate(image: Image.Image) -> dict | None:
    """Find nearest image by dHash and return confidence metadata."""
    if _state["image_dir"] is None:
        return None

    if _state["image_hash_index"] is None:
        _state["image_hash_index"] = build_image_hash_index(_state["image_dir"])
        print(f"Built near-duplicate hash index: {len(_state['image_hash_index'])} images")

    index: list[tuple[int, int]] = _state["image_hash_index"]
    if not index:
        return None

    query_hash = compute_dhash(image)
    best_id = -1
    best_distance = 1 << 30
    second_best_distance = 1 << 30

    for image_id, image_hash in index:
        distance = (query_hash ^ image_hash).bit_count()
        if distance < best_distance:
            second_best_distance = best_distance
            best_distance = distance
            best_id = image_id
        elif distance < second_best_distance:
            second_best_distance = distance

    if best_id < 0:
        return None

    gap = second_best_distance - best_distance if second_best_distance < (1 << 29) else 0
    confident = best_distance <= DUP_MAX_DISTANCE and gap >= DUP_MIN_GAP
    return {
        "id": int(best_id),
        "hashDistance": int(best_distance),
        "secondBestDistance": int(second_best_distance if second_best_distance < (1 << 29) else -1),
        "gap": int(gap),
        "confident": bool(confident),
    }


@app.route("/extract", methods=["POST"])
def extract():
    """Extract an embedding from an uploaded image."""
    import time
    timing = {}
    total_start = time.perf_counter()
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "No image file provided"}), 400
        
        t0 = time.perf_counter()
        image_file = request.files["image"]
        image = Image.open(image_file.stream).convert("RGB")
        timing["imageLoad_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        near_duplicate = find_near_duplicate(image)
        timing["nearDuplicate_ms"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        raw_vector = extract_embedding(image)
        timing["embedding_ms"] = (time.perf_counter() - t0) * 1000
        
        result_map = {
            "vector": raw_vector.flatten().tolist(),  # default (128D)
            "dims": int(raw_vector.shape[1]),
            "timing": timing
        }

        if near_duplicate is not None:
            result_map["near_duplicate"] = near_duplicate

        if _state["pca"] is not None:
            t0 = time.perf_counter()
            pca_vector = apply_pca(raw_vector, _state["pca"])
            timing["pca_ms"] = (time.perf_counter() - t0) * 1000
            result_map["vector_pca"] = pca_vector.flatten().tolist()
            result_map["pca_dims"] = int(pca_vector.shape[1])
            
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000
        # Return as a simple 1D array list
        return jsonify({"ok": True, **result_map})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "image-search-api",
        "vectorCount": len(_state["ids"]) if _state["ids"] is not None else 0,
        "dims": _state["dims"],
        "pcaEnabled": _state["pca"] is not None,
        "hashIndexReady": _state["image_hash_index"] is not None,
        "hashIndexCount": len(_state["image_hash_index"]) if _state["image_hash_index"] is not None else 0,
    })


@app.route("/search", methods=["POST"])
def search():
    """
    Search for similar images.
    
    Accepts:
    - multipart/form-data with 'image' file
    - k: number of neighbors (default: 10)
    
    Returns:
    - results: list of {id, distance, imageUrl}
    - timing: breakdown of time spent in each stage
    """
    total_start = time.perf_counter()
    timing = {}
    
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "No image file provided"}), 400
        
        k = int(request.form.get("k", 10))
        if k < 1:
            return jsonify({"ok": False, "error": "k must be >= 1"}), 400
        
        # Load image
        t0 = time.perf_counter()
        try:
            image_file = request.files["image"]
            image = Image.open(image_file.stream).convert("RGB")
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to load image: {e}"}), 400
        timing["imageLoad_ms"] = (time.perf_counter() - t0) * 1000
        
        # Extract embedding
        t0 = time.perf_counter()
        query_vector = extract_embedding(image)
        timing["embedding_ms"] = (time.perf_counter() - t0) * 1000
        
        # Apply PCA if available
        if _state["pca"] is not None:
            t0 = time.perf_counter()
            query_vector = apply_pca(query_vector, _state["pca"])
            timing["pca_ms"] = (time.perf_counter() - t0) * 1000
        
        # Verify dimensions match
        expected_dims = _state["vectors"].shape[1]
        actual_dims = query_vector.shape[-1]
        if actual_dims != expected_dims:
            return jsonify({
                "ok": False, 
                "error": f"Dimension mismatch: query has {actual_dims}D, index has {expected_dims}D"
            }), 500
        
        # KNN search
        t0 = time.perf_counter()
        results = knn_search(query_vector, k)
        timing["knnSearch_ms"] = (time.perf_counter() - t0) * 1000
        
        # Add image URLs to results
        for result in results:
            result["imageUrl"] = f"/image/{result['id']}"
        
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000
        
        return jsonify({
            "ok": True,
            "k": k,
            "results": results,
            "timing": timing,
            "dims": _state["dims"],
            "pcaEnabled": _state["pca"] is not None,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/image/<int:image_id>", methods=["GET"])
def get_image(image_id: int):
    """Serve an image by its ID."""
    image_path = get_image_path(image_id)
    
    if image_path is None:
        return jsonify({"ok": False, "error": f"Image {image_id} not found"}), 404
    
    return send_file(image_path, mimetype="image/png")


def main():
    parser = argparse.ArgumentParser(description="Image Search API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    parser.add_argument("--vec-file", required=True, help="Path to VEC1 binary file")
    parser.add_argument("--pca-model", default="data/pca_model.npz", help="Path to PCA model .npz file (default: data/pca_model.npz)")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA even if --pca-model exists")
    parser.add_argument("--image-dir", default=None, help="Directory containing source images")
    parser.add_argument("--embedding-dims", type=int, default=None,
                        help="Embedding dimensions before PCA (default: infer from PCA input dims if PCA is enabled, otherwise 128)")
    args = parser.parse_args()
    
    # Setup device
    device = "cpu"
    print(f"Using device: {device}")
    _state["device"] = device
    
    # Load backbone model
    print("Loading ResNet18 backbone...")
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    _state["backbone"] = backbone
    
    # Setup transform
    _state["transform"] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load PCA model unless explicitly disabled.
    if not args.no_pca and args.pca_model:
        pca_path = Path(args.pca_model)
        if pca_path.exists():
            print(f"Loading PCA model from {pca_path}...")
            _state["pca"] = load_pca_model(pca_path)
            print(f"  PCA reduces to {_state['pca']['components'].shape[0]} dimensions")
        else:
            print(f"Warning: PCA model not found at {pca_path}, continuing without PCA")
    elif args.no_pca:
        print("PCA disabled via --no-pca")

    # Resolve projector output dimensions.
    projector_dims = args.embedding_dims
    if projector_dims is None:
        if _state["pca"] is not None:
            projector_dims = int(_state["pca"]["components"].shape[1])
            print(f"Inferred embedding dims from PCA input: {projector_dims}")
        else:
            projector_dims = 128
    if projector_dims <= 0:
        raise SystemExit("--embedding-dims must be >= 1")

    # Setup projector
    torch.manual_seed(42)
    projector = torch.nn.Linear(512, projector_dims, bias=False).to(device)
    torch.nn.init.orthogonal_(projector.weight)
    projector.eval()
    _state["projector"] = projector
    
    # Load vectors
    vec_path = Path(args.vec_file)
    print(f"Loading vectors from {vec_path}...")
    ids, vectors, dims = load_vec1_file(vec_path)
    _state["ids"] = ids
    _state["vectors"] = vectors
    _state["dims"] = dims
    print(f"  Loaded {len(ids)} vectors with {dims} dimensions")

    # Validate dimensional alignment at startup to avoid runtime 500s.
    if _state["pca"] is None:
        if projector_dims != dims:
            raise SystemExit(
                f"Dimension mismatch at startup: query pipeline outputs {projector_dims}D "
                f"but index has {dims}D. Set --embedding-dims {dims}, provide --pca-model, or remove --no-pca."
            )
    else:
        pca_input_dims = int(_state["pca"]["components"].shape[1])
        pca_output_dims = int(_state["pca"]["components"].shape[0])
        if projector_dims != pca_input_dims:
            raise SystemExit(
                f"PCA model expects {pca_input_dims}D input, but --embedding-dims is {projector_dims}."
            )
        if pca_output_dims != dims:
            raise SystemExit(
                f"PCA model outputs {pca_output_dims}D but index vectors are {dims}D."
            )
    
    # Setup image directory
    if args.image_dir:
        _state["image_dir"] = Path(args.image_dir)
        if not _state["image_dir"].exists():
            print(f"Warning: Image directory not found: {args.image_dir}")
            _state["image_dir"] = None
        else:
            print(f"  Serving images from {_state['image_dir']}")
            print("  Near-duplicate hash index: lazy init on first /extract request")
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
