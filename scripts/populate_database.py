#!/usr/bin/env python3
"""
Download sample images from the internet and generate embeddings.

Uses Lorem Picsum for diverse, high-quality sample images.
Downloads images, extracts embeddings with optional PCA, and creates the VEC1 database.
"""

import argparse
import hashlib
import random
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
from PIL import Image
import io


def download_image(url: str, timeout: int = 10) -> bytes | None:
    """Download an image from URL, returns bytes or None on failure."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ImageDownloader/1.0)"})
        with urlopen(req, timeout=timeout) as response:
            return response.read()
    except (URLError, TimeoutError, Exception) as e:
        print(f"  Failed to download {url}: {e}")
        return None


def download_picsum_images(output_dir: Path, count: int, size: int = 224) -> list[Path]:
    """Download images from Lorem Picsum."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    
    print(f"Downloading {count} images from Lorem Picsum...")
    
    # Use seed-based URLs for reproducible, unique images
    seeds = random.sample(range(1, 10000), min(count * 2, 9999))
    
    def download_one(idx: int, seed: int) -> tuple[int, Path | None]:
        url = f"https://picsum.photos/seed/{seed}/{size}/{size}"
        img_path = output_dir / f"img_{idx:04d}.jpg"
        
        if img_path.exists():
            return idx, img_path
        
        data = download_image(url)
        if data:
            try:
                # Verify it's a valid image
                img = Image.open(io.BytesIO(data))
                img.verify()
                
                # Save as RGB JPEG
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.save(img_path, "JPEG", quality=90)
                return idx, img_path
            except Exception as e:
                print(f"  Invalid image data for seed {seed}: {e}")
        return idx, None
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        seed_idx = 0
        for i in range(count):
            futures.append(executor.submit(download_one, i, seeds[seed_idx]))
            seed_idx += 1
        
        for future in as_completed(futures):
            idx, path = future.result()
            if path:
                downloaded.append(path)
                if len(downloaded) % 50 == 0:
                    print(f"  Downloaded {len(downloaded)}/{count} images...")
    
    downloaded.sort(key=lambda p: p.name)
    print(f"Successfully downloaded {len(downloaded)} images to {output_dir}")
    return downloaded


def download_cifar10_samples(output_dir: Path, count: int) -> list[Path]:
    """Download CIFAR-10 and save as individual images."""
    try:
        from torchvision import datasets
    except ImportError:
        print("torchvision not installed, cannot use CIFAR-10")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading CIFAR-10 dataset...")
    cache_dir = output_dir.parent / "cifar10_cache"
    ds_train = datasets.CIFAR10(root=str(cache_dir), train=True, download=True)
    ds_test = datasets.CIFAR10(root=str(cache_dir), train=False, download=True)
    
    # Combine train and test to get all 60,000 images
    class ConcatCIFAR:
        def __init__(self, ds1, ds2):
            self.ds1 = ds1
            self.ds2 = ds2
        def __len__(self):
            return len(self.ds1) + len(self.ds2)
        def __getitem__(self, idx):
            if idx < len(self.ds1):
                return self.ds1[idx]
            return self.ds2[idx - len(self.ds1)]

    ds = ConcatCIFAR(ds_train, ds_test)
    
    count = min(count, len(ds))
    downloaded = []
    
    print(f"Saving {count} CIFAR-10 images...")
    for i in range(count):
        img, label = ds[i]
        img_path = output_dir / f"img_{i:04d}.png"
        img.save(img_path)
        downloaded.append(img_path)
        
        if (i + 1) % 500 == 0:
            print(f"  Saved {i + 1}/{count} images...")
    
    print(f"Saved {len(downloaded)} CIFAR-10 images to {output_dir}")
    return downloaded


def extract_embeddings(image_paths: list[Path], dims: int = 128, pca_dims: int | None = None, batch_size: int = 32):
    """Extract embeddings from images."""
    import torch
    from torchvision import models, transforms
    from sklearn.decomposition import PCA
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting embeddings on {device}...")
    
    # Load model
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    
    torch.manual_seed(42)
    projector = torch.nn.Linear(512, dims, bias=False).to(device)
    torch.nn.init.orthogonal_(projector.weight)
    projector.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process in batches
    all_vectors = []
    all_ids = []
    
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_tensors = []
        batch_ids = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = transform(img)
                batch_tensors.append(tensor)
                # Extract ID from filename (img_0042.jpg -> 42)
                img_id = int(path.stem.split("_")[-1])
                batch_ids.append(img_id)
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
                continue
        
        if not batch_tensors:
            continue
        
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            features = backbone(batch)
            vectors = projector(features)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        
        all_vectors.append(vectors.cpu().numpy())
        all_ids.extend(batch_ids)
        
        if (batch_start + batch_size) % 200 == 0:
            print(f"  Processed {min(batch_start + batch_size, len(image_paths))}/{len(image_paths)} images...")
    
    vectors = np.vstack(all_vectors).astype(np.float32)
    ids = np.array(all_ids, dtype=np.uint64)
    
    print(f"Extracted {len(ids)} vectors with {vectors.shape[1]} dimensions")
    
    # Apply PCA if requested
    pca_model = None
    if pca_dims and pca_dims < vectors.shape[1]:
        print(f"Applying PCA: {vectors.shape[1]}D -> {pca_dims}D")
        pca = PCA(n_components=pca_dims)
        vectors = pca.fit_transform(vectors).astype(np.float32)
        
        # Re-normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = (vectors / norms).astype(np.float32)
        
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"  PCA retained {explained:.2f}% variance")
        
        pca_model = {
            "components": pca.components_,
            "mean": pca.mean_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    
    return ids, vectors, pca_model


def write_vec1(path: Path, ids: np.ndarray, vectors: np.ndarray):
    """Write vectors to VEC1 binary format."""
    n, d = vectors.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<I", 1))       # version
        f.write(struct.pack("<Q", n))       # count
        f.write(struct.pack("<I", d))       # dims
        f.write(struct.pack("<I", 0))       # flags
        
        for i in range(n):
            f.write(struct.pack("<Q", ids[i]))
            f.write(vectors[i].astype(np.float32).tobytes())
    
    print(f"Wrote {n} vectors ({d}D) to {path}")


def main():
    parser = argparse.ArgumentParser(description="Download images and create embedding database")
    parser.add_argument("--source", choices=["picsum", "cifar10"], default="picsum",
                        help="Image source (default: picsum)")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of images to download (default: 500)")
    parser.add_argument("--output-dir", default="data/sample_images",
                        help="Directory to save images")
    parser.add_argument("--vec-output", default="data/sample_vecs.bin",
                        help="Output VEC1 file path")
    parser.add_argument("--pca-dims", type=int, default=12,
                        help="PCA dimensions (default: 12, set to 0 to disable)")
    parser.add_argument("--pca-output", default="data/pca_model.npz",
                        help="Output PCA model file")
    parser.add_argument("--embedding-dims", type=int, default=128,
                        help="Intermediate embedding dimensions (default: 128)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only regenerate embeddings from existing images")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    vec_output = Path(args.vec_output)
    pca_output = Path(args.pca_output) if args.pca_dims > 0 else None
    
    # Download images
    if not args.skip_download:
        if args.source == "picsum":
            image_paths = download_picsum_images(output_dir, args.count)
        else:
            image_paths = download_cifar10_samples(output_dir, args.count)
        
        if not image_paths:
            print("No images downloaded, exiting.")
            sys.exit(1)
    else:
        # Find existing images
        image_paths = sorted(output_dir.glob("img_*.jpg")) + sorted(output_dir.glob("img_*.png"))
        if not image_paths:
            print(f"No images found in {output_dir}")
            sys.exit(1)
        print(f"Found {len(image_paths)} existing images in {output_dir}")
    
    # Extract embeddings
    pca_dims = args.pca_dims if args.pca_dims > 0 else None
    ids, vectors, pca_model = extract_embeddings(
        image_paths, 
        dims=args.embedding_dims,
        pca_dims=pca_dims
    )
    
    # Write outputs
    write_vec1(vec_output, ids, vectors)
    
    if pca_model and pca_output:
        np.savez(pca_output, **pca_model)
        print(f"Saved PCA model to {pca_output}")
    
    print("\n✓ Database populated successfully!")
    print(f"  Images: {output_dir}")
    print(f"  Vectors: {vec_output} ({vectors.shape[1]}D)")
    if pca_output and pca_model:
        print(f"  PCA model: {pca_output}")


if __name__ == "__main__":
    main()
