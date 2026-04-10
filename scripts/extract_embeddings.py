#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


class ImageFolderNoLabel(Dataset):
    def __init__(self, root: Path, transform):
        self.transform = transform
        self.paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.paths.extend(sorted(root.rglob(ext)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        try:
            img_id = int(path.stem.split("_")[-1])
        except ValueError:
            img_id = idx
        return self.transform(image), img_id


def build_loader(dataset_name: str, root: Path, batch_size: int, limit: int | None):
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm)
    else:
        ds = ImageFolderNoLabel(root, tfm)

    if limit is not None:
        limit = min(limit, len(ds))
        indices = list(range(limit))
        ds = torch.utils.data.Subset(ds, indices)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def write_vectors(path: Path, vectors: np.ndarray, ids: np.ndarray = None):
    n, d = vectors.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))

        for i in range(n):
            vec_id = int(ids[i]) if ids is not None else i
            f.write(struct.pack("<Q", vec_id))
            f.write(vectors[i].astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description="Extract image embeddings and write VEC1 binary file")
    parser.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default="data/cifar10_vecs.bin")
    parser.add_argument("--dims", type=int, default=128,
                        help="Intermediate embedding dimensions from ResNet projection")
    parser.add_argument("--pca-dims", type=int, default=64,
                        help="Final dimensions after PCA reduction (e.g., 64). If not set, PCA is skipped.")
    parser.add_argument("--pca-model-path", type=str, default=None,
                        help="Path to save/load fitted PCA model (.npy file)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cpu"
    print(f"Using device: {device}")

    loader = build_loader(args.dataset, Path(args.data_root), args.batch_size, args.limit)

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    torch.manual_seed(42)
    projector = torch.nn.Linear(512, args.dims, bias=False).to(device)
    torch.nn.init.orthogonal_(projector.weight)

    out = []
    out_ids = []
    with torch.no_grad():
        for images, img_ids in loader:
            features = backbone(images.to(device))
            vectors = projector(features)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            out.append(vectors.cpu().numpy())
            out_ids.extend(img_ids.numpy() if hasattr(img_ids, "numpy") else img_ids)

    all_vecs = np.vstack(out).astype(np.float32)
    all_ids = np.array(out_ids, dtype=np.uint64)
    print(f"Extracted {all_vecs.shape[0]} vectors with dims={all_vecs.shape[1]}")

    # Apply PCA if requested
    if args.pca_dims is not None:
        if args.pca_dims >= all_vecs.shape[1]:
            print(f"Warning: pca-dims ({args.pca_dims}) >= embedding dims ({all_vecs.shape[1]}), skipping PCA")
        else:
            print(f"Applying PCA: {all_vecs.shape[1]}D -> {args.pca_dims}D")
            
            pca = PCA(n_components=args.pca_dims)
            pca_train_size = min(10000, len(all_vecs))
            print(f"Training PCA on {pca_train_size} randomly sampled samples")
            np.random.seed(42)
            indices = np.random.choice(len(all_vecs), pca_train_size, replace=False)
            pca.fit(all_vecs[indices])
            all_vecs = pca.transform(all_vecs).astype(np.float32)
            
            # Re-normalize after PCA
            norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid division by zero
            all_vecs = (all_vecs / norms).astype(np.float32)
            
            explained_var = sum(pca.explained_variance_ratio_) * 100
            print(f"PCA retained {explained_var:.2f}% of variance")
            
            # Save PCA model if path provided
            if args.pca_model_path:
                pca_path = Path(args.pca_model_path)
                pca_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    pca_path,
                    components=pca.components_,
                    mean=pca.mean_,
                    explained_variance_ratio=pca.explained_variance_ratio_
                )
                print(f"Saved PCA model to {pca_path}")

    write_vectors(Path(args.output), all_vecs, all_ids)

    print(f"Wrote {all_vecs.shape[0]} vectors with dims={all_vecs.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
