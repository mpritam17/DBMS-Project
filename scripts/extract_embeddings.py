#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
        return self.transform(image), idx


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


def write_vectors(path: Path, vectors: np.ndarray):
    n, d = vectors.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))

        for i in range(n):
            f.write(struct.pack("<Q", i))
            f.write(vectors[i].astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description="Extract image embeddings and write VEC1 binary file")
    parser.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default="data/cifar10_vecs.bin")
    parser.add_argument("--dims", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loader = build_loader(args.dataset, Path(args.data_root), args.batch_size, args.limit)

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    projector = torch.nn.Linear(512, args.dims, bias=False).to(device)
    torch.nn.init.orthogonal_(projector.weight)

    out = []
    with torch.no_grad():
        for images, _ in loader:
            features = backbone(images.to(device))
            vectors = projector(features)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            out.append(vectors.cpu().numpy())

    all_vecs = np.vstack(out).astype(np.float32)
    write_vectors(Path(args.output), all_vecs)

    print(f"Wrote {all_vecs.shape[0]} vectors with dims={all_vecs.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
