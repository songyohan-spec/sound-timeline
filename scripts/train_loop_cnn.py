from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_TARGETS = ["source_family", "reverb", "distortion", "filter_presence", "filter_motion_type", "stereo", "motion_presence"]


def read_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_stereo(path: Path, target_sr: int) -> np.ndarray:
    import librosa

    audio, sr = sf.read(path, always_2d=True)
    stereo = audio.astype(np.float32)
    if sr != target_sr:
        channels = []
        for ch in range(stereo.shape[1]):
            channels.append(librosa.resample(stereo[:, ch], orig_sr=sr, target_sr=target_sr))
        min_len = min(len(ch) for ch in channels)
        stereo = np.stack([ch[:min_len] for ch in channels], axis=1)
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    return stereo[:, :2]


def make_logmel(path: Path, sample_rate: int, n_mels: int, duration: float) -> np.ndarray:
    import librosa

    stereo = load_stereo(path, sample_rate)
    target_len = int(sample_rate * duration)
    if len(stereo) >= target_len:
        stereo = stereo[:target_len]
    else:
        stereo = np.pad(stereo, ((0, target_len - len(stereo)), (0, 0)))

    left = stereo[:, 0]
    right = stereo[:, 1]
    mono = 0.5 * (left + right)
    side = 0.5 * (left - right)

    channels = []
    for y in [mono, side]:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            fmin=30,
            fmax=sample_rate // 2,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db + 80.0) / 80.0
        channels.append(mel_db.astype(np.float32))
    return np.stack(channels, axis=0)


class LoopDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        indices: np.ndarray,
        dataset_root: Path,
        targets: list[str],
        encoders: dict[str, LabelEncoder],
        sample_rate: int,
        n_mels: int,
        duration: float,
        cache_dir: Path | None,
    ) -> None:
        self.rows = rows
        self.indices = list(indices)
        self.dataset_root = dataset_root
        self.targets = targets
        self.encoders = encoders
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.indices)

    def _feature_path(self, row_index: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{row_index:06d}.npy"

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row_index = self.indices[item]
        row = self.rows[row_index]
        feature_path = self._feature_path(row_index)
        if feature_path is not None and feature_path.exists():
            mel = np.load(feature_path)
        else:
            mel = make_logmel(self.dataset_root / row["file"], self.sample_rate, self.n_mels, self.duration)
            if feature_path is not None:
                np.save(feature_path, mel)

        x = torch.from_numpy(mel)
        y = {
            target: torch.tensor(int(self.encoders[target].transform([row[target]])[0]), dtype=torch.long)
            for target in self.targets
        }
        return x, y


class SmallAudioCNN(nn.Module):
    def __init__(self, targets: list[str], num_classes: dict[str, int], in_channels: int = 2) -> None:
        super().__init__()
        self.targets = targets
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.25)
        self.heads = nn.ModuleDict({target: nn.Linear(96, num_classes[target]) for target in targets})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x).flatten(1)
        z = self.dropout(z)
        return {target: head(z) for target, head in self.heads.items()}


def collate_batch(batch):
    xs = torch.stack([item[0] for item in batch])
    ys: dict[str, list[torch.Tensor]] = {}
    for _, target_dict in batch:
        for key, value in target_dict.items():
            ys.setdefault(key, []).append(value)
    return xs, {key: torch.stack(values) for key, values in ys.items()}


def evaluate(model: nn.Module, loader: DataLoader, targets: list[str], encoders: dict[str, LabelEncoder], device: torch.device) -> None:
    model.eval()
    truth = {target: [] for target in targets}
    pred = {target: [] for target in targets}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            for target in targets:
                truth[target].extend(y[target].numpy().tolist())
                pred[target].extend(outputs[target].argmax(dim=1).cpu().numpy().tolist())

    for target in targets:
        print(f"\n[{target}]")
        print(classification_report(truth[target], pred[target], target_names=encoders[target].classes_, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/loop_nsynth_stage1"))
    parser.add_argument("--out", type=Path, default=Path("models/loop_cnn.pt"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--n-mels", type=int, default=96)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/logmel_loop_nsynth_stage1"))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    rows = read_rows(args.dataset / "metadata.jsonl")
    if len(rows) < 50:
        raise SystemExit("Need at least 50 samples for CNN training.")
    for target in targets:
        if target not in rows[0]:
            raise SystemExit(f"Target '{target}' is missing from metadata.")

    encoders = {}
    encoded_for_split = None
    for target in targets:
        encoder = LabelEncoder()
        values = [row[target] for row in rows]
        encoder.fit(values)
        encoders[target] = encoder
        if target == "source_family":
            encoded_for_split = encoder.transform(values)

    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=args.seed, stratify=encoded_for_split)

    train_ds = LoopDataset(rows, train_idx, args.dataset, targets, encoders, args.sample_rate, args.n_mels, args.duration, args.cache_dir)
    test_ds = LoopDataset(rows, test_idx, args.dataset, targets, encoders, args.sample_rate, args.n_mels, args.duration, args.cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    num_classes = {target: len(encoders[target].classes_) for target in targets}
    model = SmallAudioCNN(targets, num_classes, in_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = {}
    if args.class_weight:
        for target in targets:
            labels = encoders[target].transform([row[target] for row in rows])
            counts = np.bincount(labels, minlength=num_classes[target]).astype(np.float32)
            weights = counts.sum() / np.maximum(counts, 1.0)
            weights = weights / weights.mean()
            class_weights[target] = torch.tensor(weights, dtype=torch.float32, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = {target: value.to(device) for target, value in y.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = sum(F.cross_entropy(outputs[target], y[target], weight=class_weights.get(target)) for target in targets) / len(targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * x.shape[0]
            seen += x.shape[0]
        print(f"epoch {epoch:02d} loss {total_loss / max(1, seen):.4f}")

    evaluate(model, test_loader, targets, encoders, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "targets": targets,
            "classes": {target: encoders[target].classes_.tolist() for target in targets},
            "sample_rate": args.sample_rate,
            "n_mels": args.n_mels,
            "duration": args.duration,
        },
        args.out,
    )
    joblib.dump({"targets": targets, "encoders": encoders}, args.out.with_suffix(".encoders.joblib"))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
