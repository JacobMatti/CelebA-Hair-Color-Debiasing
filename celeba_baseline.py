import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image


EPOCHS = 10
BATCH_SIZE = 128
LR = 3e-3
DATA_DIR = "./data/celeba"
DEVICE = "cuda"

torch.manual_seed(42)
np.random.seed(42)


class CelebAHair(Dataset):
    def __init__(self, split, transform, indices=None):
        self.transform = transform
        self.img_dir = os.path.join(DATA_DIR, "img_align_celeba", "img_align_celeba")

        attr_df = pd.read_csv(os.path.join(DATA_DIR, "list_attr_celeba.csv"))
        part_df = pd.read_csv(os.path.join(DATA_DIR, "list_eval_partition.csv"))
        df = attr_df.merge(part_df, on="image_id")
        df = df[df["partition"] == {"train": 0, "valid": 1, "test": 2}[split]].reset_index(drop=True)

        self.filenames = df["image_id"].values
        self.blond = ((df["Blond_Hair"].values + 1) // 2).astype(np.int64)
        self.male  = ((df["Male"].values + 1) // 2).astype(np.int64)
        self.indices = np.array(indices) if indices is not None else np.arange(len(self.filenames))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image = Image.open(os.path.join(self.img_dir, self.filenames[i])).convert("RGB")
        return self.transform(image), self.blond[i], self.male[i]


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader):
    model.eval()
    correct = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
    total   = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
    with torch.no_grad():
        for images, blond, male in loader:
            preds = model(images.to(DEVICE)).argmax(dim=1).cpu()
            for b, m, p in zip(blond.numpy(), male.numpy(), preds.numpy()):
                total[(b,m)]   += 1
                correct[(b,m)] += int(p == b)
    accs = {k: 100.0 * correct[k] / max(total[k], 1) for k in correct}
    avg = sum(accs.values()) / 4
    wg  = min(accs.values())
    print(f"  Non-blond women: {accs[(0,0)]:.2f}%")
    print(f"  Non-blond men:   {accs[(0,1)]:.2f}%")
    print(f"  Blond women:     {accs[(1,0)]:.2f}%")
    print(f"  Blond men:       {accs[(1,1)]:.2f}%")
    print(f"  Overall average: {avg:.2f}%")
    print(f"  Worst group:     {wg:.2f}%")
    return avg, wg


# No oversampling — use the training set as-is
train_ds = CelebAHair("train", get_transform())
val_ds   = CelebAHair("valid", get_transform())
test_ds  = CelebAHair("test",  get_transform())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

os.makedirs("./results", exist_ok=True)
best_wg = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for images, blond, _ in train_loader:
        images, blond = images.to(DEVICE), blond.to(DEVICE)
        loss = criterion(model(images), blond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} done — avg loss: {avg_loss:.4f}")
    _, wg = evaluate(model, val_loader)

    if wg > best_wg:
        best_wg = wg
        torch.save(model.state_dict(), "./results/best_model_baseline.pt")

print("\nBaseline Final Results:")
model.load_state_dict(torch.load("./results/best_model_baseline.pt", map_location=DEVICE))
avg, wg = evaluate(model, test_loader)
if avg >= 89.0:
    print("Average accuracy target met.")
else:
    print("Average accuracy target not met.")

if wg >= 85.0:
    print("Blond men accuracy target met.")
else:
    print("Blond men accuracy target not met.")
