import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from submission.utils.utils import ImageData
import gc
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
BASE_PATH = ""
PATH_TO_IMAGES = os.path.join(BASE_PATH, "images")
PATH_TO_TRAIN_GT = os.path.join(BASE_PATH, "gt_for_classification_multiclass_from_filenames_0_index.csv")
VAL_FRACTION = 0.1
IMAGE_SIZE = (360, 640)  # (H, W)
MAX_EPOCHS = 30
BATCH_SIZE =16
NUM_CLASSES = 3
LEARNING_RATE = 1e-4
DEVICE = "cuda"
MODEL_SAVE_PATH = "./submission/multiclass_model.pth"  # final best model

torch.manual_seed(0)

# ===================== AUGMENTATIONS =====================
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===================== MODEL LIST =====================
MODELS = [
    'mobilenetv3_large_100',
    'ghostnet_100',
    'regnety_004',

    'efficientnet_b0',
    'efficientnet_b3',
    'densenet121',
    'resnet34',
    'resnet50',
    'regnety_008',
    
    'efficientnetv2_s',
    'convnext_tiny',
]   


# ===================== TRAINING & EVALUATION =====================


def train_and_evaluate(model_name,
                       early_stopping_patience=5,
                       min_delta=0.0001):

    print(f"\n==== Training {model_name} ====")

    torch.cuda.empty_cache()
    gc.collect()

    # Load + split dataframe
    df = pd.read_csv(PATH_TO_TRAIN_GT) 
    df["validation_set"] = 0
    df.loc[df.sample(frac=VAL_FRACTION).index, "validation_set"] = 1
    train_dataset = ImageData( img_dir=PATH_TO_IMAGES,
                               annotation_file=PATH_TO_TRAIN_GT,
                                 validation_set=False,
                                   transform=train_transform )
    val_dataset = ImageData( img_dir=PATH_TO_IMAGES,
                             annotation_file=PATH_TO_TRAIN_GT,
                               validation_set=True,
                                transform=val_transform )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Create model
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    scaler = torch.amp.GradScaler(device="cuda")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = None

    # For plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Early stopping counter
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        vloss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                vloss += loss.item()
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = vloss / len(val_loader)
        avg_val_acc = accuracy_score(all_labels, all_preds)

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Early stopping check — based on validation loss
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0  # reset
        else:
            patience_counter += 1
            print(f"⚠️ Early stopping patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print("🛑 Early stopping triggered!")
            break

        scheduler.step()

    # Cleanup
    del model, optimizer, scaler, criterion
    torch.cuda.empty_cache()
    gc.collect()

    # =============================
    # 📊 SAVE TRAINING PLOTS
    # =============================
    os.makedirs("plots", exist_ok=True)

    # Loss plot
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Loss Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/{model_name}_loss.png")
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8,5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title(f"Accuracy Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"plots/{model_name}_accuracy.png")
    plt.close()

    print(f"📁 Plots saved to: plots/{model_name}_*.png")

    return best_model_state, best_val_acc



if __name__ == "__main__":
    best_overall_acc = 0.0
    best_overall_model_state = None
    best_model_name = ""

    for model_name in MODELS:
        model_state, val_acc = train_and_evaluate(model_name)
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_model_state = model_state
            best_model_name = model_name


    torch.save(best_overall_model_state, MODEL_SAVE_PATH)
    print(f"\nBest model: {best_model_name} saved to {MODEL_SAVE_PATH} with Val Acc: {best_overall_acc:.4f}")
