import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
import torch.optim as optim
import random

# Dataset paths
DATASET_PATH = r"C:\Users\sushm\OneDrive\Desktop\Food_taste_dataset"

TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "train", "_annotations.coco.json")
TEST_ANNOTATIONS = os.path.join(DATASET_PATH, "test", "_annotations.coco.json")
VALID_ANNOTATIONS = os.path.join(DATASET_PATH, "valid", "_annotations.coco.json")

TRAIN_IMAGES = os.path.join(DATASET_PATH, "train")
TEST_IMAGES = os.path.join(DATASET_PATH, "test")
VALID_IMAGES = os.path.join(DATASET_PATH, "valid")

class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_path, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize to [0,1] range
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to Tensor

        # Prepare target
        boxes = []
        labels = []
        for ann in annotations:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

def create_category_mapping(coco_annotation_path):
    """Create category mapping from COCO annotations"""
    coco = COCO(coco_annotation_path)
    categories = coco.loadCats(coco.getCatIds())
    category_mapping = {cat["id"]: cat["name"] for cat in categories}
    return category_mapping

def train_one_epoch(model, optimizer, train_loader, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip batches with empty targets
            if any(len(t['boxes']) == 0 for t in targets):
                continue

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 20 == 0:  # Reduced logging frequency for VS Code
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    return running_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate the model"""
    model.train()  # Set to training mode to compute loss
    val_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Skip batches with empty targets
                if any(len(t['boxes']) == 0 for t in targets):
                    continue

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
                valid_batches += 1
                
            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue

    model.eval()  # Switch back to evaluation mode after validation
    return val_loss / valid_batches if valid_batches > 0 else 0

def test_model_visualization(model, test_dataset, category_mapping, device):
    """Test the model with a random image and visualize results"""
    # Select a random image from the test dataset
    random_index = random.randint(0, len(test_dataset) - 1)
    image, _ = test_dataset[random_index]  # Get image tensor
    image_np = image.permute(1, 2, 0).numpy()  # Convert tensor to NumPy for visualization

    # Move model to evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])  # Run inference

    # Extract detected boxes, labels, and confidence scores
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    # Confidence threshold for displaying detections
    threshold = 0.2
    filtered_indices = pred_scores > threshold
    pred_boxes = pred_boxes[filtered_indices]
    pred_labels = pred_labels[filtered_indices]
    pred_scores = pred_scores[filtered_indices]

    print(f"Detected {len(pred_labels)} objects above threshold {threshold}")

    # Get class names
    class_names = [category_mapping.get(label, f"Unknown_{label}") for label in pred_labels]

    # Visualize the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    ax = plt.gca()

    # Draw bounding boxes with actual class names
    for box, label, score in zip(pred_boxes, class_names, pred_scores):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{label}: {score:.2f}",
                color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis("off")
    plt.title(f"Detected Foods: {', '.join(class_names)}")
    plt.show()
    
    return class_names

def main():
    print("="*50)
    print("FOOD DETECTION MODEL TRAINING")
    print("="*50)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = COCODataset(TRAIN_IMAGES, TRAIN_ANNOTATIONS)
    test_dataset = COCODataset(TEST_IMAGES, TEST_ANNOTATIONS)
    valid_dataset = COCODataset(VALID_IMAGES, VALID_ANNOTATIONS)

    # Create category mapping from COCO annotations
    category_mapping = create_category_mapping(TRAIN_ANNOTATIONS)
    print(f"Category mapping created with {len(category_mapping)} classes:")
    for k, v in list(category_mapping.items())[:10]:  # Show first 10
        print(f"  {k}: {v}")

    # Create data loaders with smaller batch size for VS Code
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=6, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Create model
    num_classes = len(category_mapping) + 1  # +1 for background class
    print(f"Number of classes (including background): {num_classes}")

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training parameters - optimized for VS Code environment
    EPOCHS = 3

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        try:
            val_loss = validate(model, valid_loader, device)
            print(f"Validation Loss: {val_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Error in validation at epoch {epoch+1}: {e}")

    # Save the trained model
    print("\nSaving model...")
    torch.save(model.state_dict(), "food_detection_model.pth")
    print("Model saved as 'food_detection_model.pth'")

    # Save category mapping for use in app.py
    import json
    with open("category_mapping.json", "w") as f:
        json.dump(category_mapping, f)
    print("Category mapping saved as 'category_mapping.json'")

    # Test the model with visualization
    print("\nTesting model with a random image...")
    detected_foods = test_model_visualization(model, test_dataset, category_mapping, device)
    print(f"Test completed. Detected foods: {detected_foods}")

if __name__ == "__main__":
    main()