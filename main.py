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

DATASET_PATH = "/content/drive/MyDrive/Food_taste_dataset"

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

train_dataset = COCODataset(TRAIN_IMAGES, TRAIN_ANNOTATIONS)
test_dataset = COCODataset(TEST_IMAGES, TEST_ANNOTATIONS)
valid_dataset = COCODataset(VALID_IMAGES, VALID_ANNOTATIONS)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

num_classes = 38  # Change based on your dataset

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, val_loader, device):
    model.train()  # Set to training mode to compute loss
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()

    model.eval()  # Switch back to evaluation mode after validation
    return val_loss / len(val_loader) if val_loader else 0

EPOCHS = 2

for epoch in range(EPOCHS):
    try:
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}")
    except Exception as e:
        print(f"⚠️ Error in training at epoch {epoch+1}: {e}")

import random
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch

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
threshold = 0.5
filtered_indices = pred_scores > threshold
pred_boxes = pred_boxes[filtered_indices]
pred_labels = pred_labels[filtered_indices]
pred_scores = pred_scores[filtered_indices]

# Load class name mapping (Ensure category_mapping is defined)
# Example: category_mapping = {1: "Burger", 2: "Fries", 3: "Pizza", ...}
class_names = [category_mapping[label] for label in pred_labels]  # Convert IDs to names

# Visualize the image with bounding boxes
plt.figure(figsize=(8, 8))
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
plt.show()

import requests
import json
import torch
import random
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# API Keys
USDA_API_KEY = "MhGg5wNKenUlxwKKA7U8wvOTg7IvGmj2BGQrA8dd"  # Replace with your USDA API Key
GROQ_API_KEY = "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"  # Replace with your Groq API Key

# Age-based RDA dataset (ICMR/WHO)
RDA_DATA = {
    "child_1_3": {"Calories": 1000, "Protein": 13, "Carbohydrates": 130, "Fat": 35},
    "child_4_8": {"Calories": 1400, "Protein": 19, "Carbohydrates": 180, "Fat": 40},
    "child_9_13": {"Calories": 1800, "Protein": 34, "Carbohydrates": 220, "Fat": 55},
    "teen_male_14_18": {"Calories": 2800, "Protein": 52, "Carbohydrates": 300, "Fat": 80},
    "teen_female_14_18": {"Calories": 2200, "Protein": 46, "Carbohydrates": 275, "Fat": 70},
    "adult_male": {"Calories": 2500, "Protein": 56, "Carbohydrates": 300, "Fat": 70},
    "adult_female": {"Calories": 2000, "Protein": 46, "Carbohydrates": 275, "Fat": 60},
    "elderly_male": {"Calories": 2200, "Protein": 65, "Carbohydrates": 280, "Fat": 65},
    "elderly_female": {"Calories": 1800, "Protein": 60, "Carbohydrates": 250, "Fat": 55}
}

# Standardized USDA Nutrient IDs
NUTRIENT_IDS = {
    "Calories": 1008,
    "Protein": 1003,
    "Carbohydrates": 1005,
    "Fat": 1004,
    "Sodium": 1093
}

# Fetch food names from USDA API and dynamically query based on similarity
def query_usda_food_api(food_name):
    """Query the USDA API with the food name and fetch the closest match using fuzzy matching."""
    search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={USDA_API_KEY}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        if "foods" in data and len(data["foods"]) > 0:
            # Extract all food descriptions and fuzzy match with the input food name
            food_names = [food["description"] for food in data["foods"]]
            closest_match, score = process.extractOne(food_name, food_names)
            if score > 80:  # Only consider matches with a good similarity score
                matched_food = next(food for food in data["foods"] if food["description"] == closest_match)
                return matched_food
    return None

def get_nutrition_data(fdc_id):
    """Fetch nutrition data for a given FDC ID."""
    food_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={USDA_API_KEY}"
    response = requests.get(food_url)
    return response.json() if response.status_code == 200 else None

def extract_key_nutrients(nutrition_data):
    """Extract only required nutrients using USDA standard nutrient IDs."""
    extracted_nutrients = {key: 0 for key in NUTRIENT_IDS.keys()}
    if "foodNutrients" in nutrition_data:
        for nutrient in nutrition_data["foodNutrients"]:
            nutrient_id = nutrient.get("nutrient", {}).get("id")
            for key, usda_id in NUTRIENT_IDS.items():
                if nutrient_id == usda_id:
                    extracted_nutrients[key] = nutrient.get("amount", 0)
    return extracted_nutrients

def compare_with_rda(nutrition, age_group):
    """Compare meal's nutrition with RDA values."""
    rda = RDA_DATA.get(age_group, {})
    comparison = {key: {"value": nutrition.get(key, 0), "RDA": rda.get(key, 0),
                "percentage": round((nutrition.get(key, 0) / rda.get(key, 1)) * 100, 1)}
                for key in RDA_DATA["adult_male"].keys()}
    return comparison

def get_groq_suggestions(food_name, nutrition, age_group):
    """Use Groq LLM to suggest meal improvements based on deficiencies."""
    rda = RDA_DATA.get(age_group, {})
    prompt = f"""
    A {age_group.replace('_', ' ')} should consume daily:
    Calories: {rda.get('Calories', 'N/A')} kcal
    Protein: {rda.get('Protein', 'N/A')} g
    Carbohydrates: {rda.get('Carbohydrates', 'N/A')} g
    Fat: {rda.get('Fat', 'N/A')} g

    The meal currently contains:
    Food: {food_name}
    Calories: {nutrition.get('Calories', 0)} kcal
    Protein: {nutrition.get('Protein', 0)} g
    Carbohydrates: {nutrition.get('Carbohydrates', 0)} g
    Fat: {nutrition.get('Fat', 0)} g
    Sodium: {nutrition.get('Sodium', 0)} mg

    Suggest modifications to balance the meal and improve nutrition.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else "Error: Could not fetch meal suggestions."

def detect_and_process_food():
    """Runs Faster R-CNN to detect food and automatically fetch nutrition info."""
    random_index = random.randint(0, len(test_dataset) - 1)
    image, _ = test_dataset[random_index]
    image_np = image.permute(1, 2, 0).numpy()

    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    threshold = 0.5
    filtered_indices = pred_scores > threshold
    pred_boxes = pred_boxes[filtered_indices]
    pred_labels = pred_labels[filtered_indices]
    pred_scores = pred_scores[filtered_indices]

    detected_foods = [category_mapping[label] for label in pred_labels]

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    ax = plt.gca()

    for box, label, score in zip(pred_boxes, detected_foods, pred_scores):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{label}: {score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis("off")
    plt.show()

    return detected_foods

def main():
    age_group = input("Enter age group (e.g., child_4_8, adult_male): ")

    detected_foods = detect_and_process_food()

    total_nutrition = {key: 0 for key in NUTRIENT_IDS.keys()}

    for food_name in detected_foods:
        print(f"\nProcessing: {food_name}")

        # Query USDA API for closest match using fuzzy matching
        closest_match = query_usda_food_api(food_name)
        if not closest_match:
            print(f"No close match found for {food_name}. Please enter a common name for this food.")
            common_food_name = input("Enter common food name: ")
            closest_match = query_usda_food_api(common_food_name)
            if not closest_match:
                print(f"No match found even after entering common name for {food_name}. Skipping.")
                continue

        mapped_food_name = closest_match["description"]
        fdc_id = closest_match["fdcId"]
        
        print(f"Using food name: {mapped_food_name}")

        # Fetch nutrition data for the matched food
        nutrition_data = get_nutrition_data(fdc_id)
        if nutrition_data:
            key_nutrients = extract_key_nutrients(nutrition_data)

            # Sum up the nutrition values
            for key in total_nutrition.keys():
                total_nutrition[key] += key_nutrients.get(key, 0)

            print("\n🔹 Fetched Nutrition Data: ")
            print(json.dumps(key_nutrients, indent=4))

        else:
            print(f"Could not retrieve nutrition data for {mapped_food_name}.")

    if total_nutrition["Calories"] > 0:
        # Compare summed nutrition with RDA
        comparison = compare_with_rda(total_nutrition, age_group)
        print("\n🔹 Total Nutrition Comparison with RDA: ")
        print(json.dumps(comparison, indent=4))

        print("\n🔹 Groq's Meal Suggestions: ")
        suggestions = get_groq_suggestions("Mixed meal", total_nutrition, age_group)
        print(suggestions)

if __name__ == "__main__":
    main()
