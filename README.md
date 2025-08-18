⚠️ **Note:** This project is built for research & educational purposes. The nutritional analysis and RDA values may not be fully accurate. Do not use this app for medical or dietary decisions.

---

# 🍽️ FoodVision  
AI-powered Food Detection & Nutrition Analysis App  

<img src="https://github.com/user-attachments/assets/3ae50461-bae2-4965-8c24-9ce55f98a01b" alt="App Interface" width="800">

---

## 🚀 What is FoodVision?  
FoodVision is a **Streamlit app** that uses a **PyTorch Faster R-CNN model** to detect food items from images and provide their **nutritional breakdown**. It is an application designed to analyze images of food. Its main purpose is to identify the food items in an image and then provide nutritional information for those items. Think of it as a smart nutrition assistant: you take or upload a picture of your meal, and it tells you what foods are present and gives you calorie, protein, carb, and fat content and if it doesn't meet the RDA we can get quality suggestion to balance out the meal.

👉 Try it live on **[Streamlit Cloud](https://foodvision-lnbhnwvbdhq6dgbqgk8vst.streamlit.app/)**  

---

## ✨ Features  
* Detects **38 food categories**
* Trained on **3,330 images**
* Upload a food image and get detected items
* See calorie & nutrition facts instantly
* Compare results with **USDA nutrition database**
* Tracks health with **Recommended Daily Allowance (RDA)** comparisons
* Powered by **Streamlit**  

---
### 📂 Dataset
The model was trained on a dataset containing 38 food categories.  
You can access the dataset [here](https://universe.roboflow.com/suji-nanjundan-hvarn/food-taste/dataset/1) (if publicly available).  

---
## 🥗 Nutrition Data & Health Tracking

* USDA API → FoodVision integrates with the USDA FoodData Central API to fetch calorie and nutrient values for detected foods.
* RDA (Recommended Daily Allowance) → The app compares your food intake against standard RDA values (age- and gender-based), helping you understand whether you’re meeting, exceeding, or missing daily nutritional needs.

---
### 🍔 Supported Food Categories
FoodVision can currently detect **38 types of food items**:

- Food, AW cola, Beijing Beef, Chow Mein, Fried Rice, Hashbrown, Honey Walnut Shrimp, Kung Pao Chicken, String Bean Chicken Breast, Super Greens, The Original Orange Chicken, White Steamed Rice, Black Pepper Rice Bowl, Burger, Carrot & Eggs, Cheese Burger, Chicken Waffle, Chicken Nuggets, Chinese Cabbage, Chinese Sausage, Crispy Corn, Curry, French Fries, Fried Chicken, Fried Dumplings, Fried Eggs, Mango Chicken Pocket, Mozza Burger, Mung Bean Sprouts, Nugget, Perkedel, Rice, Sprite, Tostitos Cheese Dip Sauce, Triangle Hash Brown, Water Spinach
> ⚠️ Note: The model will only recognize the above items. Foods not listed may not be detected accurately.

---
## 📸 Application Features  
### API Configuration & Select Your Profile  
<img src="https://github.com/user-attachments/assets/f5365ad4-27eb-4a0c-bb68-531007ece09a" alt="API Configuration & Select Your Profile" width="800">

### Once API & Profile Setup is Done Upload Your Meal Image  
<img src="https://github.com/user-attachments/assets/acc04f34-68c8-4661-8968-613142297fc3" alt="Upload Meal Image" width="800">

### Detected Food Items Result  
<img src="https://github.com/user-attachments/assets/1bcd115a-83f3-4b2a-8721-e972c108d235" alt="Detected Food Items" width="800">

### Analyze the Nutrition  
<img src="https://github.com/user-attachments/assets/e9537035-810d-491a-a35c-d2b6a35be754" alt="Nutrition Analysis" width="800">

### Nutrition Suggestions for Additional Food Intake (click on suggestions)
<img src="https://github.com/user-attachments/assets/660f0d42-0392-4be8-8c0e-f74f4e788557" alt="Food Intake Suggestions" width="800">

### Nutrition Analysis Breakdown (Charts)  
<img src="https://github.com/user-attachments/assets/8d184274-b95b-4190-80a3-39b9f4b25970" alt="Nutrition Charts" width="800">

### AI-Generated Suggestions Based on Nutrition Analysis 
<img src="https://github.com/user-attachments/assets/71223293-1c53-493a-b6e4-9ff722c59cbb" alt="AI Suggestions" width="800">

### The End  
<img src="https://github.com/user-attachments/assets/fa7cdd0f-1500-4368-bcc2-b0bc2c1db1c7" alt="End" width="800">

---

## 🚀 How to Use the App

You can try the app directly without setting up locally!  

- **Live Demo:** [👉 Click here to open the app](https://foodvision-lnbhnwvbdhq6dgbqgk8vst.streamlit.app/)  
- **Model on Hugging Face:** [View/Download the model](https://huggingface.co/sushmi087/Food_detection_model/blob/main/food_detection_model.pth)  

### Steps:
1. Open the live app link.  
2. Upload a food image (single or multiple items).  
3. The app will:
   - Detect the food items in the image 🥗  
   - Fetch nutritional values from USDA API 🍎  
   - Compare them with age-specific RDA (Recommended Dietary Allowance) 📊  
   - Suggest additional foods to balance nutrition ⚖️
     
---

⚠️ **Caution / Disclaimer**  
- This application is developed for **project/demo use only**.  
- The model predictions and nutritional analysis are **approximations** and may not always be accurate.  
- Recommended Daily Allowance (RDA) values are based on general references and may **not match individual dietary needs**.  
- For personalized dietary or medical advice, please consult a **qualified nutritionist or healthcare professional**.

---



