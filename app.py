import streamlit as st
import torch
import torchvision
import numpy as np
import cv2
import requests
import json
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Food Nutrition Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .nutrition-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .detected-food {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Constants
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

NUTRIENT_IDS = {
    "Calories": 1008,
    "Protein": 1003,
    "Carbohydrates": 1005,
    "Fat": 1004,
    "Sodium": 1093
}

# Sample category mapping (you'll need to update this based on your model)
category_mapping = {0: "Food", 1: "AW cola", 2: "Beijing Beef", 3: "Chow Mein", 4: "Fried Rice", 5: "Hashbrown", 6: "Honey Walnut Shrimp", 7: "Kung Pao Chicken", 8: "String Bean Chicken Breast", 9: "Super Greens", 10: "The Original Orange Chicken", 11: "White Steamed Rice", 12: "black pepper rice bowl", 13: "burger", 14: "carrot_eggs", 15: "cheese burger", 16: "chicken waffle", 17: "chicken_nuggets", 18: "chinese_cabbage", 19: "chinese_sausage", 20: "crispy corn", 21: "curry", 22: "french fries", 23: "fried chicken", 24: "fried_chicken", 25: "fried_dumplings", 26: "fried_eggs", 27: "mango chicken pocket", 28: "mozza burger", 29: "mung_bean_sprouts", 30: "nugget", 31: "perkedel", 32: "rice", 33: "sprite", 34: "tostitos cheese dip sauce", 35: "triangle_hash_brown", 36: "water_spinach"}


@st.cache_resource
def load_model():
    try:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # You'll need to get the correct number of classes from your training
        num_classes = 38  # Update this to match your actual dataset
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Load your trained weights
        model.load_state_dict(torch.load("food_detection_model.pth", map_location='cpu'))
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def query_usda_food_api(food_name, api_key):
    """Query the USDA API with the food name and fetch the closest match"""
    try:
        search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={api_key}"
        response = requests.get(search_url)
        
        if response.status_code == 200:
            data = response.json()
            if "foods" in data and len(data["foods"]) > 0:
                food_names = [food["description"] for food in data["foods"]]
                closest_match, score = process.extractOne(food_name, food_names)
                if score > 80:
                    matched_food = next(food for food in data["foods"] if food["description"] == closest_match)
                    return matched_food
        return None
    except Exception as e:
        st.error(f"Error querying USDA API: {str(e)}")
        return None

def get_nutrition_data(fdc_id, api_key):
    """Fetch nutrition data for a given FDC ID"""
    try:
        food_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={api_key}"
        response = requests.get(food_url)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error getting nutrition data: {str(e)}")
        return None

def extract_key_nutrients(nutrition_data):
    """Extract only required nutrients using USDA standard nutrient IDs"""
    extracted_nutrients = {key: 0 for key in NUTRIENT_IDS.keys()}
    if "foodNutrients" in nutrition_data:
        for nutrient in nutrition_data["foodNutrients"]:
            nutrient_id = nutrient.get("nutrient", {}).get("id")
            for key, usda_id in NUTRIENT_IDS.items():
                if nutrient_id == usda_id:
                    extracted_nutrients[key] = nutrient.get("amount", 0)
    return extracted_nutrients

def compare_with_rda(nutrition, age_group):
    """Compare meal's nutrition with RDA values"""
    rda = RDA_DATA.get(age_group, {})
    comparison = {}
    for key in RDA_DATA["adult_male"].keys():
        value = nutrition.get(key, 0)
        rda_value = rda.get(key, 1)
        percentage = round((value / rda_value) * 100, 1) if rda_value > 0 else 0
        comparison[key] = {
            "value": value,
            "RDA": rda_value,
            "percentage": percentage
        }
    return comparison

def get_groq_suggestions(food_names, nutrition, age_group, api_key):
    """Use Groq LLM to suggest meal improvements"""
    try:
        rda = RDA_DATA.get(age_group, {})
        foods_str = ", ".join(food_names)
        
        prompt = f"""
        A {age_group.replace('_', ' ')} should consume daily:
        Calories: {rda.get('Calories', 'N/A')} kcal
        Protein: {rda.get('Protein', 'N/A')} g
        Carbohydrates: {rda.get('Carbohydrates', 'N/A')} g
        Fat: {rda.get('Fat', 'N/A')} g

        The meal currently contains: {foods_str}
        Calories: {nutrition.get('Calories', 0)} kcal
        Protein: {nutrition.get('Protein', 0)} g
        Carbohydrates: {nutrition.get('Carbohydrates', 0)} g
        Fat: {nutrition.get('Fat', 0)} g
        Sodium: {nutrition.get('Sodium', 0)} mg

        Provide specific, actionable suggestions to improve this meal's nutritional balance.
        """
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "Error: Could not fetch meal suggestions."
    except Exception as e:
        st.error(f"Error getting suggestions: {str(e)}")
        return "Error generating suggestions."

def detect_food_in_image(image, model):
    """Detect food items in the uploaded image"""
    try:
        # Convert PIL image to tensor
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            st.error("Please upload a color image")
            return [], image_np
        
        # Run inference
        with torch.no_grad():
            prediction = model([image_tensor])
        
        # Extract predictions
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_labels = prediction[0]['labels'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        threshold = 0.5
        filtered_indices = pred_scores > threshold
        pred_boxes = pred_boxes[filtered_indices]
        pred_labels = pred_labels[filtered_indices]
        pred_scores = pred_scores[filtered_indices]
        
        # Convert labels to food names
        detected_foods = [category_mapping.get(label, f"Unknown_{label}") for label in pred_labels]
        
        return detected_foods, image_np
        
    except Exception as e:
        st.error(f"Error in food detection: {str(e)}")
        return [], np.array(image)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Food Nutrition Analyzer</h1>
        <p>Upload an image of your meal to get nutrition information and personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API keys and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        usda_api_key = st.text_input(
            "USDA API Key",
            type="password",
            help="Get your free API key from https://fdc.nal.usda.gov/api-guide.html"
        )
        
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your API key from https://groq.com/"
        )
        
        st.header("👤 User Information")
        age_group = st.selectbox(
            "Select Age Group",
            options=list(RDA_DATA.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            index=5  # Default to adult_male
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Food Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of your meal"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Meal", type="primary"):
                if not usda_api_key:
                    st.error("Please enter your USDA API key in the sidebar")
                    return
                
                # Load model
                model = load_model()
                if model is None:
                    st.error("Could not load the food detection model")
                    return
                
                with st.spinner("Detecting food items..."):
                    detected_foods, image_np = detect_food_in_image(image, model)
                
                if detected_foods:
                    st.session_state['detected_foods'] = detected_foods
                    st.session_state['image_np'] = image_np
                    st.session_state['age_group'] = age_group
                    st.session_state['usda_api_key'] = usda_api_key
                    st.session_state['groq_api_key'] = groq_api_key
                    st.rerun()
                else:
                    st.warning("No food items detected in the image. Please try another image.")
    
    with col2:
        st.header("🍎 Detected Foods")
        
        if 'detected_foods' in st.session_state:
            detected_foods = st.session_state['detected_foods']
            
            st.success(f"Detected {len(detected_foods)} food items:")
            for food in detected_foods:
                st.markdown(f'<div class="detected-food">🍽️ {food}</div>', unsafe_allow_html=True)
            
            # Get nutrition information
            if st.button("📊 Get Nutrition Analysis", type="secondary"):
                with st.spinner("Fetching nutrition data..."):
                    total_nutrition = {key: 0 for key in NUTRIENT_IDS.keys()}
                    successful_foods = []
                    
                    for food_name in detected_foods:
                        closest_match = query_usda_food_api(food_name, st.session_state['usda_api_key'])
                        if closest_match:
                            nutrition_data = get_nutrition_data(closest_match["fdcId"], st.session_state['usda_api_key'])
                            if nutrition_data:
                                key_nutrients = extract_key_nutrients(nutrition_data)
                                for key in total_nutrition.keys():
                                    total_nutrition[key] += key_nutrients.get(key, 0)
                                successful_foods.append(food_name)
                    
                    if successful_foods:
                        st.session_state['total_nutrition'] = total_nutrition
                        st.session_state['successful_foods'] = successful_foods
                        st.rerun()
    
    # Nutrition Results Section
    if 'total_nutrition' in st.session_state:
        st.header("📈 Nutrition Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Nutritional Content")
            nutrition = st.session_state['total_nutrition']
            
            # Create nutrition cards
            for nutrient, value in nutrition.items():
                unit = "kcal" if nutrient == "Calories" else "g" if nutrient != "Sodium" else "mg"
                st.markdown(f"""
                <div class="nutrition-card">
                    <h4>{nutrient}</h4>
                    <h2>{value:.1f} {unit}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("RDA Comparison")
            comparison = compare_with_rda(nutrition, st.session_state['age_group'])
            
            for nutrient, data in comparison.items():
                percentage = data['percentage']
                color = "green" if percentage >= 80 else "orange" if percentage >= 50 else "red"
                
                st.metric(
                    label=nutrient,
                    value=f"{data['value']:.1f}",
                    delta=f"{percentage:.1f}% of RDA"
                )
                st.progress(min(percentage / 100, 1.0))
        
        # AI Suggestions
        if st.session_state.get('groq_api_key'):
            st.header("🤖 AI Recommendations")
            
            if st.button("Get Personalized Suggestions"):
                with st.spinner("Generating recommendations..."):
                    suggestions = get_groq_suggestions(
                        st.session_state['successful_foods'],
                        nutrition,
                        st.session_state['age_group'],
                        st.session_state['groq_api_key']
                    )
                    
                    st.markdown("### Meal Improvement Suggestions:")
                    st.write(suggestions)
        else:
            st.info("💡 Add your Groq API key in the sidebar to get AI-powered meal suggestions!")
    
    # How it works section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Upload a photo** of your meal
        2. **Our AI detects** food items in the image using computer vision
        3. **We analyze** the nutritional content using USDA food database
        4. **Get personalized** recommendations based on your age group
        
        **Note:** Make sure to get your free API keys:
        - [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide.html)
        - [Groq API](https://groq.com/) (optional, for AI suggestions)
        """)

if __name__ == "__main__":
    main()