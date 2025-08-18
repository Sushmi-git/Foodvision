import os
import torch
import torchvision
import numpy as np
import requests
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Food Nutrition Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, sleek CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding: 0 !important;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Hide default header */
    .main > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Light beige gradient background */
    .stApp {
        background: linear-gradient(135deg, #f7f3e9 0%, #e8dcc6 50%, #d4c5a9 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero section styling with beige theme */
    .hero-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(139, 119, 101, 0.2);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #8b7765;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(139, 119, 101, 0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a0896b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .hero-description {
        font-size: 1.1rem;
        color: #8b7765;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    
    /* Feature cards with beige theme */
    .feature-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(139, 119, 101, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #8b7765;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        color: #a0896b;
        line-height: 1.6;
    }
    
    /* Navigation pills with beige theme */
    .nav-pill {
        background: rgba(255, 255, 255, 0.6);
        color: #8b7765;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 500;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-pill:hover {
        background: rgba(255, 255, 255, 0.8);
        border-color: rgba(139, 119, 101, 0.3);
        transform: translateY(-2px);
    }
    
    .nav-pill.active {
        background: #8b7765;
        color: white;
        font-weight: 600;
    }
    
    /* Page content containers */
    .page-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #8b7765;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .page-subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b7765;
        box-shadow: 0 0 0 3px rgba(139, 119, 101, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stMultiSelect > div > div {
        border-radius: 10px;
    }
    
    /* Button styling with beige theme */
    .stButton > button {
        background: linear-gradient(135deg, #8b7765 0%, #a0896b 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 119, 101, 0.3);
    }
    
    /* File uploader styling with beige theme */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #8b7765;
        background: rgba(139, 119, 101, 0.05);
    }
    
    /* Detection results with beige theme */
    .detected-food {
        background: linear-gradient(135deg, #8b7765, #a0896b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(139, 119, 101, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    /* Suggestions container with beige theme */
    .suggestions-container {
        background: linear-gradient(135deg, #d4c5a9 0%, #c4b59a 100%);
        color: #8b7765;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(139, 119, 101, 0.3);
    }
    
    .suggestions-container h3 {
        color: #8b7765;
        margin-bottom: 1rem;
    }
    
    /* Success/Info messages styling with beige theme */
    .stSuccess {
        background: linear-gradient(135deg, #8b7765, #a0896b);
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #8b7765, #a0896b);
        border-radius: 10px;
    }
    
    /* Progress bar with beige theme */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #8b7765 0%, #a0896b 100%);
    }
</style>
""", unsafe_allow_html=True)

# Constants (keeping the same as original)
RDA_DATA = {
    "adult_male": {"Calories": 2500, "Protein": 56, "Carbohydrates": 300, "Fat": 70},
    "adult_female": {"Calories": 2000, "Protein": 46, "Carbohydrates": 275, "Fat": 60},
    "teen_male": {"Calories": 2800, "Protein": 52, "Carbohydrates": 300, "Fat": 80},
    "teen_female": {"Calories": 2200, "Protein": 46, "Carbohydrates": 275, "Fat": 70},
    "child_4_8": {"Calories": 1400, "Protein": 19, "Carbohydrates": 180, "Fat": 40},
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

category_mapping = {
    0: "Food", 1: "AW cola", 2: "Beijing Beef", 3: "Chow Mein", 4: "Fried Rice", 
    5: "Hashbrown", 6: "Honey Walnut Shrimp", 7: "Kung Pao Chicken", 
    8: "String Bean Chicken Breast", 9: "Super Greens", 10: "The Original Orange Chicken", 
    11: "White Steamed Rice", 12: "black pepper rice bowl", 13: "burger", 14: "carrot_eggs", 
    15: "cheese burger", 16: "chicken waffle", 17: "chicken_nuggets", 18: "chinese_cabbage", 
    19: "chinese_sausage", 20: "crispy corn", 21: "curry", 22: "french fries", 
    23: "fried chicken", 24: "fried_chicken", 25: "fried_dumplings", 26: "fried_eggs", 
    27: "mango chicken pocket", 28: "mozza burger", 29: "mung_bean_sprouts", 30: "nugget", 
    31: "perkedel", 32: "rice", 33: "sprite", 34: "tostitos cheese dip sauce", 
    35: "triangle_hash_brown", 36: "water_spinach"
}

# Helper functions (keeping the same logic)
import os
import torch
import streamlit as st
from huggingface_hub import hf_hub_download
from torchvision.models.detection import fasterrcnn_resnet50_fpn

REPO_ID = "sushmi087/Food_detection_model"
FILENAME = "food_detection_model.pth"
NUM_CLASSES = 38

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Download the model using Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_files_only=True 
        )

        # Build the detection model architecture
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, NUM_CLASSES
        )

        # Load and evaluate
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Example usage in your app flow:
model = load_model()
if model is None:
    st.stop()

def create_intelligent_prompt(foods, nutrition, age_group, health_goals, dietary_restrictions):
    rda = RDA_DATA.get(age_group, RDA_DATA['adult_male'])
    
    nutrient_status = {}
    for nutrient in ['Calories', 'Protein', 'Carbohydrates', 'Fat']:
        current = nutrition.get(nutrient, 0)
        target = rda.get(nutrient, 1)
        percentage = (current / target) * 100 if target > 0 else 0
        
        if percentage < 60:
            status = "low"
        elif percentage > 120:
            status = "high" 
        else:
            status = "adequate"
            
        nutrient_status[nutrient] = {
            'value': current,
            'target': target,
            'percentage': percentage,
            'status': status
        }
    
    prompt = f"""As a nutrition expert, analyze this meal for a {age_group.replace('_', ' ')}:

FOODS CONSUMED: {', '.join(foods)}

CURRENT NUTRITION:
"""
    
    for nutrient, data in nutrient_status.items():
        unit = "kcal" if nutrient == "Calories" else "g"
        prompt += f"• {nutrient}: {data['value']:.1f}{unit} ({data['percentage']:.0f}% of {data['target']}{unit} target) - {data['status'].upper()}\n"
    
    if health_goals:
        prompt += f"\nHEALTH GOALS: {', '.join(health_goals)}"
    
    if dietary_restrictions:
        prompt += f"\nDIETARY RESTRICTIONS: {', '.join(dietary_restrictions)}"
    
    prompt += f"""

Please provide:
1. MEAL RATING: Score this meal 1-10 for nutritional quality and explain why
2. KEY INSIGHTS: What's good and what needs improvement
3. SPECIFIC RECOMMENDATIONS: 3 practical suggestions to enhance this meal
4. QUICK FIXES: Simple swaps or additions for better nutrition
5. PORTION ADVICE: Any adjustments needed for optimal nutrition

Keep advice practical, specific, and actionable for real-world implementation.
"""
    
    return prompt

def query_usda_api(food_name, api_key):
    try:
        clean_name = food_name.replace('_', ' ').strip()
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={clean_name}&api_key={api_key}&pageSize=3"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "foods" in data and data["foods"]:
                food_names = [food["description"] for food in data["foods"]]
                best_match, score = process.extractOne(clean_name, food_names)
                
                if score > 70:
                    matched_food = next(food for food in data["foods"] if food["description"] == best_match)
                    return matched_food
        return None
    except Exception as e:
        st.warning(f"API error for {food_name}: {str(e)}")
        return None

def get_nutrition_data(fdc_id, api_key):
    try:
        url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={api_key}"
        response = requests.get(url, timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.warning(f"Nutrition data error: {str(e)}")
        return None

def extract_nutrients(nutrition_data):
    nutrients = {key: 0 for key in NUTRIENT_IDS.keys()}
    
    if "foodNutrients" in nutrition_data:
        for nutrient in nutrition_data["foodNutrients"]:
            nutrient_id = nutrient.get("nutrient", {}).get("id")
            for key, usda_id in NUTRIENT_IDS.items():
                if nutrient_id == usda_id:
                    nutrients[key] = nutrient.get("amount", 0)
    
    return nutrients

def get_ai_suggestions(foods, nutrition, age_group, health_goals, restrictions, api_key):
    try:
        prompt = create_intelligent_prompt(foods, nutrition, age_group, health_goals, restrictions)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a professional nutritionist providing practical, evidence-based dietary advice."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "Unable to generate suggestions at this time."
            
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"

def detect_foods(image, model):
    try:
        image_np = np.array(image)
        if len(image_np.shape) != 3:
            st.error("Please upload a color image")
            return []
        
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        with torch.no_grad():
            prediction = model([image_tensor])
        
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        threshold = 0.5
        valid_indices = scores > threshold
        valid_labels = labels[valid_indices]
        
        detected_foods = [category_mapping.get(label, f"Unknown_{label}") for label in valid_labels]
        return detected_foods
        
    except Exception as e:
        st.error(f"Food detection error: {str(e)}")
        return []

def create_nutrition_chart(nutrition_data, rda_data):
    nutrients = ['Calories', 'Protein', 'Carbohydrates', 'Fat']
    current_values = [nutrition_data.get(n, 0) for n in nutrients]
    target_values = [rda_data.get(n, 0) for n in nutrients]
    
    fig = go.Figure(data=[
        go.Bar(name='Current', x=nutrients, y=current_values, marker_color='#667eea'),
        go.Bar(name='Target', x=nutrients, y=target_values, marker_color='#764ba2')
    ])
    
    fig.update_layout(
        title='Nutrition vs Daily Targets',
        barmode='group',
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# Page navigation
def show_home_page():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🍽️ NutriVision AI</div>
        <div class="hero-subtitle">Your Personal Food & Nutrition Intelligence</div>
        <div class="hero-description">
            Transform your meals into actionable nutrition insights using cutting-edge AI and computer vision. 
            Upload a photo, get instant food detection, detailed nutrition analysis, and personalized recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <span class="feature-icon">🔑</span>
            <div class="feature-title">API Configuration</div>
            <div class="feature-description">
                Set up your USDA and Groq API keys securely. Configure your personal profile including age group, 
                health goals, and dietary restrictions for personalized analysis.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🍕 Foods Detected", "35+", "Categories")
    with col2:
        st.metric("🔬 Nutrients Analyzed", "5", "Key Nutrients")
    with col3:
        st.metric("👥 Age Groups", "7", "Supported")
    with col4:
        st.metric("⚡ Analysis Time", "<30s", "Average")

def show_api_page():
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">🔑 API Configuration</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Configure your API keys and personal profile for optimal nutrition analysis</p>', unsafe_allow_html=True)
    
    # API Keys Section
    st.markdown("### 🌐 API Keys")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### USDA Food Data API")
        usda_key = st.text_input(
            "USDA API Key", 
            type="password",
            help="Get your free key at https://fdc.nal.usda.gov/api-guide.html",
            placeholder="Enter your USDA API key..."
        )
        
        if usda_key:
            st.success("✅ USDA API key configured")
            st.session_state['usda_key'] = usda_key
        else:
            st.info("🔗 [Get USDA API Key](https://fdc.nal.usda.gov/api-guide.html)")
    
    with col2:
        st.markdown("#### Groq AI API (Optional)")
        groq_key = st.text_input(
            "Groq AI Key", 
            type="password",
            help="Get your key at https://groq.com/ for AI-powered recommendations",
            placeholder="Enter your Groq API key..."
        )
        
        if groq_key:
            st.success("✅ Groq AI API key configured")
            st.session_state['groq_key'] = groq_key
        else:
            st.info("🔗 [Get Groq API Key](https://groq.com/)")
    
    st.markdown("---")
    
    # User Profile Section
    st.markdown("### 👤 Personal Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_group = st.selectbox(
            "Age Group", 
            options=list(RDA_DATA.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select your age group for accurate daily nutrition targets"
        )
        st.session_state['age_group'] = age_group
        
        # Show RDA for selected group
        rda = RDA_DATA[age_group]
        st.markdown("**Daily Nutrition Targets:**")
        for nutrient, value in rda.items():
            unit = "kcal" if nutrient == "Calories" else "g"
            st.markdown(f"• {nutrient}: {value} {unit}")
    
    with col2:
        health_goals = st.multiselect(
            "Health Goals",
            options=["Weight Loss", "Muscle Gain", "Heart Health", "General Wellness", "Athletic Performance"],
            default=["General Wellness"],
            help="Select your primary health objectives"
        )
        st.session_state['health_goals'] = health_goals
        
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions",
            options=["Vegetarian", "Vegan", "Gluten Free", "Dairy Free", "Low Sodium", "Diabetic"],
            default=[],
            help="Select any dietary restrictions or preferences"
        )
        st.session_state['dietary_restrictions'] = dietary_restrictions
    
    st.markdown("---")
    
    # Configuration Summary
    if usda_key and age_group:
        st.success("🎉 Configuration Complete! You can now proceed to food detection.")
        
        if st.button("🚀 Start Food Detection", type="primary"):
            st.session_state['current_page'] = 'detection'
            st.rerun()
    else:
        st.warning("⚠️ Please configure at least the USDA API key to continue.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_detection_page():
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">📸 Food Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload a photo of your meal for AI-powered food recognition and nutrition analysis</p>', unsafe_allow_html=True)
    
    if 'usda_key' not in st.session_state:
        st.error("❌ Please configure your API keys first!")
        if st.button("⬅️ Go to API Configuration"):
            st.session_state['current_page'] = 'api'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Meal Photo")
        uploaded_file = st.file_uploader(
            "Choose an image of your meal", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo with good lighting for best results"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Meal", use_column_width=True)
            
            if st.button("🔍 Detect Foods", type="primary"):
                if 'usda_key' not in st.session_state:
                    st.error("Please configure USDA API key first")
                    st.stop()
                
                # Food Detection
                with st.spinner("🤖 Loading AI model..."):
                    model = load_model()
                    if not model:
                        st.error("Could not load detection model")
                        st.stop()
                
                with st.spinner("🔍 Detecting foods in your image..."):
                    detected_foods = detect_foods(image, model)
                
                if detected_foods:
                    st.session_state.update({
                        'foods': detected_foods,
                        'image': image
                    })
                    st.rerun()
                else:
                    st.warning("🤔 No foods detected. Try a clearer image with better lighting.")
    
    with col2:
        st.markdown("### 🎯 Detection Results")
        
        if 'foods' in st.session_state:
            foods = st.session_state['foods']
            
            st.success(f"🎉 Successfully detected {len(foods)} food items!")
            
            for i, food in enumerate(foods, 1):
                st.markdown(f'<div class="detected-food">#{i} {food.replace("_", " ").title()}</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("---")
            
            if st.button("📊 Analyze Nutrition", type="primary"):
                with st.spinner("🔍 Fetching nutrition data from USDA database..."):
                    progress_bar = st.progress(0)
                    
                    total_nutrition = {key: 0 for key in NUTRIENT_IDS.keys()}
                    successful_foods = []
                    
                    for i, food in enumerate(foods):
                        progress_bar.progress((i + 1) / len(foods))
                        
                        matched_food = query_usda_api(food, st.session_state['usda_key'])
                        if matched_food:
                            nutrition_data = get_nutrition_data(matched_food["fdcId"], st.session_state['usda_key'])
                            if nutrition_data:
                                nutrients = extract_nutrients(nutrition_data)
                                for key in total_nutrition:
                                    total_nutrition[key] += nutrients.get(key, 0)
                                successful_foods.append(food)
                    
                    progress_bar.empty()
                    
                    if successful_foods:
                        st.session_state.update({
                            'nutrition': total_nutrition,
                            'successful_foods': successful_foods
                        })
                        st.success(f"✅ Nutrition data found for {len(successful_foods)}/{len(foods)} foods!")
                        
                        # Show nutrition preview
                        nutrition = total_nutrition
                        rda = RDA_DATA[st.session_state.get('age_group', 'adult_male')]
                        
                        st.markdown("#### 📊 Nutrition Preview")
                        for nutrient in ['Calories', 'Protein', 'Carbohydrates', 'Fat']:
                            value = nutrition.get(nutrient, 0)
                            target = rda.get(nutrient, 1)
                            percentage = (value / target) * 100 if target > 0 else 0
                            unit = "kcal" if nutrient == "Calories" else "g"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <strong>{nutrient}</strong>: {value:.1f} {unit} 
                                <span style="color: #8b7765;">({percentage:.0f}% of daily target)</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if st.button("🤖 Get AI Recommendations", type="primary"):
                            st.session_state['current_page'] = 'suggestions'
                            st.rerun()
                    else:
                        st.error("❌ Could not retrieve nutrition data for any foods. Please try again.")
        else:
            st.info("📸 Upload and detect foods first to see nutrition analysis options.")
            
            # Tips for better detection
            st.markdown("#### 💡 Tips for Better Detection")
            st.markdown("""
            - Use good lighting and avoid shadows
            - Make sure food items are clearly visible
            - Avoid cluttered backgrounds
            - Take photos from a slight angle above the food
            - Ensure foods are not overlapping too much
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_suggestions_page():
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">🤖 AI Nutrition Coach</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Get personalized nutrition insights and recommendations from our AI nutritionist</p>', unsafe_allow_html=True)
    
    if 'nutrition' not in st.session_state or 'successful_foods' not in st.session_state:
        st.error("❌ Please complete food detection and nutrition analysis first!")
        if st.button("⬅️ Go to Food Detection"):
            st.session_state['current_page'] = 'detection'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Nutrition Analysis Section
    st.markdown("### 📊 Detailed Nutrition Analysis")
    
    nutrition = st.session_state['nutrition']
    foods = st.session_state['successful_foods']
    rda = RDA_DATA[st.session_state.get('age_group', 'adult_male')]
    
    # Nutrition metrics and chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📈 Your Meal Breakdown")
        
        for nutrient in ['Calories', 'Protein', 'Carbohydrates', 'Fat']:
            value = nutrition.get(nutrient, 0)
            target = rda.get(nutrient, 1)
            percentage = (value / target) * 100 if target > 0 else 0
            unit = "kcal" if nutrient == "Calories" else "g"
            
            # Color coding based on percentage
            if percentage < 60:
                color = "#d2691e"  # Orange-brown for low
                status = "Low"
            elif percentage > 120:
                color = "#cd853f"  # Sandy brown for high
                status = "High"
            else:
                color = "#8b7765"  # Main beige for good
                status = "Good"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <strong>{nutrient}</strong><br>
                <span style="font-size: 1.2rem; color: {color};">{value:.1f} {unit}</span><br>
                <small>{percentage:.0f}% of {target}{unit} daily target • <strong>{status}</strong></small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Visual Comparison")
        chart = create_nutrition_chart(nutrition, rda)
        st.plotly_chart(chart, use_container_width=True)
    
    # Foods detected summary
    st.markdown("#### 🍽️ Detected Foods")
    food_cols = st.columns(min(len(foods), 4))
    for i, food in enumerate(foods):
        with food_cols[i % 4]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8b7765, #a0896b); color: white; 
                        padding: 0.8rem; border-radius: 10px; text-align: center; margin: 0.2rem;">
                <strong>{food.replace('_', ' ').title()}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Suggestions Section
    st.markdown("### 🤖 Personalized AI Recommendations")
    
    if 'groq_key' in st.session_state and st.session_state['groq_key']:
        if st.button("✨ Generate AI Analysis", type="primary"):
            with st.spinner("🧠 AI is analyzing your meal and generating personalized recommendations..."):
                suggestions = get_ai_suggestions(
                    foods,
                    nutrition,
                    st.session_state.get('age_group', 'adult_male'),
                    st.session_state.get('health_goals', ['General Wellness']),
                    st.session_state.get('dietary_restrictions', []),
                    st.session_state['groq_key']
                )
                
                st.markdown(f"""
                <div class="suggestions-container">
                    <h3>🎯 Your Personalized Nutrition Report</h3>
                    <div style="background: rgba(255,255,255,0.3); padding: 1.5rem; border-radius: 10px; 
                                border: 1px solid rgba(255,255,255,0.4);">
                        {suggestions.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.markdown("#### ⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Analyze New Meal"):
                # Clear current data and go back to detection
                for key in ['foods', 'nutrition', 'successful_foods', 'image']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state['current_page'] = 'detection'
                st.rerun()
        
        with col2:
            if st.button("⚙️ Update Profile"):
                st.session_state['current_page'] = 'api'
                st.rerun()
        
        with col3:
            if st.button("📊 View Charts"):
                # Create additional detailed charts
                st.markdown("#### 🥧 Macronutrient Distribution")
                
                # Pie chart for macronutrients
                macros = ['Protein', 'Carbohydrates', 'Fat']
                values = [nutrition.get(macro, 0) for macro in macros]
                
                if sum(values) > 0:
                    fig_pie = px.pie(
                        values=values, 
                        names=macros,
                        title="Macronutrient Breakdown",
                        color_discrete_sequence=['#8b7765', '#a0896b', '#d4c5a9']
                    )
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b7765, #a0896b); color: white; 
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>🤖 AI Recommendations Available!</h3>
            <p>Add your Groq API key in the configuration page to unlock personalized AI-powered nutrition advice.</p>
            <p>Our AI nutritionist can provide meal ratings, specific recommendations, and health insights tailored to your goals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔑 Configure Groq API"):
            st.session_state['current_page'] = 'api'
            st.rerun()
    
    # Manual insights (always available)
    st.markdown("---")
    st.markdown("### 💡 Quick Insights")
    
    insights = []
    
    # Calorie insights
    cal_percentage = (nutrition.get('Calories', 0) / rda.get('Calories', 1)) * 100
    if cal_percentage < 30:
        insights.append("🔥 This meal is quite light in calories - consider if you need additional food today.")
    elif cal_percentage > 60:
        insights.append("🍽️ This is a substantial meal that covers a large portion of your daily caloric needs.")
    
    # Protein insights
    protein_percentage = (nutrition.get('Protein', 0) / rda.get('Protein', 1)) * 100
    if protein_percentage > 40:
        insights.append("💪 Excellent protein content! Great for muscle maintenance and satiety.")
    elif protein_percentage < 15:
        insights.append("🥩 Consider adding more protein sources to this meal.")
    
    # Carb insights
    carb_percentage = (nutrition.get('Carbohydrates', 0) / rda.get('Carbohydrates', 1)) * 100
    if carb_percentage > 40:
        insights.append("⚡ High in carbohydrates - good for energy, but balance with activity.")
    
    # Fat insights
    fat_percentage = (nutrition.get('Fat', 0) / rda.get('Fat', 1)) * 100
    if fat_percentage > 50:
        insights.append("🥑 High fat content - ensure these are healthy fats from good sources.")
    
    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.success("👍 This meal appears to have a balanced nutritional profile!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
    
    # Navigation buttons (functional)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state['current_page'] = 'home'
            st.rerun()
    with col2:
        if st.button("🔑 API Setup", use_container_width=True):
            st.session_state['current_page'] = 'api'
            st.rerun()
    with col3:
        if st.button("📸 Detection", use_container_width=True):
            st.session_state['current_page'] = 'detection'
            st.rerun()
    with col4:
        if st.button("🤖 Suggestions", use_container_width=True):
            st.session_state['current_page'] = 'suggestions'
            st.rerun()
    
    # Page routing
    if st.session_state['current_page'] == 'home':
        show_home_page()
    elif st.session_state['current_page'] == 'api':
        show_api_page()
    elif st.session_state['current_page'] == 'detection':
        show_detection_page()
    elif st.session_state['current_page'] == 'suggestions':
        show_suggestions_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(139, 119, 101, 0.8);">
        <p>🍽️ <strong>NutriVision AI</strong> - Powered by Computer Vision & AI</p>
        <p>Made with ❤️ using Streamlit • USDA Food Data • Groq AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
