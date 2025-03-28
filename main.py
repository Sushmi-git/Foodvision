import requests
import json

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
    "Calories": 1008,       # Energy (kcal)
    "Protein": 1003,        # Protein (g)
    "Carbohydrates": 1005,  # Carbohydrates (g)
    "Fat": 1004,            # Total Fat (g)
    "Sodium": 1093          # Sodium (mg)
}

def get_fdc_id(food_name):
    """Fetch the FDC ID for a given food item"""
    search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={USDA_API_KEY}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        if "foods" in data and len(data["foods"]) > 0:
            return data["foods"][0]["fdcId"]  # Selecting the first available FDC ID
    return None

def get_nutrition_data(fdc_id):
    """Fetch the nutrition data for a given FDC ID"""
    food_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={USDA_API_KEY}"
    response = requests.get(food_url)
    
    if response.status_code == 200:
        return response.json()
    return None

def extract_key_nutrients(nutrition_data):
    """Extract only the necessary nutrients using USDA standard nutrient IDs"""
    extracted_nutrients = {key: 0 for key in NUTRIENT_IDS.keys()}  # Default values

    if "foodNutrients" in nutrition_data:
        for nutrient in nutrition_data["foodNutrients"]:
            nutrient_id = nutrient.get("nutrient", {}).get("id")
            for key, usda_id in NUTRIENT_IDS.items():
                if nutrient_id == usda_id:
                    extracted_nutrients[key] = nutrient.get("amount", 0)

    return extracted_nutrients

def compare_with_rda(nutrition, age_group):
    """Compare the meal's nutrition with the RDA values"""
    rda = RDA_DATA.get(age_group, {})
    comparison = {}

    for key in RDA_DATA["adult_male"].keys():  # Use keys from any RDA group
        comparison[key] = {
            "value": nutrition.get(key, 0),
            "RDA": rda.get(key, 0),
            "percentage": round((nutrition.get(key, 0) / rda.get(key, 1)) * 100, 1)  # Avoid division by zero
        }

    return comparison

def get_groq_suggestions(food_name, nutrition, age_group):
    """Use Groq LLM to suggest better meal plans based on deficiencies"""
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

    if response.status_code == 200:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    else:
        return "Error: Could not fetch meal suggestions."

def main():
    food_name = input("Enter food name: ")
    age_group = input("Enter age group (e.g., child_4_8, adult_male): ")

    fdc_id = get_fdc_id(food_name)
    if fdc_id:
        print(f"Fetching nutrition data for {food_name} (FDC ID: {fdc_id})...\n")
        nutrition_data = get_nutrition_data(fdc_id)

        if nutrition_data:
            key_nutrients = extract_key_nutrients(nutrition_data)

            food_info = {
                "Food": food_name,
                "Nutrients": key_nutrients
            }

            print("\n🔹 Fetched Nutrition Data:")
            print(json.dumps(food_info, indent=4))

            # Compare with RDA
            comparison = compare_with_rda(key_nutrients, age_group)
            print("\n🔹 Nutrition Comparison with RDA:")
            print(json.dumps(comparison, indent=4))

            # Get meal suggestions from Groq LLM
            print("\n🔹 Groq's Meal Suggestions: ")
            suggestions = get_groq_suggestions(food_name, key_nutrients, age_group)
            print(suggestions)

        else:
            print("No nutrition data found.")
    else:
        print("Could not retrieve FDC ID.")

if __name__ == "__main__":
    main()
