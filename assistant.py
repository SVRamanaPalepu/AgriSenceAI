# assistant.py - AgroSense AI Agricultural Assistant Engine
import os
import json
import re
import requests

SYSTEM_PROMPT = """You are AgroSense AI's Agricultural Assistant.
Your job is to help farmers and students understand crop recommendations, soil conditions, climate factors, and basic agriculture concepts.
Use simple English.
Give clear and practical explanations.
When prediction context is provided, use only the supplied prediction data.
Never invent values.
Never change the ML prediction.
Never claim that a model prediction guarantees crop success.
Clearly explain that actual crop suitability can depend on local soil conditions, season, irrigation, pests, weather, farming practices, and other factors.
For medical, financial, political, or unrelated questions, politely explain that you are focused on agriculture.
Do not provide dangerous pesticide, chemical, or unsafe farming instructions.
If the user asks why a crop was recommended, explain the available prediction context rather than pretending to know hidden model reasoning.
If the user asks for a crop recommendation without providing prediction context, explain that the AgroSense ML prediction requires the seven field inputs and direct them to the Crop Recommendation page.
Always prioritize clarity and farmer-friendly language."""

def get_agricultural_knowledge(query, context, chat_history):
    """
    Intelligent Agricultural Knowledge Engine for agronomic concepts,
    field parameters, prediction explanations, and multi-turn follow-ups.
    """
    q = query.strip().lower()
    crop = context.get("primary_crop", "").capitalize() if context else None
    
    # 1. Prediction Explanation: Why was this crop recommended?
    if any(phrase in q for phrase in ["why was this crop", "why was", "why this crop", "why recommend", "why rice", "why maize", "why cotton", "explain my prediction", "explain the prediction"]):
        if context:
            c_name = context.get("primary_crop", "the selected crop").capitalize()
            conf = context.get("confidence", "")
            n = context.get("nitrogen", "")
            p = context.get("phosphorus", "")
            k = context.get("potassium", "")
            ph = context.get("ph", "")
            temp = context.get("temperature", "")
            hum = context.get("humidity", "")
            rain = context.get("rainfall", "")
            
            resp = (
                f"{c_name} was recommended because the specific soil and climate values you provided "
                f"(Nitrogen: {n} kg/ha, Phosphorus: {p} kg/ha, Potassium: {k} kg/ha, Soil pH: {ph}, "
                f"Temperature: {temp}°C, Humidity: {hum}%, Rainfall: {rain} mm) match the optimal growing "
                f"conditions associated with {c_name} in our trained machine learning dataset"
            )
            if conf:
                resp += f" with a model confidence of {conf}%."
            else:
                resp += "."
            resp += (
                "\n\n🌱 The model evaluates all 7 parameters together to identify the most compatible crop. "
                "Please note that local soil texture, seasonal timing, irrigation access, and pest management "
                "should also be considered before planting."
            )
            return resp
        else:
            return (
                "To explain a specific recommendation, please first submit your 7 field values "
                "(Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, Rainfall) on the "
                "Find Crop page. The AI model will then evaluate your land and I can explain the result in detail!"
            )

    # 2. Tell me about the recommended crop / Crop specifics
    if any(phrase in q for phrase in ["tell me about the recommended crop", "about the recommended crop", "about this crop", "about the crop", "tell me about"]):
        if context and crop:
            return (
                f"🌱 **About {crop}:**\n\n"
                f"{crop} is a well-suited crop for your entered conditions. It thrives with the balanced combination "
                f"of soil nutrients and environmental factors you provided (pH {context.get('ph', '')}, "
                f"Temperature {context.get('temperature', '')}°C, Rainfall {context.get('rainfall', '')} mm).\n\n"
                f"For best yields, ensure timely seedbed preparation, adequate weed management, and monitor soil moisture throughout the growing cycle."
            )
        elif "rice" in q:
            return "Rice is a staple cereal grain that typically thrives in warm, humid conditions with abundant water or rainfall and fertile, moisture-retentive soils."
        elif "maize" in q or "corn" in q:
            return "Maize is a versatile cereal grain requiring moderate to warm temperatures, good sunlight, and well-drained loamy soils rich in nitrogen."
        elif "cotton" in q:
            return "Cotton is a major fiber crop that requires warm temperatures, moderate rainfall, and well-drained soils with balanced potassium."

    # 3. Is my soil suitable / Soil condition
    if any(phrase in q for phrase in ["is my soil suitable", "soil suitable", "soil condition", "how is my soil"]):
        if context:
            n = context.get("nitrogen", "")
            p = context.get("phosphorus", "")
            k = context.get("potassium", "")
            ph = context.get("ph", "")
            return (
                f"Based on your input values (Nitrogen: {n} kg/ha, Phosphorus: {p} kg/ha, Potassium: {k} kg/ha, pH: {ph}), "
                f"your soil conditions are consistent with the requirements for {crop if crop else 'the recommended crop'} in our dataset.\n\n"
                "You can review individual nutrient statuses (Suitable, Moderate, Needs Attention) on the **Soil & Climate Analysis** page."
            )
        else:
            return "Soil suitability depends on your Nitrogen, Phosphorus, Potassium levels and Soil pH. Enter your soil test values in the Find Crop form to check suitability for 22+ different crops!"

    # 4. Multi-turn Follow-up: "What does the first nutrient mean?" / "Which nutrient..."
    if any(phrase in q for phrase in ["first nutrient", "first one", "what does the first"]):
        return (
            "The first nutrient in NPK is **Nitrogen (N)**. It is essential for leafy green vegetative growth, "
            "chlorophyll production (photosynthesis), and building proteins in plants."
        )
    if any(phrase in q for phrase in ["second nutrient", "second one"]):
        return (
            "The second nutrient in NPK is **Phosphorus (P)**. It is critical for root development, flowering, "
            "seed formation, and early plant energy transfer."
        )
    if any(phrase in q for phrase in ["third nutrient", "third one"]):
        return (
            "The third nutrient in NPK is **Potassium (K)**. It regulates water balance inside plant cells, "
            "strengthens stalks against diseases, and improves crop resilience against drought and cold."
        )

    # 5. What is NPK?
    if "npk" in q:
        return (
            "**NPK** stands for the three primary macronutrients essential for healthy plant growth:\n\n"
            "1. **N — Nitrogen:** Promotes vigorous leaf and stem development and rich green color.\n"
            "2. **P — Phosphorus:** Stimulates deep root growth, blooming, and seed formation.\n"
            "3. **K — Potassium (Potash):** Enhances overall plant vigor, water regulation, and disease resistance.\n\n"
            "Balanced NPK levels ensure crops reach their full growth potential."
        )

    # 6. What is Nitrogen / What does nitrogen do?
    if "nitrogen" in q:
        return (
            "**Nitrogen (N)** is a vital plant nutrient needed for producing chlorophyll, which plants use to convert sunlight into energy (photosynthesis). "
            "Adequate nitrogen leads to lush, healthy green foliage and strong vegetative growth."
        )

    # 7. What is Phosphorus / What does phosphorus do?
    if "phosphorus" in q:
        return (
            "**Phosphorus (P)** is a key macronutrient that aids in energy transfer within plant cells. "
            "It is especially important for root establishment, flower budding, fruit set, and uniform seed development."
        )

    # 8. What is Potassium / What does potassium do?
    if "potassium" in q:
        return (
            "**Potassium (K)** helps regulate the opening and closing of plant leaf pores (stomata), managing water retention and transpiration. "
            "It strengthens crop stems, improves grain quality, and bolsters resistance against pests and adverse weather."
        )

    # 9. What is Soil pH?
    if "ph" in q or "soil ph" in q:
        return (
            "**Soil pH** measures the acidity or alkalinity of the soil on a scale from 0 to 14:\n\n"
            "• **0 - 6.5:** Acidic Soil\n"
            "• **6.5 - 7.5:** Neutral Soil (ideal for most agricultural crops)\n"
            "• **7.5 - 14:** Alkaline Soil\n\n"
            "Soil pH directly affects how easily plant roots can absorb essential nutrients like Nitrogen, Phosphorus, and Potassium."
        )

    # 10. What does Humidity mean?
    if "humidity" in q:
        return (
            "**Relative Humidity (%)** refers to the amount of moisture present in the air compared to the maximum amount it could hold at that temperature. "
            "High humidity reduces plant water loss through transpiration, which benefits tropical crops like rice and banana, while lower humidity suits dryland grains."
        )

    # 11. What does Rainfall affect?
    if "rainfall" in q or "rain" in q:
        return (
            "**Rainfall (mm)** provides natural soil moisture required for seed germination, nutrient uptake, and plant hydration. "
            "Different crops have distinct water requirements — for example, rice requires high rainfall (>200 mm), while crops like chickpea and lentil thrive in drier conditions."
        )

    # 12. What does Temperature affect?
    if "temperature" in q or "temp" in q:
        return (
            "**Temperature (°C)** governs plant metabolic rates, enzyme activity, photosynthesis, and growth speed. "
            "Warm-season crops (like maize, cotton, and watermelon) prefer 24°C - 35°C, while cool-season crops (like chickpea and apple) require lower temperatures."
        )

    # 13. General farming / greeting fallback
    if any(greet in q for greet in ["hello", "hi", "hey", "help", "who are you"]):
        intro = "Hello! I am AgroSense AI's Agricultural Assistant. "
        if context and crop:
            intro += f"I can help explain why **{crop}** was recommended for your field, explain NPK nutrients, soil pH, climate factors, and answer farming questions."
        else:
            intro += "I can answer questions about soil nutrients (NPK), soil pH, climate factors, crops, and explain your field recommendations."
        return intro

    # Default informative answer
    if context and crop:
        return (
            f"Regarding your land and the recommendation for **{crop}**: our system evaluated your soil nutrients "
            f"(N: {context.get('nitrogen', '')}, P: {context.get('phosphorus', '')}, K: {context.get('potassium', '')}), "
            f"pH ({context.get('ph', '')}), and climate conditions (Temp: {context.get('temperature', '')}°C, "
            f"Humidity: {context.get('humidity', '')}%, Rainfall: {context.get('rainfall', '')} mm). "
            "Feel free to ask about any specific nutrient, climate factor, or farming practice!"
        )
    
    return (
        "In agriculture, crop performance depends on the balance of soil nutrients (Nitrogen, Phosphorus, Potassium), "
        "soil pH, and environmental climate conditions (Temperature, Humidity, Rainfall). "
        "You can ask me about any of these parameters or submit your field values in the Find Crop form for a tailored recommendation!"
    )

def call_gemini_api(user_message, context, chat_history, api_key, model_name="gemini-1.5-flash"):
    """
    Sends request to Gemini REST API with agricultural system instruction and context.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    # Build context string
    context_str = ""
    if context:
        context_str = (
            f"Active Field Prediction Context:\n"
            f"- Recommended Crop: {context.get('primary_crop', '')}\n"
            f"- Model Confidence: {context.get('confidence', '')}%\n"
            f"- Nitrogen (N): {context.get('nitrogen', '')} kg/ha\n"
            f"- Phosphorus (P): {context.get('phosphorus', '')} kg/ha\n"
            f"- Potassium (K): {context.get('potassium', '')} kg/ha\n"
            f"- Soil pH: {context.get('ph', '')}\n"
            f"- Temperature: {context.get('temperature', '')} °C\n"
            f"- Relative Humidity: {context.get('humidity', '')} %\n"
            f"- Rainfall: {context.get('rainfall', '')} mm\n"
        )
        if context.get("top_predictions"):
            top_crops_str = ", ".join([f"{c['crop']} ({c.get('confidence', '')}%)" for c in context['top_predictions']])
            context_str += f"- Top Alternative Crops: {top_crops_str}\n"

    # Build conversation contents
    contents = []
    # System instruction as initial developer context
    full_system = SYSTEM_PROMPT + ("\n\n" + context_str if context_str else "")
    
    # Add history
    for item in chat_history[-6:]:
        role = "user" if item["sender"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": item["text"]}]
        })
    
    # Add current query
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    payload = {
        "systemInstruction": {
            "parts": [{"text": full_system}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 600
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates:
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if text:
                return text.strip()
    return None

def get_assistant_response(user_message, context=None, chat_history=None):
    """
    Main entry point for assistant replies with API calling and agricultural engine fallback.
    """
    if chat_history is None:
        chat_history = []
        
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "gemini-1.5-flash")
    
    # Try live LLM call if API key is provided
    if api_key and api_key.strip() and api_key != "your_gemini_api_key_here":
        try:
            llm_reply = call_gemini_api(user_message, context, chat_history, api_key.strip(), model_name)
            if llm_reply:
                return llm_reply
        except Exception:
            pass  # Fall through to knowledge engine on any network or API error
            
    # Use intelligent agricultural knowledge engine
    return get_agricultural_knowledge(user_message, context, chat_history)
