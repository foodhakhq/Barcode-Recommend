import os
import re
import json
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Header, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import logging
from httpx import Timeout, ReadTimeout
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI()
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment variables and configurations
STAGING_OPENAI_API_KEY = os.getenv("STAGING_OPENAI_API_KEY")
STAGING_API_KEY = os.getenv("STAGING_API_KEY")


async def safe_send(websocket: WebSocket, message: dict) -> bool:
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected while sending message.")
        return False
    except RuntimeError as e:
        logger.warning(f"Runtime error while sending message: {e}")
        return False
    return True


def get_openai_client():
    api_key = os.getenv("STAGING_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("STAGING_OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    return client


# Bearer token dependency
async def verify_token(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split(" ")[1]
    if token != STAGING_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


# Define request body schema
class RecommendationRequest(BaseModel):
    user_id: str
    barcode: str


# Function to get user profile (Async)
async def get_user_profile(user_id):
    url = f"{os.getenv('STAGING_OPENSEARCH_HOST')}/user-profiles/_search"
    user = os.getenv("STAGING_OPENSEARCH_USER")
    password = os.getenv("STAGING_OPENSEARCH_PWD")
    query = {"query": {"match": {"foodhak_user_id": user_id}}}
    async with httpx.AsyncClient(auth=(user, password)) as client:
        response = await client.post(url, json=query)  # Use POST since it includes a JSON body.

    if response.status_code == 200:
        results = response.json()
        if results['hits']['total']['value'] > 0:
            return results['hits']['hits'][0]['_source']
        else:
            return None
    else:
        raise HTTPException(status_code=500, detail=f"Error fetching user profile: {response.status_code}")


# Function to get product details
async def get_product_details(barcode, retries=3):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    timeout = Timeout(10.0, connect=5.0)
    attempt = 0
    while attempt < retries:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json().get('product', {})
                else:
                    raise HTTPException(status_code=500,
                                        detail=f"Error fetching product details: {response.status_code}")
        except ReadTimeout:
            attempt += 1
            if attempt >= retries:
                raise HTTPException(status_code=504,
                                    detail="Timeout occurred while fetching product details after multiple attempts.")
        await asyncio.sleep(1)  # Optional: Add a delay between retries


# Function to extract ingredients (Sync)
def extract_ingredients(ingredients_text):
    ingredients_text = ingredients_text.strip()
    pattern = r',\s*(?![^()]*\))'
    ingredients = re.split(pattern, ingredients_text)
    cleaned_ingredients = []
    for ingredient in ingredients:
        sub_pattern = r'[:;]\s*(?![^()]*\))'
        sub_ingredients = re.split(sub_pattern, ingredient)
        for sub_ingredient in sub_ingredients:
            sub_ingredient = re.sub(r'\(.*?\)', '', sub_ingredient)
            sub_ingredient = re.sub(r'\d+%|\d+\.\d+%', '', sub_ingredient)
            sub_ingredient = sub_ingredient.strip().lower()
            if sub_ingredient:
                cleaned_ingredients.append(sub_ingredient)
    return cleaned_ingredients


# Async function to query vector store (Async)
async def query_vector_store_async(ingredient_names):
    unique_ingredients = []
    for ingredient in ingredient_names:
        if ingredient not in unique_ingredients:
            unique_ingredients.append(ingredient)
    ingredient_names = unique_ingredients
    logger.info(f"Ingredients for vector store query: {ingredient_names}")
    query_texts = [f"Health benefits of {ingredient} ?" for ingredient in ingredient_names]
    url = "https://ai-foodhak.com/chromadb_vecstore"
    headers = {"Content-Type": "application/json"}
    data = {"queries": query_texts, "count": 1}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers)
    try:
        response_json = response.json()
    except Exception as e:
        raise Exception("Failed to parse JSON response from vector store") from e

    combined_context = json.dumps(response_json, indent=2)
    return combined_context


# Function to extract detailed user information
def extract_user_info(user_profile):
    user_health_goals = user_profile.get('user_health_goals', [])
    goals = [goal.get('user_goal', {}).get('title', 'Unknown goal') for goal in user_health_goals]
    recommended_ingredients = []
    avoided_ingredients = []
    for goal in user_health_goals:
        goal_info = goal.get('user_goal', {})
        goal_title = goal_info.get('title', 'Unknown goal')
        for ingredient in goal.get('ingredients_to_recommend', []):
            recommended_ingredients.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'goal': goal_title,
                'relationship': ingredient.get('relationships', [])
            })
        for ingredient in goal.get('ingredients_to_avoid', []):
            avoided_ingredients.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'goal': goal_title,
                'relationship': ingredient.get('relationships', [])
            })

    dietary_preferences = user_profile.get('dietary_preferences', 'Not specified')
    dietary_restrictions = user_profile.get('dietary_restrictions', {}).get('name', 'None')
    allergens = [allergen.get('type', 'Unknown') for allergen in user_profile.get('allergens', [])]
    diseases = [disease.get('data', {}).get('name', 'Unknown disease') for disease in user_profile.get('diseases', [])]

    disease_recommendations = []
    disease_avoidances = []

    for disease in user_profile.get('diseases', []):
        disease_name = disease.get('data', {}).get('name', 'Unknown disease')

        for ingredient in disease.get('ingredients_to_recommend', []):
            disease_recommendations.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'disease': disease_name,
                'relationship': ingredient.get('relationships', [])
            })
        for ingredient in disease.get('ingredients_to_avoid', []):
            disease_avoidances.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'disease': disease_name,
                'relationship': ingredient.get('relationships', [])
            })
    all_recommendations = recommended_ingredients + disease_recommendations
    all_avoidances = avoided_ingredients + disease_avoidances
    # Nutrient mapping dictionary
    nutrient_mapping = {
        "energy": "Energy (KCAL)",
        "protein": "Protein (G)",
        "fats": "Total Fat (G)",
        "saturated fat": "Saturated Fat (G)",
        "cholesterol": "Cholesterol (MG)",
        "sodium": "Sodium Na (MG)",
        "carbohydrates": "Total Carbohydrate (G)",
        "dietary fibre": "Dietary Fiber (G)",
        "vitamin c": "Vitamin C (MG)",
        "calcium": "Calcium (MG)",
        "iron": "Iron (MG)",
        "potassium": "Potassium K (MG)",
        "hydration": "Hydration (ML)"
    }
    # Extract and format nutrients correctly
    nutrients_data = user_profile.get("nutrients", {}).get("results", {})
    formatted_nutrients = {}

    for nutrient_type_list in nutrients_data.values():
        for nutrient_type in nutrient_type_list:
            item_name = nutrient_type.get("nutrition_guideline", {}).get("item", "").strip()
            item_name_lower = item_name.lower() if item_name else ""  # Safety check for None

            if item_name_lower in nutrient_mapping:
                formatted_name = nutrient_mapping[item_name_lower]
                formatted_nutrients[formatted_name] = str(nutrient_type.get("target_value"))

    daily_nutritional_requirement = formatted_nutrients
    # print(daily_nutritional_requirement)
    return (
        goals,
        dietary_preferences,
        dietary_restrictions,
        allergens,
        diseases,
        all_recommendations,
        all_avoidances,
        daily_nutritional_requirement
    )


# Generate response with Claude (Primary API)
async def generate_response_with_claude(
        client,
        user_prompt,
        system_prompt,
        websocket=None,
        model="claude-3-7-sonnet-20250219",
        max_tokens=5024
):
    try:
        # Initialize variables
        message_id = None
        response = ""

        # Stream chunks from Claude
        async with client.beta.prompt_caching.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=user_prompt,
                temperature = 0,
        ) as stream:
            async for chunk in stream:
                if hasattr(chunk, 'type'):
                    event_type = chunk.type

                    if event_type == "message_start":
                        # Extract message ID
                        message_id = chunk.message.id

                    elif event_type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                        # Stream chunk of text
                        delta_text = chunk.delta.text
                        response += delta_text

                        if websocket:
                            sent_ok = await safe_send(websocket, {
                                "message_id": message_id,
                                "type": "streaming",
                                "data": delta_text
                            })
                            if not sent_ok:
                                break
                    elif event_type == "message_stop":
                        # Send message_stop event
                        if websocket:
                            sent_ok = await safe_send(websocket, {
                                "message_id": message_id,
                                "type": "message_stop",
                                "data": "Message stream completed."
                            })
                            if not sent_ok:
                                break
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response with Claude: {str(e)}")


# Generate response with Openai (Fallback API)
async def generate_response_with_openai_streaming(client, prompt, system, websocket):
    try:
        message_id = None
        full_response = ""
        # Build the messages list using system instruction and user prompt.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        # Create a streaming chat completion.
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust model as needed.
            messages=messages,
            max_tokens=5024,
            temperature =0,
            stream=True,
        )
        # Process each chunk from the streaming response.
        for chunk in response:
            # Use the chunk ID as a message_id if not already set.
            if message_id is None:
                message_id = chunk.id  # This is our simulated message ID.

            # Simulate "message_start": Check if this chunk carries the role info.
            # OpenAI sends role information only once, typically on the first chunk.
            delta = chunk.choices[0].delta
            if message_id and hasattr(delta, "role") and delta.role:
                # Send a message start event.
                sent_ok = await safe_send(websocket, {
                    "message_id": message_id,
                    "type": "message_start",
                    "data": f"Role: {delta.role}"
                })
                if not sent_ok:
                    break

            # Simulate "content_block_delta": If there is new text content.
            if hasattr(delta, "content") and delta.content:
                delta_text = delta.content
                full_response += delta_text
                sent_ok = await safe_send(websocket, {
                    "message_id": message_id,
                    "type": "streaming",
                    "data": delta_text
                })
                if not sent_ok:
                    break
                # Brief sleep to yield control.
                await asyncio.sleep(0.01)

            # Simulate "message_stop": If finish_reason is set, it indicates completion.
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                sent_ok = await safe_send(websocket, {
                    "message_id": message_id,
                    "type": "message_stop",
                    "data": "Message stream completed."
                })
                if not sent_ok:
                    break
        return full_response
    except Exception as e:
        logging.error(f"Error streaming response with OpenAI: {e}")
        sent_ok = await safe_send(websocket, {
            "type": "error",
            "data": str(e)
        })
        if not sent_ok:
            logging.warning("Attempted to send error message on a closed WebSocket")
        return ""


# Function to generate recommendation summary using Gemini
async def generate_recommendation_summary(user_profile, product_details, vector_store_context):
    product_name = product_details.get('product_name', '').strip()
    fallback_product_names = [
        product_details.get('product_name_en', '').strip(),
        product_details.get('product_name_es', '').strip(),
        product_details.get('product_name_fr', '').strip(),
        product_details.get('product_name_pl', '').strip(),
        product_details.get('product_name_pt', '').strip(),
        product_details.get('product_name_zh', '').strip(),
        product_details.get('product_name_de', '').strip(),
        product_details.get('product_name_it', '').strip(),
        product_details.get('product_name_ru', '').strip(),
        product_details.get('product_name_no', '').strip(),
        product_details.get('product_name_fi', '').strip()
    ]
    product_name_fallback = product_name if product_name else next((name for name in fallback_product_names if name), "Unnamed")

    goals, dietary_preferences, dietary_restrictions, allergens, diseases, all_recommendations, all_avoidances, daily_nutritional_requirement = extract_user_info(
        user_profile)

    recommendations_text = "\n".join([
        f"{rec['common_name']} is recommended for {rec.get('goal', rec.get('disease', 'Unknown'))}. "
        f"{rec['relationship'][0].get('extracts', 'No relationship data available') if rec.get('relationship') and len(rec['relationship']) > 0 else 'No relationship data available'}"
        for rec in all_recommendations if rec.get('common_name')
    ])
    avoidances_text = "\n".join([
        f"{avoid['common_name']} should be avoided for {avoid.get('goal', avoid.get('disease', 'Unknown'))}. "
        f"{avoid['relationship'][0].get('extracts', 'No relationship data available') if avoid.get('relationship') and len(avoid['relationship']) > 0 else 'No relationship data available'}"
        for avoid in all_avoidances if avoid.get('common_name')
    ])

    system_prompt = [{
        "type": "text",
        "text": f"""
        You are **Foodhak Assistant**, the expert nutrition sidekick in the Foodhak app. Whenever a user scans a product, your mission is to deliver a concise, personalized verdict—highlighting key nutrition points and whether the item aligns with their goals and daily targets.

        ---
        ### 1. Voice & Style
        - **Evidence-based**: draw on the Foodhak database and the latest research.  
        - **Tone**: warm, supportive, and motivational—use emojis sparingly.  
        - **Concise**: limit to 2–4 short sentences suitable for UI display.
        - **Wellness Expertise**: use your own knowledge to supplement answers with general health and wellness guidance.

        ### 2. Personalization
        When available, weave in this profile data:
        name:          {user_profile.get('name', 'Friend')}
        age:           {user_profile.get('age', 'Not specified')}
        sex:           {user_profile.get('sex', 'Not specified')}
        height_cm:     {user_profile.get('height', 'N/A')}
        weight_kg:     {user_profile.get('weight', 'N/A')}
        ethnicity:     {user_profile.get('ethnicity', {}).get('title', 'N/A')}
        all_goals:     {', '.join(goals) or 'N/A'}
        preferences:   {dietary_preferences or 'None'}
        restrictions:  {dietary_restrictions or 'None'}
        allergies:     {', '.join(allergens) or 'None'}
        daily_targets: {daily_nutritional_requirement}

        **3. Product Context
        Use these placeholders for the scanned item:
        product_name:  {product_name}
        nutriments:    {product_details.get('nutriments', 'Unavailable')}
        ingredients:   {product_details.get('ingredients_text', 'Unavailable')}
        nutri_score:   {product_details.get('nutriscore_grade', 'N/A')}
        labels:        {product_details.get('labels', 'None')}

        **4. Behavior Rules
        Suitability verdict:
        If the product fits the profile and daily targets, affirm suitability and note one or two key benefits.
        If it falls short, state why and—only then—suggest 1–2 realistic alternatives.
        No unsolicited swaps: avoid offering alternatives when the scanned product is already a good fit.
        Transparency: acknowledge missing data when necessary.

        **5. Formatting
        Markdown only:
        Use - for lists and **bold** for emphasis.
        No greetings or sign-offs: jump straight into the verdict.
        UI-friendly: keep text snappy and on-point.
        """,
        "cache_control": {"type": "ephemeral"}
    }]

    user_prompt = [{
        "role": "user",
        "content": f"""
        Scan Data:
        - Product: {product_name}
        - Nutri-Score: {product_details.get('nutriscore_grade', 'N/A')}
        - Key Nutriments: {product_details.get('nutriments', 'Unavailable')}

        User Profile:
        - Goals: {', '.join(goals) or 'N/A'}
        - Daily Targets: {daily_nutritional_requirement}

        Ingredient Insights:
        - Recommended: {recommendations_text or 'None highlighted'}
        - To Avoid: {avoidances_text or 'None highlighted'}

        Additional Context:
        {vector_store_context or 'No extra ingredient data available.'}

        Please provide a **2–4-sentence**, **friendly**, **personalized** summary stating:
        1. Whether **{product_name}** suits the user's profile and why.  
        2. If not suitable, explain briefly **why** and give **1–2** alternative suggestions.  
        3. Highlight one key benefit when it is suitable.
        """
    }]

    system_prompt_openai = f"""
        You are **Foodhak Assistant**, the expert nutrition sidekick in the Foodhak app. Whenever a user scans a product, your mission is to deliver a concise, personalized verdict—highlighting key nutrition points and whether the item aligns with their goals and daily targets.

        ---
        ### 1. Voice & Style
        - **Evidence-based**: draw on the Foodhak database and the latest research.  
        - **Tone**: warm, supportive, and motivational—use emojis sparingly.  
        - **Concise**: limit to 2–4 short sentences suitable for UI display.
        - **Wellness Expertise**: use your own knowledge to supplement answers with general health and wellness guidance.

        ### 2. Personalization
        When available, weave in this profile data:
        name:          {user_profile.get('name', 'Friend')}
        age:           {user_profile.get('age', 'Not specified')}
        sex:           {user_profile.get('sex', 'Not specified')}
        height_cm:     {user_profile.get('height', 'N/A')}
        weight_kg:     {user_profile.get('weight', 'N/A')}
        ethnicity:     {user_profile.get('ethnicity', {}).get('title', 'N/A')}
        all_goals:     {', '.join(goals) or 'N/A'}
        preferences:   {dietary_preferences or 'None'}
        restrictions:  {dietary_restrictions or 'None'}
        allergies:     {', '.join(allergens) or 'None'}
        daily_targets: {daily_nutritional_requirement}

        **3. Product Context
        Use these placeholders for the scanned item:
        product_name:  {product_name}
        nutriments:    {product_details.get('nutriments', 'Unavailable')}
        ingredients:   {product_details.get('ingredients_text', 'Unavailable')}
        nutri_score:   {product_details.get('nutriscore_grade', 'N/A')}
        labels:        {product_details.get('labels', 'None')}

        **4. Behavior Rules
        Suitability verdict:
        If the product fits the profile and daily targets, affirm suitability and note one or two key benefits.
        If it falls short, state why and—only then—suggest 1–2 realistic alternatives.
        No unsolicited swaps: avoid offering alternatives when the scanned product is already a good fit.
        Transparency: acknowledge missing data when necessary.

        **5. Formatting
        Markdown only:
        Use - for lists and **bold** for emphasis.
        No greetings or sign-offs: jump straight into the verdict.
        UI-friendly: keep text snappy and on-point.
        """
    user_prompt_openai = f"""
        Scan Data:
        - Product: {product_name}
        - Nutri-Score: {product_details.get('nutriscore_grade', 'N/A')}
        - Key Nutriments: {product_details.get('nutriments', 'Unavailable')}

        User Profile:
        - Goals: {', '.join(goals) or 'N/A'}
        - Daily Targets: {daily_nutritional_requirement}

        Ingredient Insights:
        - Recommended: {recommendations_text or 'None highlighted'}
        - To Avoid: {avoidances_text or 'None highlighted'}

        Additional Context:
        {vector_store_context or 'No extra ingredient data available.'}

        Please provide a **2–4-sentence**, **friendly**, **personalized** summary stating:
        1. Whether **{product_name}** suits the user's profile and why.  
        2. If not suitable, explain briefly **why** and give **1–2** alternative suggestions.  
        3. Highlight one key benefit when it is suitable.
        """
    return user_prompt, system_prompt, user_prompt_openai, system_prompt_openai


@app.websocket("/ws/recommend/{user_id}")
async def websocket_recommend(websocket: WebSocket, user_id: str):
    # Retrieve the clients from app.state
    claude_client = websocket.app.state.claude_client
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()
                barcode = data.get("barcode")
                if not barcode:
                    sent_ok = await safe_send(websocket, {
                        "error": {
                            "code": "WS-400",
                            "message": "Invalid Payload",
                            "description": "'barcode' is required."
                        }
                    })
                    if sent_ok:
                        await websocket.close()
                    return

                # Fetch user profile
                try:
                    user_profile = await get_user_profile(user_id)
                    if not user_profile:
                        sent_ok = await safe_send(websocket, {
                            "error": {
                                "code": "WS-404",
                                "message": "User Profile Not Found",
                                "description": f"The specified user ID {user_id} does not exist."
                            }
                        })
                        if sent_ok:
                            await websocket.close()
                        return
                except Exception as e:
                    sent_ok = await safe_send(websocket, {
                        "error": {
                            "code": "WS-500",
                            "message": "Internal Server Error",
                            "description": f"Error fetching user profile: {str(e)}"
                        }
                    })
                    if sent_ok:
                        await websocket.close()
                    return

                # Fetch product details
                try:
                    product_details = await get_product_details(barcode)
                    if not product_details:
                        sent_ok = await safe_send(websocket, {
                            "error": {
                                "code": "WS-404",
                                "message": "Product Details Not Found",
                                "description": f"No details found for the provided barcode {barcode}."
                            }
                        })
                        if sent_ok:
                            await websocket.close()
                        return
                except Exception as e:
                    sent_ok = await safe_send(websocket, {
                        "error": {
                            "code": "WS-500",
                            "message": "Internal Server Error",
                            "description": f"Error fetching product details: {str(e)}"
                        }
                    })
                    if sent_ok:
                        await websocket.close()
                    return

                # Extract ingredients and perform vector store query
                ingredients_text = product_details.get("ingredients_text", "")
                ingredient_names = extract_ingredients(ingredients_text) if ingredients_text else []
                vector_store_context = None
                try:
                    if ingredient_names:
                        vector_store_context = await query_vector_store_async(ingredient_names)
                except Exception as e:
                    sent_ok = await safe_send(websocket, {
                        "error": {
                            "code": "WS-500",
                            "message": "Internal Server Error",
                            "description": f"Error querying vector store: {str(e)}"
                        }
                    })
                    if sent_ok:
                        await websocket.close()
                    return

                # Generate recommendation summary and associated prompts
                try:
                    user_prompt, system_prompt, user_prompt_openai, system_prompt_openai = await generate_recommendation_summary(
                        user_profile, product_details, vector_store_context
                    )
                except Exception as e:
                    sent_ok = await safe_send(websocket, {
                        "error": {
                            "code": "WS-500",
                            "message": "Internal Server Error",
                            "description": f"Error generating recommendation summary: {str(e)}"
                        }
                    })
                    if sent_ok:
                        await websocket.close()
                    return

                # Stream response with fallback logic using Claude (primary API)
                try:
                    await generate_response_with_claude(
                        client=claude_client,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        websocket=websocket
                    )
                except Exception as anthropic_error:
                    # Log the error to see what args look like
                    logging.error(f"Anthropic error args: {anthropic_error.args}")
                    error_data = anthropic_error.args[0] if anthropic_error.args else {}

                    # If error_data is not a dict, try to parse it as JSON
                    if not isinstance(error_data, dict):
                        try:
                            error_data = json.loads(error_data)
                        except Exception:
                            error_data = {}

                    # Also check the error message string as a fallback
                    error_str = str(anthropic_error)

                    if error_data.get("error", {}).get("type") == "overloaded_error" or "overloaded_error" in error_str:
                        logging.warning("Anthropic Claude is overloaded. Falling back to OpenAI...")
                        openai_client = get_openai_client()
                        await generate_response_with_openai_streaming(
                            openai_client,
                            user_prompt_openai,
                            system_prompt_openai,
                            websocket
                        )
                    else:
                        sent_ok = await safe_send(websocket, {
                            "error": {
                                "code": e.status_code,
                                "message": "Error generating response",
                                "description": str(e.detail)
                            }
                        })
                        if sent_ok:
                            await websocket.close()
                        return

            except json.JSONDecodeError:
                sent_ok = await safe_send(websocket, {
                    "error": {
                        "code": "WS-400",
                        "message": "Invalid JSON",
                        "description": "The received data is not valid JSON."
                    }
                })
                if sent_ok:
                    await websocket.close()
                return

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for user_id: {user_id}")
    finally:
        await websocket.close()


@app.post("/barcode-recommend")
async def barcode_recommend(request: RecommendationRequest, token_valid: str = Depends(verify_token)):
    user_id = request.user_id
    websocket_url = f"wss://staging.ai-foodhak.com/ws/recommend/{user_id}"
    return {
        "websocket_url": websocket_url,
        "message": "Please use this WebSocket URL to connect for real-time recommendations."
    }


@app.get("/health")
async def health_check():
    print("Barcode Scanner Health Check up")
    return {"status": "healthy", "message": "Service is up and running."}


@app.on_event("startup")
async def startup_event():
    required_vars = [
        "ANTHROPIC_STAGING_API_KEY",
        "STAGING_API_KEY",
        "STAGING_OPENSEARCH_HOST",
        "STAGING_OPENSEARCH_USER",
        "STAGING_OPENSEARCH_PWD",
        "STAGING_OPENAI_API_KEY"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        raise RuntimeError(f"Missing required environment variables: {missing}")

    app.state.claude_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_STAGING_API_KEY"))
    logger.info("Startup complete and clients initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    pass