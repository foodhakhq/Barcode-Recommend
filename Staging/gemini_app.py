import os
import re
import json
import requests
from requests.auth import HTTPBasicAuth
import asyncio
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()
load_dotenv()

# Set up environment variables and configurations
PROJECT_ID = "central-muse-388319"
REGION = "us-central1"
BUCKET = "chatbot-relationships"
DISPLAY_NAME = "4661872127165595648"
DEPLOYED_INDEX_ID = "5511996925575954432"
STAGING_OPENAI_API_KEY = os.getenv("STAGING_OPENAI_API_KEY")
STAGING_API_KEY = os.getenv("STAGING_API_KEY")

# Initialize Google AI platform
aiplatform.init(project=PROJECT_ID, location=REGION)
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=REGION,
    gcs_bucket_name=BUCKET,
    index_id=aiplatform.MatchingEngineIndex(DISPLAY_NAME).name,
    endpoint_id=aiplatform.MatchingEngineIndexEndpoint(DEPLOYED_INDEX_ID).name,
    embedding=embedding_model,
    stream_update=True,
)

# Bearer token dependency
def verify_token(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=500, detail="Authorization header missing or invalid")
    
    token = authorization.split(" ")[1]
    if token != STAGING_API_KEY:
        raise HTTPException(status_code=500, detail="Invalid token")

# Define request body schema
class RecommendationRequest(BaseModel):
    user_id: str
    barcode: str

# Function to get user profile (Sync)
def get_user_profile(user_id):
    url = f"{os.getenv('STAGING_OPENSEARCH_HOST')}/user-profiles/_search"
    user = os.getenv("STAGING_OPENSEARCH_USER")
    password = os.getenv("STAGING_OPENSEARCH_PWD")

    query = {"query": {"match": {"foodhak_user_id": user_id}}}

    response = requests.get(url, auth=HTTPBasicAuth(user, password), json=query)

    if response.status_code == 200:
        results = response.json()
        if results['hits']['total']['value'] > 0:
            return results['hits']['hits'][0]['_source']
        else:
            return None
    else:
        raise HTTPException(status_code=500, detail=f"Error fetching user profile: {response.status_code}")

# Function to get product details (Sync)
def get_product_details(barcode):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('product', {})
    else:
        raise HTTPException(status_code=500, detail=f"Error fetching product details: {response.status_code}")

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
    # Remove duplicates
    ingredient_names = list(set(ingredient_names))
    print(ingredient_names)

    # Prepare all query texts
    query_texts = [f"Health impacts and nutritional benefits of {ingredient}." for ingredient in ingredient_names]

    async def fetch_context(ingredient, query_text):
        # Use the asynchronous similarity search method
        results = await vector_store.asimilarity_search(query_text, k=1)
        context = "\n".join(result.page_content for result in results)
        return f"{ingredient.capitalize()}:\n{context}"

    # Create a list of tasks for concurrent execution
    tasks = [fetch_context(ingredient, query_text) for ingredient, query_text in zip(ingredient_names, query_texts)]

    # Run the tasks concurrently and collect results
    all_contexts = await asyncio.gather(*tasks)

    # Combine all contexts into a single string
    combined_context = "\n\n".join(all_contexts)
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
    return goals, dietary_preferences, dietary_restrictions, allergens, diseases, all_recommendations, all_avoidances

# Function to initialize the Gemini model
def initialize_gemini_model(project_id, location, system_instruction):
    vertexai.init(project=project_id, location=location)
    return GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=[system_instruction]
    )

# Function to generate response with Gemini model
def generate_response_with_gemini(model, prompt):
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0,
                "top_p": 1,
            }
        )
        response_content = response.text.strip()
        return response_content.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

# Function to generate recommendation summary using Gemini
def generate_recommendation_summary(user_profile, product_details, vector_store_context):
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
    
    product_name = product_name if product_name else next((name for name in fallback_product_names if name), "Unnamed")

    goals, dietary_preferences, dietary_restrictions, allergens, diseases, all_recommendations, all_avoidances = extract_user_info(user_profile)

    recommendations_text = "\n".join([
        f"{rec['common_name']} is recommended for {rec.get('goal', rec.get('disease', 'Unknown'))}. "
        f"{rec['relationship'][0].get('extracts', 'No relationship data available') if rec.get('relationship') and len(rec['relationship']) > 0 else 'No relationship data available'}"
        for rec in all_recommendations if rec.get('common_name')  # Ensure common_name is present
    ])
    avoidances_text = "\n".join([
        f"{avoid['common_name']} should be avoided for {avoid.get('goal', avoid.get('disease', 'Unknown'))}. "
        f"{avoid['relationship'][0].get('extracts', 'No relationship data available') if avoid.get('relationship') and len(avoid['relationship']) > 0 else 'No relationship data available'}"
        for avoid in all_avoidances if avoid.get('common_name')  # Ensure common_name is present
    ])
    # Define system instructions
    system_instruction = f"""
    You are Foodhak, the ultimate nutrition sidekick, here to provide personalized, evidence-based advice on all things nutrition. 
    Your mission is to be approachable, supportive, and insightful while helping users make informed food and health choices. 
    Remember to be engaging, motivational, and ensure the advice feels like a friendly conversation with a trusted advisor.

    **User Profile:**
    - **Name:** {user_profile.get('name', 'Friend')}
    - **Age:** {user_profile.get('age', 'Age not specified')}
    - **Sex:** {user_profile.get('sex', 'Sex not specified')}
    - **Goals:** {', '.join(goals) if goals else 'Not specified.'}
    - **Dietary Preferences:** {dietary_preferences or 'None specified'}
    - **Dietary Restrictions:** {dietary_restrictions or 'None specified'}
    - **Allergens:** {', '.join(allergens) if allergens else 'None reported.'}
    - **Health Conditions:** {', '.join(diseases) if diseases else 'None reported'}
    
    Tone and Style Guidelines:

    - **Concise and Clear:**
        - Keep the response brief and focused.
        - Use simple language and avoid jargon.

    - **Warm and Supportive:**
        - Use friendly and encouraging language.
        - Support the user's health journey.

    - **Personalized:**
        - Tailor the advice to the user's specific goals and preferences.

    - **Professional:**
        - Do not include greetings, sign-offs, or personal anecdotes.

    - **Transparency and Honesty:**
        - Acknowledge if certain information is unavailable.
        - Do not speculate or provide unsure information.

    **Remember:**
    Your goal is to empower the user with a short, personalized summary that helps them feel supported and confident in their food choices. The summary should be suitable for display on a user interface, avoiding any unnecessary content.

    ### Response Format (Generate a detailed response with json nested structures under each key):

    {{
    "response": "Please provide a concise summary for **{product_name}** tailored to the user's goals and preferences. 
                                The response should:
                                - Be suitable for display on a user interface after the user scans a product.
                                - Be a few lines long, focusing on the most relevant information.
                                - Exclude any sign-offs.
                                - Include a conclusion on whether the product is suitable for the user and why.
                                - Suggest alternatives if the product is not suitable.
                                - Maintain a warm and supportive tone."
       
                
    }}

    """    
    # Initialize the Gemini model
    model = initialize_gemini_model(PROJECT_ID, REGION, system_instruction)

    # Formulate the prompt
    prompt = f"""
    **Ingredient Recommendations:**
    - **Recommended Ingredients:** 
      {recommendations_text or 'No specific ingredients are highlighted.'}

    - **Ingredients to Avoid:**
      {avoidances_text or 'No specific ingredients to avoid.'}
      
    **Additional Context from Ingredient Data:**
      {vector_store_context or 'No additional context available from ingredient data.'}
    
    **Product Analysis:**
    - **Product Name:** {product_name}
    - **Nutritional Overview:** {product_details.get('nutriments', 'Detailed nutritional information is unavailable.')}
    - **Key Ingredients:** {product_details.get('ingredients_text', 'Ingredients not specified.')}
    - **Nutri-Score:** {product_details.get('nutriscore_grade', 'Not available')}
    - **Labels and Certifications:** {product_details.get('labels', 'No specific labels found.')}
    """
    return generate_response_with_gemini(model, prompt)

# Endpoint for barcode recommendations
@app.post("/barcode-recommend")
async def barcode_recommend(request: RecommendationRequest, token_valid: str = Depends(verify_token)):
    # Fetch user profile
    user_profile = get_user_profile(request.user_id)

    if not user_profile:
        raise HTTPException(status_code=500, detail="User profile not found")

    # Fetch product details
    product_details = get_product_details(request.barcode)

    if not product_details:
        raise HTTPException(
            status_code=404,
            detail="Product details not found."
        )

    # Extract ingredients
    ingredients_text = product_details.get('ingredients_text', '')
    ingredient_names = extract_ingredients(ingredients_text) if ingredients_text else []

    # Initialize recommendation data
    recommendation_data = {}

    # Retry logic with a maximum of 3 attempts
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            # Attempt to generate the recommendation even if ingredients are missing
            vector_store_context = await query_vector_store_async(ingredient_names) if ingredient_names else None

            # Generate recommendation summary using the available data
            recommendation = generate_recommendation_summary(user_profile, product_details, vector_store_context)

            # Attempt to parse the recommendation as JSON
            recommendation_data = json.loads(recommendation)
            break  # Exit loop if recommendation is successfully generated

        except json.JSONDecodeError as e:
            # If JSON parsing fails after all retries, raise an HTTPException
            if attempt >= max_attempts - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse the recommendation JSON after {max_attempts} attempts."
                )
            else:
                print(f"JSON decoding failed on attempt {attempt + 1}, retrying...")

        except Exception as e:
            # For any other errors, retry up to max_attempts
            if attempt >= max_attempts - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate the recommendation after {max_attempts} attempts."
                )
            else:
                print(f"Error generating recommendation on attempt {attempt + 1}, retrying...")

        await asyncio.sleep(1)  # Wait before retrying
        attempt += 1

    if not ingredient_names:
        recommendation_data["error"] = {
            "code": "FS-204",
            "message": "Insufficient Data",
            "description": "Not enough nutritional data is available to calculate a reliable health score for this product."
        }

    return recommendation_data
