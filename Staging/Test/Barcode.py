import os
import requests
from requests.auth import HTTPBasicAuth
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
import openai
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from openai import OpenAI
from google.cloud import aiplatform
import time

aiplatform.init(project="central-muse-388319", location="us-central1")


# Set up GCP and OpenAI configurations
PROJECT_ID = "central-muse-388319"
REGION = "us-central1"
BUCKET = "chatbot-relationships"
DISPLAY_NAME = "4661872127165595648"
DEPLOYED_INDEX_ID = "5511996925575954432"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set environment variables for OpenSearch
OPENAI_API_KEY="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"


# OpenSearch Credentials
OPENSEARCH_USER="admin"
OPENSEARCH_PWD="HealthyAsianKitchen1$3"
OPENSEARCH_HOST="https://search-foodhak-staging-core-ffnbha54vi5fo2hm6vjcjkffpe.eu-west-2.es.amazonaws.com"


# Initialize the OpenAI client
client = OpenAI(
    api_key=("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"),
)

# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

# Initialize the embedding model
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

# Vector Search Setup
vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=REGION,
    gcs_bucket_name=BUCKET,
    index_id=aiplatform.MatchingEngineIndex(DISPLAY_NAME).name,
    endpoint_id=aiplatform.MatchingEngineIndexEndpoint(DEPLOYED_INDEX_ID).name,
    embedding=embedding_model,
    stream_update=True,
)

# Function to fetch user profile from OpenSearch
def get_user_profile(user_id):
    url = f"OPENSEARCH_HOST/user-profiles/_search"
    user = OPENSEARCH_USER
    password = OPENSEARCH_PWD
    
    query = {
        "query": {
            "match": {
                "foodhak_user_id": user_id
            }
        }
    }
    
    response = requests.get(url, auth=HTTPBasicAuth(user, password), json=query)
    
    if response.status_code == 200:
        results = response.json()
        if results['hits']['total']['value'] > 0:
            return results['hits']['hits'][0]['_source']
        else:
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to fetch product details from Open Food Facts API
def get_product_details(barcode):
    url = f"https://world.openfoodfacts.net/api/v2/product/{barcode}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json().get('product', {})
    else:
       print(f"Error fetching product details: {response.status_code}")
    return {}


# Function to parse and extract individual ingredients from the ingredients_text
def extract_ingredients(ingredients_text):
    # Split the ingredients_text by commas and clean up each ingredient
    ingredients = [ingredient.strip().lower() for ingredient in ingredients_text.split(',')]
    return ingredients

# Function to query the vector store for each ingredient individually
def query_vector_store(ingredient_names):
    all_contexts = []
    
    for ingredient in ingredient_names:
        #Formulate the query text for the individual ingredient
        query_text = f"Health impacts and nutritional benefits of {ingredient}."
        
        # Use the vector store to find relevant information based on the query text
        results = vector_store.similarity_search(query_text, k=1)  # Fetch the top 3 relevant results for each ingredient
        context = "\n".join(result.page_content for result in results)
        
        # Collect the context for each ingredient
        all_contexts.append(f"{ingredient.capitalize()}:\n{context}")
    
    # Combine all contexts into a single string
    combined_context = "\n\n".join(all_contexts)
    return combined_context

# Function to extract detailed information from user profile, including goal titles
def extract_user_info(user_profile):
    # Extract user goals and associated recommended and avoided ingredients
    user_health_goals = user_profile.get('user_health_goals', [])
    goals = [goal.get('user_goal', {}).get('title', 'Unknown goal') for goal in user_health_goals]

    # Extract recommended and avoided ingredients from user health goals
    recommended_ingredients = []
    avoided_ingredients = []
    
    for goal in user_health_goals:
        goal_info = goal.get('user_goal', {})
        goal_title = goal_info.get('title', 'Unknown goal')
        
        # Extract ingredients to recommend
        for ingredient in goal.get('ingredients_to_recommend', []):
            recommended_ingredients.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'goal': goal_title,
                'relationship': ingredient.get('relationships', [])
            })
        
        # Extract ingredients to avoid (if available in the data)
        for ingredient in goal.get('ingredients_to_avoid', []):
            avoided_ingredients.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'goal': goal_title,
                'relationship': ingredient.get('relationships', [])
            })
    
    # Extract dietary preferences
    dietary_preferences = user_profile.get('dietary_preferences', 'Not specified')

    # Extract dietary restrictions
    dietary_restrictions = user_profile.get('dietary_restrictions', {}).get('name', 'None')

    # Extract allergens
    allergens = [allergen.get('type', 'Unknown') for allergen in user_profile.get('allergens', [])]

    # Extract diseases and associated recommended and avoided ingredients
    diseases = []
    disease_recommendations = []
    disease_avoidances = []
    
    for disease in user_profile.get('diseases', []):
        disease_info = disease.get('data', {})
        disease_name = disease_info.get('name', 'Unknown disease')
        diseases.append(disease_name)
        
        # Extract ingredients to recommend for diseases
        for ingredient in disease.get('ingredients_to_recommend', []):
            disease_recommendations.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'disease': disease_name,
                'relationship': ingredient.get('relationships', [])
            })
        
        # Extract ingredients to avoid for diseases
        for ingredient in disease.get('ingredients_to_avoid', []):
            disease_avoidances.append({
                'common_name': ingredient.get('common_name', 'Unknown'),
                'disease': disease_name,
                'relationship': ingredient.get('relationships', [])
            })

    # Combine all recommendations and avoidances
    all_recommendations = recommended_ingredients + disease_recommendations
    all_avoidances = avoided_ingredients + disease_avoidances
    
    return goals, dietary_preferences, dietary_restrictions, allergens, diseases, all_recommendations, all_avoidances

def generate_recommendation_summary(user_profile, product_details, vector_store_context):
    # Extract user information, including detailed ingredient recommendations and avoidances
    goals, dietary_preferences, dietary_restrictions, allergens, diseases, all_recommendations, all_avoidances = extract_user_info(user_profile)

    # Create strings for recommendations and avoidances
    recommendations_text = "\n".join([
        f"{rec['common_name']} is recommended for {rec['goal'] or rec['disease']}. {rec['relationship'][0].get('extracts', '')}" 
        for rec in all_recommendations
    ])
    avoidances_text = "\n".join([
        f"{avoid['common_name']} should be avoided for {avoid['goal'] or avoid['disease']}. {avoid['relationship'][0].get('extracts', '')}" 
        for avoid in all_avoidances
    ])

    # **Define product_name and goal**
    product_name = product_details.get('product_name_en', 'the product')
    goal = goals[0] if goals else 'your health goals'

    # Formulate the prompt with a friendly, conversational tone, including a failsafe mechanism
    prompt = f"""
You are Foodhak, the ultimate nutrition sidekick, here to provide personalized, evidence-based advice rooted in scientific research. Your mission is to empower users to make smart, informed food and health choices by providing concise, relevant information. Your tone should be warm and personable, yet professional and credible.

**User Profile:**
- **Name:** {user_profile.get('name', 'Friend')}
- **Age:** {user_profile.get('age', 'Age not specified')}
- **Sex:** {user_profile.get('sex', 'Sex not specified')}
- **Goals:** {', '.join(goals) if goals else 'Not specified.'}
- **Dietary Preferences:** {dietary_preferences or 'None specified'}
- **Dietary Restrictions:** {dietary_restrictions or 'None specified'}
- **Allergens:** {', '.join(allergens) if allergens else 'None reported.'}
- **Health Conditions:** {', '.join(diseases) if diseases else 'None reported'}

**Ingredient Recommendations:**
- **Recommended Ingredients:**
  {recommendations_text or 'No specific ingredients highlighted.'}

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

**Instructions for Your Response:**

Please provide a concise summary for **{product_name}** tailored to the user's goals and preferences. The response should:

- **Be suitable for display on a user interface after the user scans a product.**
- **Be a few lines long, focusing on the most relevant information.**
- **Exclude any sign-offs**
- Include a conclusion on whether the product is suitable for the user.
- Suggest alternatives if the product is not suitable.
- Maintain a warm and supportive tone.

**Tone and Style Guidelines:**

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

"""

    # Call the OpenAI GPT-4 model to generate the Foodhak recommendation
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return chat_completion.choices[0].message.content



# New flow where the vector store context is passed
def main(barcode, user_profile):
    # Step 1: Fetch product details
    product_details = get_product_details(barcode)

    # Step 2: Extract ingredients from the product
    ingredients = extract_ingredients(product_details.get('ingredients_text', ''))

    # Step 3: Query the vector store with the ingredient names to get context
    vector_store_context = query_vector_store(ingredients)

    # Step 4: Generate the recommendation summary
    recommendation = generate_recommendation_summary(user_profile, product_details, vector_store_context)

    # Output the recommendation (for example, print it or send it to your UI)
    print(recommendation)
    
# Example usage
user_id = "b2321fec-1b96-4bef-825f-95b743b9121b"
barcode = "7394376616228"

# Fetch user profile
user_profile = get_user_profile(user_id)

if user_profile:
    main(barcode, user_profile)
else:
    print("User profile not found.")
