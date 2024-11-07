```markdown
# Barcode-Recommend
This repo contains the scripts for Barcode-Recommend staging and production environments.
## Overview

The **Foodhak-Barcode-Recommend** service provides personalized nutritional recommendations based on a user's health goals and the ingredients of a specific product. It leverages user profiles from OpenSearch and product information from the OpenFoodFacts API, along with vector search for additional ingredient context. The API returns a comprehensive recommendation with nutritional advice and health impacts tailored to the user's health goals.

## API Endpoint

### `POST /barcode-recommend`

This endpoint receives a user ID and product barcode, fetches relevant user profile information, extracts product ingredients, queries a vector store for context about the health impacts of each ingredient, and generates personalized nutritional recommendations.

### Request Format

- **Method**: `POST`
- **URL**: `https://www.staging-foodhakai.com/barcode-recommend`
- **Headers**:
    - `Content-Type: application/json`
    - `Authorization`: Bearer token required in the `Authorization` header for authentication.
- **Body** (JSON):
    
    ```json
    {
      "user_id": "b2321fec-1b96-4bef-825f-95b743b9121b",
      "barcode": "80177173"
    }
    ```
    
### Example CURL Request

```bash
curl -X POST https://www.staging-foodhakai.com/barcode-recommend \
-H "Content-Type: application/json" \
-H "Authorization: Bearer mS6WabEO.1Qj6ONyvNvHXkWdbWLFi9mLMgHFVV4m7" \
-d '{
      "user_id": "b2321fec-1b96-4bef-825f-95b743b9121b",
      "barcode": "80177173"
   }'
```

### Parameters

- **user_id**: (Required) The unique ID of the user whose health profile will be fetched.
- **barcode**: (Required) The barcode of the product to analyze.

### Response Format

The API returns a JSON object containing personalized nutritional recommendations based on the user's health goals and the product's ingredient analysis.

### Example Response

```json
{
  "recommendation": "### Nutritional Advice\n\n#### Goals:\n1. **Healthy Diet**\n2. **Improve Blood Pressure**\n3. **Improve Gut Health**\n4. **Improve Energy Levels**\n5. **Build Muscle**\n6. **Improve Skin**\n\n#### Product Analysis:\n- **Product Name**: N/A\n- **Ingredients**:\n  - Sugar\n  - Palm oil\n  - Hazelnuts (13%)\n  - Skimmed milk powder (8.7%)\n  - Fat-reduced cocoa (7.4%)\n  - Emulsifier: lecithins (soya)\n  - Vanillin\n\n#### Ingredient Impacts:\n1. **Sugar**:\n   - **Pros**: Provides immediate energy.\n   - **Cons**: Associated with obesity, diabetes, and cardiovascular diseases. Regular consumption can negatively impact blood pressure and skin health.\n\n2. **Palm oil**:\n   - **Pros**: Enhances cognitive function.\n   - **Cons**: Generally high in saturated fats, which can adversely affect blood pressure.\n\n3. **Hazelnuts**:\n   - **Pros**: Improves cerebrovascular function, supports heart health.\n   - **Cons**: High in calories, which may not be ideal for certain weight management goals.\n\n4. **Skimmed milk powder**:\n   - **Pros**: Improves muscle functionality, beneficial for building muscle.\n   - **Cons**: Some people may be lactose intolerant.\n\n5. **Fat-reduced cocoa**:\n   - **Pros**: Lowers LDL (bad) cholesterol, beneficial for cardiovascular health.\n   - **Cons**: Often contains added sugars.\n\n6. **Emulsifier: lecithins (soya)**:\n   - **Pros**: Increases HDL (good) cholesterol, beneficial for heart health.\n   - **Cons**: Some individuals may have soy allergies.\n\n7. **Vanillin**:\n   - **Pros**: Contains cocoa flavanols which improve cardiometabolic biomarkers.\n   - **Cons**: Can be an artificial flavoring, with limited health benefits compared to pure vanilla.\n\n### Recommendations:\n\n#### 1. Healthy Diet:\n- **Limit Sugar** intake to avoid obesity, diabetes, and cardiovascular diseases.\n- **Incorporate Whole Foods**: Focus on fruits, vegetables, lean proteins, and whole grains.\n- **Moderate Nut and Dairy Intake**: While hazelnuts and milk powder have benefits, control portions to avoid excess calories.\n\n#### 2. Improve Blood Pressure:\n- **Reduce Saturated Fats**: Palm oil can be high in saturated fats, which may raise blood pressure.\n- **Increase Fiber**: Include foods like oats, beans, and vegetables which help improve blood pressure.\n\n#### 3. Improve Gut Health:\n- **Probiotics and Fiber**: Include yogurt, fermented foods, and high-fiber foods to support gut health.\n- **Limit Processed Foods**: High sugar content can negatively affect gut bacteria.\n\n#### 4. Improve Energy Levels:\n- **Balanced Meals**: Combine carbohydrates, protein, and healthy fats.\n- **Complex Carbohydrates**: Opt for whole grains and vegetables over sugar for sustained energy.\n\n#### 5. Build Muscle:\n- **Protein Intake**: Lean meats, dairy (like skimmed milk powder), beans, and legumes are crucial.\n- **Strength Training**: Combine a high-protein diet with resistance exercises to build muscle.\n\n#### 6. Improve Skin:\n- **Antioxidants**: Include foods rich in vitamins A, C, and E such as berries, nuts, seeds, and green leafy vegetables.\n- **Hydration**: Ample water intake for hydration, along with a diet low in sugar and processed foods to avoid skin inflammation.\n\n### Conclusion:\nThis product has a mixed nutritional profile with beneficial and potentially harmful ingredients. You should limit your intake of this product, particularly due to its sugar and palm oil content. Balance it with a diet rich in fruits, vegetables, lean proteins, and whole grains to achieve your comprehensive health goals. Make dietary choices that support heart health, muscle building, gut health, and skin vitality."
}
```

## Flow Overview

1. **Client Request**:
    - The client sends a `POST` request to the `/barcode-recommend` endpoint with the user's `user_id` and the product's `barcode` in the request body.
    - The request must include a valid Bearer token in the `Authorization` header.
2. **Authentication**:
    - The server validates the API key from the `Authorization` header. If the token is invalid or missing, a `401 Unauthorized` response is returned.
3. **Fetching User Profile**:
    - The server retrieves the user's profile from the OpenSearch database using the `user_id`. This profile contains the user's health goals and dietary preferences.
4. **Fetching Product Details**:
    - The product details (e.g., ingredients) are fetched from the OpenFoodFacts API using the product's barcode.
5. **Extracting Ingredients**:
    - The `ingredients_text` from the product details is parsed to extract individual ingredient names.
6. **Querying Vector Store**:
    - The extracted ingredient names are used to query the Vertex AI-based vector store, which provides context on the health impacts of each ingredient. This context is used to generate more specific nutritional advice.
7. **Generating Recommendations**:
    - A personalized recommendation is generated using the user's health goals and the ingredient context. The recommendation includes advice on how the product aligns with the user's health objectives.
8. **Response to Client**:
    - The generated recommendation is returned to the client in the form of a JSON response.

## Example CURL requests for staging and production Barcode-Recommend

### Staging-Barcode-Recommend

```bash
curl -X POST https://www.staging-foodhakai.com/barcode-recommend \
-H "Content-Type: application/json" \
-H "Authorization: Bearer mS6WabEO.1Qj6ONyvNvHXkWdbWLFi9mLMgHFVV4m7" \
-d '{
  "user_id": "b2321fec-1b96-4bef-825f-95b743b9121b",
  "barcode": "80177173"
}'
```

### Production-Barcode-Recommend

```bash
curl -X POST https://www.foodhakai.com/barcode-recommend \
-H "Content-Type: application/json" \
-H "Authorization: Bearer viJ8u142.NaQl7JEW5u8bEJpqnnRuvilTfDbHyWty" \
-d '{
  "user_id": "9acc25b6-b238-407e-bc85-44d723bf4551",
  "barcode": "80177173"
}'
```

## Conclusion

The **Foodhak-Barcode-Recommend** API provides personalized, health-based recommendations for food products based on user-specific goals and product ingredients. It combines user profile data, product details, and AI-powered ingredient analysis to generate detailed nutritional advice, helping users make informed dietary choices.
```

This README.md provides a clear, concise guide that covers everything from how to make requests to the service, to understanding the responses it generates. This should be quite helpful for developers integrating this service into their systems or for users interested in understanding how to interact with it.
