import requests
from requests.auth import HTTPBasicAuth
import json

# Set environment variables
OPENSEARCH_USER = "admin"
OPENSEARCH_PWD = "HealthyAsianKitchen1$3"
OPENSEARCH_HOST = "https://search-foodhak-staging-core-ffnbha54vi5fo2hm6vjcjkffpe.eu-west-2.es.amazonaws.com"

# The user ID you want to search for
user_id = "b2321fec-1b96-4bef-825f-95b743b9121b"

def query_user_profile(user_id):
    url = f"{OPENSEARCH_HOST}/user-profiles/_search"

    query = {
        "query": {
            "match": {
                "foodhak_user_id": user_id
            }
        }
    }

    # Send the query to OpenSearch
    response = requests.post(url, auth=HTTPBasicAuth(OPENSEARCH_USER, OPENSEARCH_PWD), json=query)

    # Print status and response body
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        response_data = response.json()
        print(response_data['hits']['hits'][0]['_source'])

        #print(f"Response Body:\n{json.dumps(response_data, indent=2)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Run the query
if __name__ == "__main__":
    query_user_profile(user_id)
