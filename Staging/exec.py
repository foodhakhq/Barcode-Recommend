import subprocess
import time

# Define the curl command
curl_command = '''
curl -X POST https://www.staging-foodhakai.com/barcode-recommend \
-H "Content-Type: application/json" \
-H "Authorization: Bearer mS6WabEO.1Qj6ONyvNvHXkWdbWLFi9mLMgHFVV4m7" \
-d '{
  "user_id": "b2321fec-1b96-4bef-825f-95b743b9121b",
  "barcode": "80177173"
}'
'''

# Record the start time
start_time = time.time()

# Execute the curl command using subprocess
subprocess.run(curl_command, shell=True)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
