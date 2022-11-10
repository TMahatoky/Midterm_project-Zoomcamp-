import requests

url = "http://localhost:9696/predict"

offer = {
    "location": "US", 
    "telecommuting": 1, 
    "has_company_logo": 0, 
    "has_questions": 0, 
    "employment_type": "Full-time"
}

response = requests.post(url, json = offer).json()
print(response)

if response["fraud"] == True:
    print("Advised to stay away from offer and company")
else: 
    print("You are safe to apply")