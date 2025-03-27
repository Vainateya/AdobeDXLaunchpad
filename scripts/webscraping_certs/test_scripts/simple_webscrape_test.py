import requests

# Define the URL
url = "https://certification.adobe.com/auth/oauth-redirect"

# Define headers based on your browser's request details
headers = {
}

# Make the GET request
response = requests.get(url, headers=headers)

# Print the response HTML
if response.status_code == 200:
    print("Page content retrieved successfully!")
    print(response.text) # This contains the full HTML of the page
else:
    print(f"Failed to retrieve content. Status code: {response.status_code}")