def fact_check_google(query):
    API_KEY = "YOUR_GOOGLE_FACT_CHECK_API_KEY"
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# Example usage
# result = fact_check_google("Some news text here")
# print(result)
