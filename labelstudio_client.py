LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = ""

from label_studio_sdk import LabelStudio

client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

me = client.users.whoami()
print("username:", me.username)
print("email:", me.email)