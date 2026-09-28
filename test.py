import requests
from PIL import Image
from io import BytesIO

# -----------------------------
# Your local FastAPI endpoint
# -----------------------------
API_URL = "http://127.0.0.1:8000/api/generate-image"  # Replace if your FastAPI runs on a different port

# -----------------------------
# Path to your images
# -----------------------------
person_path = "D:\\bot--for-outfit\\download (3).png"   # Person/model image
cloth_path = "D:\\bot--for-outfit\\download (4).png"           # Shirt image

# -----------------------------
# Prompt and settings
# -----------------------------
payload = {
    "user_id": "user123",
    "prompt": "A realistic photo of the person wearing the shirt shown in the reference image, natural lighting, detailed fabric",
    "negative_prompt": "blurry, bad quality, distorted",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
}

# -----------------------------
# Send POST request with images
# -----------------------------
with open(person_path, "rb") as f1, open(cloth_path, "rb") as f2:
    files_data = {
        "person_image": f1,
        "cloth_image": f2
    }
    response = requests.post(API_URL, data=payload, files=files_data)

# -----------------------------
# Handle Response
# -----------------------------
if response.status_code == 200:
    # Assuming FastAPI returns raw image
    result_image = Image.open(BytesIO(response.content))
    result_image.save("tryon_output.png")
    result_image.show()  # This opens the image in your default image viewer
    print(" Try-on image generated successfully!")
else:
    print(f" Failed: {response.status_code}")
    print(response.text)