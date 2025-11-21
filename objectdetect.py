import requests
import os
from dotenv import load_dotenv
import cv2

# Load environment variables
load_dotenv()
endpoint = os.getenv("AZURE_VISION_ENDPOINT")
key = os.getenv("AZURE_VISION_KEY")

# Image path
image_path = "img.jpg"

# Read image data
with open(image_path, "rb") as f:
    img_data = f.read()

# Azure Computer Vision request
url = endpoint + "/vision/v3.2/analyze?visualFeatures=Objects"
headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}
response = requests.post(url, headers=headers, data=img_data)
result = response.json()

# Load image using OpenCV
image = cv2.imread(image_path)

# Draw bounding boxes for objects with confidence > 60%
for obj in result.get("objects", []):
    confidence = obj.get("confidence", 0)
    if confidence >= 0.4:
        rect = obj["rectangle"]
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Prepare label: object + confidence
        label = f"{obj['object']} {confidence*100:.1f}%"
        # Add parent object if available
        if "parent" in obj:
            parent = obj["parent"].get("object", "")
            label += f" ({parent})"

        # Draw label
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display result
output_path = "output.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Output saved to {output_path}")
