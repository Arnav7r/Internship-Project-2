import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# =====================================================================
# 1. SETUP SYSTEM DEVICE & PREPROCESSING
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GTSRB models typically expect 32x32 images. 
# Change (32, 32) to (224, 224) if your network was trained on larger images.
transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =====================================================================
# 2. LOAD YOUR TRAINED MODEL
# =====================================================================
@st.cache_resource  # Keeps the model in memory so it doesn't reload on every run
def load_my_model():
    # Pulls the network architecture class from your model.py file
    from model import TrafficSignCNN 
    
    # Initialize the structure
    net = TrafficSignCNN()
    
    # Loads your saved weights file (Make sure 'model.pth' is in this folder)
    weights_path = "model.pth" 
    net.load_state_dict(torch.load(weights_path, map_location=device))
    
    net.to(device)
    net.eval()  # Set to evaluation mode for inference
    return net

model = load_my_model()

# =====================================================================
# 3. GTSRB DATASET CLASS NAMES (All 43 Classes)
# =====================================================================
CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing veh over 3.5 tons", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Veh > 3.5 tons prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End speed + passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End no passing veh > 3.5 tons"
}

# =====================================================================
# 4. STREAMLIT USER INTERFACE
# =====================================================================
st.title("🚦 Traffic Sign Recognition")

# Sidebar
st.sidebar.title("Project Info")
st.sidebar.write("Model: Custom CNN")
st.sidebar.write("Dataset: GTSRB")
st.sidebar.write("Classes: 43")
st.sidebar.write("Test Accuracy: 97.28%")
st.sidebar.write("Framework: PyTorch")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image to tensor
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension [1, 3, 32, 32]

    # Model Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    # Get Top 3 predictions
    top_probs, top_classes = torch.topk(probs, k=3)

    predicted_class = top_classes[0][0].item()
    predicted_prob = top_probs[0][0].item()

    # Main Display Results
    st.write("---")
    st.success(f"**Prediction:** {CLASS_NAMES.get(predicted_class, f'Class {predicted_class}')}")
    
    st.metric(label="Confidence", value=f"{predicted_prob * 100:.2f}%")
    st.progress(float(predicted_prob))

    # Detailed Top 3 breakdown
    st.subheader("Top 3 Predictions")
    for i in range(3):
        class_id = top_classes[0][i].item()
        prob = top_probs[0][i].item() * 100
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        st.write(f"**{i+1}.** {class_name} — `{prob:.2f}%`")