import torch
import clip
from PIL import Image

print("Welcome the MyCLIP project.")

device = "cuda" if torch.cuda.is_available() else "cpu"

print('Use device:', device)

model, preprocess = clip.load("ViT-B/32", device=device)

# Chuẩn bị hình ảnh
image = preprocess(Image.open("data/muoi_nih.png")).unsqueeze(0).to(device)

# Chuẩn bị văn bản
text = clip.tokenize(["một con muỗi"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Tính toán điểm tương đồng (cosine similarity)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Khả năng hình ảnh là 'một con muỗi'", probs)

