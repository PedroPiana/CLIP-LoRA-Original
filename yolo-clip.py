from ultralytics import YOLO
import torch
import clip
import os
import cv2
from PIL import Image
import numpy as np
import json

# carregar modelo YOLO e CLIP
yolo_model_path = r"C:\Users\Pedro\Downloads\PIGS_L_S\runs_laying_standing_bboxes\detect\train\weights\best.pt"
image_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\images"
label_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\labels"
clip_classes = ["a photo of a pig standing", "a photo of a pig laying"]  # 0 = standing, 1 = laying

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(clip_classes).to(device)

# métricas
total = 0
acertos = 0

# loop nas imagens
for image_name in os.listdir(image_dir):
    if not image_name.endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls_id, cx, cy, bw, bh = map(float, line.strip().split())
        cls_id = int(cls_id)

        # converter bbox YOLO para coordenadas em pixel
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image_pil.crop((x1, y1, x2, y2))
        crop_tensor = preprocess(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(crop_tensor)
            text_features = clip_model.encode_text(text_tokens)
            logits = (image_features @ text_features.T).softmax(dim=-1)
            pred = int(logits.argmax().item())

        total += 1
        if pred == cls_id:
            acertos += 1

        print(f"{image_name} - bbox {i}: GT={cls_id}, Pred={pred}, {'✓' if pred==cls_id else 'X'}")

# imprimir acurácia final
acc = acertos / total if total > 0 else 0
print(f"\nAcurácia CLIP nos crops GT: {acertos}/{total} = {acc:.2%}")

# Salvar resultado em JSON (acumula resultados de diferentes scripts)
result_path = "acuracias_clip_lora.json"
try:
    with open(result_path, "r") as f:
        results = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    results = {}

# Gera chave única para cada script/execução
key = f"CLIP_crops_YOLO"
results[key] = {
    "acertos": acertos,
    "total": total,
    "acuracia": acc
}

with open(result_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Acurácia salva em {result_path}")

#python yolo-clip.py 
