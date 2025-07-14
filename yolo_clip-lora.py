from ultralytics import YOLO
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from PIL import Image
from loralib.utils import apply_lora, load_lora
from utils import *
from run_utils import *
from lora import lora_inference
import os
        

args = get_arguments()


set_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP
clip_model, preprocess = clip.load(args.backbone)
clip_model.eval()
clip_model = clip_model.to(device)

# LoRA
list_lora_layers = apply_lora(args, clip_model)
load_lora(args, list_lora_layers)

# Prepare dataset
print("Preparing dataset.")
dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)

# Usa o mesmo preprocess do CLIP
preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

# carregar modelo YOLO e CLIP
yolo_model_path = r"C:\Users\Pedro\Downloads\PIGS_L_S\runs_laying_standing_bboxes\detect\train\weights\best.pt"
image_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\images"
label_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\labels"


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


        template = lora_inference(args, clip_model, crop_tensor, dataset)

        total += 1
        pred = False
        if "laying" in template and cls_id == 0:
            acertos += 1
            pred = True
        elif "standing" in template and cls_id == 1:
            acertos += 1
            pred = True


        print(f"{image_name} - bbox {i}: GT={cls_id}, {'✓' if pred else '✗'}")

# imprimir acurácia final
acc = acertos / total if total > 0 else 0
print(f"\nAcurácia CLIP-LoRA nos crops GT: {acertos}/{total} = {acc:.2%}")


#python yolo_clip-lora.py --root_path C:/Users/Pedro/Downloads/DATA --dataset pigs --seed 1 --shots 16 --save_path weights --filename "CLIP-LoRA_pigs" 
