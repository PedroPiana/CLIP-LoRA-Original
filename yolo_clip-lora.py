from ultralytics import YOLO
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from loralib.utils import apply_lora, load_lora
from utils import *
from run_utils import *
from lora import lora_inference
import os


args = get_arguments()
set_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP
clip_model, preprocess_clip = clip.load(args.backbone)
clip_model.eval()
clip_model = clip_model.to(device)

# LoRA
list_lora_layers = apply_lora(args, clip_model)
load_lora(args, list_lora_layers)

# Dataset para CLIP-LoRA
print("Preparing dataset.")
dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess_clip)

# Transformação padrão
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# Caminhos
yolo_model_path = r"C:\Users\Pedro\Downloads\PIGS_L_S\runs_laying_standing_bboxes\detect\train\weights\best.pt"
image_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\images"
label_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\labels"

yolo = YOLO(yolo_model_path)

# Métricas
total = 0
acertos = 0

# Função auxiliar: IOU entre duas bboxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Loop de avaliação
for image_name in os.listdir(image_dir):
    if not image_name.endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    # Carrega ground truth do YOLO
    with open(label_path, "r") as f:
        gt_boxes = []
        for line in f:
            cls_id, cx, cy, bw, bh = map(float, line.strip().split())
            gt_boxes.append({
                "cls_id": int(cls_id),
                "bbox": (cx, cy, bw, bh)
            })

    # Carrega imagem
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Predição YOLO
    pred_yolo = yolo.predict(img_path, verbose=False)[0]
    for i, det in enumerate(pred_yolo.boxes.data):
        x1, y1, x2, y2, conf, class_id = det.cpu().numpy()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        # Faz crop da bbox
        crop = image_pil.crop((x1, y1, x2, y2))
        crop_tensor = preprocess(crop).unsqueeze(0).to(device)

        # Encontra melhor bbox da label original (GT)
        best_iou = 0
        gt_cls = None
        for gt in gt_boxes:
            cx, cy, bw, bh = gt["bbox"]
            gx1 = int((cx - bw / 2) * w)
            gy1 = int((cy - bh / 2) * h)
            gx2 = int((cx + bw / 2) * w)
            gy2 = int((cy + bh / 2) * h)
            current_iou = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if current_iou > best_iou:
                best_iou = current_iou
                gt_cls = gt["cls_id"]

        # Pula se nenhuma GT razoável encontrada
        if gt_cls is None or best_iou < 0.3:
            continue

        # CLIP-LoRA
        template = lora_inference(args, clip_model, crop_tensor, dataset)

        total += 1
        pred = False
        if "laying" in template and gt_cls == 0:
            acertos += 1
            pred = True
        elif "standing" in template and gt_cls == 1:
            acertos += 1
            pred = True

        print(f"{image_name} - bbox {i}: GT={gt_cls}, {'✓' if pred else '✗'} (IOU={best_iou:.2f})")

# Resultado final
acc = acertos / total if total > 0 else 0
print(f"\nAcurácia CLIP-LoRA nos crops YOLO: {acertos}/{total} = {acc:.2%}")


#python yolo_clip-lora.py --root_path C:/Users/Pedro/Downloads/DATA --dataset pigs --seed 1 --shots 16 --save_path weights --filename "CLIP-LoRA_pigs" 
