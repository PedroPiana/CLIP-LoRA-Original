from ultralytics import YOLO
import cv2
from PIL import Image
import os
import json

# Caminhos
yolo_model_path = r"C:\Users\Pedro\Downloads\PIGS_L_S\runs_laying_standing_bboxes\detect\train\weights\best.pt"
image_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\images"
label_dir = r"C:\Users\Pedro\Downloads\PIGS_L_S\laying_standing.v2i.yolov8\valid\labels"

# Inicializa modelo
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

# Loop pelas imagens
for image_name in os.listdir(image_dir):
    if not image_name.endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    # Carrega GT
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

    # Predição YOLO
    pred_yolo = yolo.predict(img_path, verbose=False)[0]

    matched_gt = [False] * len(gt_boxes)  # controle de matching

    for i, det in enumerate(pred_yolo.boxes.data):
        x1, y1, x2, y2, conf, pred_cls = det.cpu().numpy()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        best_iou = 0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_boxes):
            if matched_gt[idx]:
                continue
            cx, cy, bw, bh = gt["bbox"]
            gx1 = int((cx - bw / 2) * w)
            gy1 = int((cy - bh / 2) * h)
            gx2 = int((cx + bw / 2) * w)
            gy2 = int((cy + bh / 2) * h)
            current_iou = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx

        if best_gt_idx != -1 and best_iou >= 0.3:
            matched_gt[best_gt_idx] = True
            gt_cls = gt_boxes[best_gt_idx]["cls_id"]
            total += 1
            if int(pred_cls) == gt_cls:
                acertos += 1
                print(f"{image_name} - bbox {i}: GT={gt_cls}, Pred={int(pred_cls)} ✓ (IOU={best_iou:.2f})")
            else:
                print(f"{image_name} - bbox {i}: GT={gt_cls}, Pred={int(pred_cls)} ✗ (IOU={best_iou:.2f})")

# Resultado final
acc = acertos / total if total > 0 else 0
print(f"\nAcurácia YOLO: {acertos}/{total} = {acc:.2%}")

# Salvar resultado em JSON
result_path = "acuracias_yolo.json"
try:
    with open(result_path, "r") as f:
        results = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    results = {}

results["YOLO"] = {
    "acertos": acertos,
    "total": total,
    "acuracia": acc
}

with open(result_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Acurácia salva em {result_path}")
