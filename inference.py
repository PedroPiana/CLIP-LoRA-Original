import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from PIL import Image
from loralib.utils import apply_lora, load_lora
from utils import *
from run_utils import *
from lora import lora_inference
        

args = get_arguments()

img_path = args.image_path
image = Image.open(img_path).convert("RGB")

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

image_tensor = preprocess(image).unsqueeze(0)  # Adiciona batch dimension



template = lora_inference(args, clip_model, image_tensor, dataset)
print(f"Best template for the image: {template}")



#python inference.py --root_path C:/Users/Pedro/Downloads/DATA --dataset pigs --seed 1 --shots 1 --save_path weights --image_path "C:/Users/Pedro/Downloads/CLIP-LoRA-Original/cropped_image_6359.jpg" --filename "CLIP-LoRA_pigs" 

