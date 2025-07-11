import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from PIL import Image


from utils import *
from run_utils import *
from lora import run_lora_inference, lora_inference


def inference(args, image):

    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
        
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    

    template = run_lora_inference(args, clip_model, image, dataset)

    print(f"Best template for the image: {template}")


args = get_arguments()

img_path = args.image_path
image = Image.open(img_path).convert("RGB")

# Use o mesmo preprocess do CLIP
preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

image_tensor = preprocess(image).unsqueeze(0)  # Adiciona batch dimension

# Rode a inferência
inference(args, image_tensor)

'''
to run inference:

python inference.py \
  --root_path C:/Users/Pedro/Downloads/DATA \
  --dataset pigs \
  --seed 1 \
  --shots 1
  --save_path weights \
  --image_path "C:/Users/Pedro/Downloads/CLIP-LoRA-Original/cropped_image_6359.jpg" \
  --filename "CLIP-LoRA_PIGS"

python inference.py --root_path C:/Users/Pedro/Downloads/DATA --dataset pigs --seed 1 --shots 1 --save_path weights --image_path "C:/Users/Pedro/Downloads/CLIP-LoRA-Original/cropped_image_6359.jpg" --filename "CLIP-LoRA_PIGS" 
'''

