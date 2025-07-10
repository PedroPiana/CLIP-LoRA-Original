import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import apply_lora, load_lora
from loralib import layers as lora_layers


def lora_inference(args, clip_model, image, dataset):
    """
    Dado um tensor de imagem, retorna o template que gera maior similaridade para essa imagem.
    """
    clip_model.eval()
    best_template = None
    best_similarity = -float('inf')
    best_idx = -1
    with torch.no_grad():
        # Normaliza a imagem e move para cuda
        image = image.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Testa todos os templates
        for idx, template in enumerate(dataset.template):
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts_tokenized = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts_tokenized)
            text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            # Similaridade da imagem com todas as classes desse template
            cosine_similarity = (image_features @ text_features.t()).squeeze(0)
            max_sim = cosine_similarity.max().item()
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_template = template
                best_idx = idx
    return best_template


def run_lora_inference(args, clip_model, image, dataset):
    """
    Recebe uma imagem, roda o lora_inference e exibe o melhor template para essa imagem.
    """
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    load_lora(args, list_lora_layers)
    best_template = lora_inference(args, clip_model, image, dataset)
    print(f"Melhor template para a imagem: {best_template}")
    return best_template




