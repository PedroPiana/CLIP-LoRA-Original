import torch
import torch.nn.functional as F
from contextlib import nullcontext
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def evaluate_lora(args, clip_model, loader, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        autocast = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        autocast = nullcontext
    clip_model = clip_model.to(device)
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with autocast():
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.to(device), target.to(device)
            with autocast():
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples
    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        autocast = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        autocast = nullcontext
    clip_model = clip_model.to(device)
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(device)
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return
    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    count_iters = 0
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.to(device), target.to(device)
            if args.encoder == 'text' or args.encoder == 'both':
                with autocast():
                    texts_tokenized = clip.tokenize(texts).to(device)
                    class_embeddings = clip_model.encode_text(texts_tokenized)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if args.encoder == 'vision' or args.encoder == 'both':
                with autocast():
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with autocast():
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count_iters += 1
            if count_iters == total_iters:
                break
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return


def lora_inference(args, clip_model, image, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        autocast = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        autocast = nullcontext
    clip_model = clip_model.to(device)
    clip_model.eval()
    best_template = None
    best_classname = None
    best_similarity = -float('inf')
    with torch.no_grad():
        image = image.to(device)
        with autocast():
            image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for template in dataset.template:
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            with autocast():
                texts_tokenized = clip.tokenize(texts).to(device)
                class_embeddings = clip_model.encode_text(texts_tokenized)
            text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            cosine_similarity = (image_features @ text_features.t()).squeeze(0)
            max_sim_idx = cosine_similarity.argmax().item()
            max_sim = cosine_similarity[max_sim_idx].item()
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_template = template
                best_classname = dataset.classnames[max_sim_idx]
    return best_template.format(best_classname.replace('_', ' '))


