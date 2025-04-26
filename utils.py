import torch

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import os
import glob
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def modcrop(image, scale):
    width, height = image.size

    new_width = width - (width % scale)
    new_height = height - (height % scale)

    cropped_image = image.crop((0, 0, new_width, new_height))
    
    return cropped_image

def get_name(path, is_lr):
    name = os.path.basename(path).split(".")[0]
    if is_lr:
        return name[:-2]
    else:
        return name

def get_paths(path):
    file_types = ['*.png', '*.bmp']

    paths = []

    for file_type in file_types:
        pattern = os.path.join(path, file_type)
        paths.extend(glob.glob(pattern))

    paths = list(map(lambda x: x.replace('\\', '/'), paths))

    if len(paths) > 0:
        is_lr = True if "LR" in paths[0] else False
        return sorted(paths, key = lambda x:get_name(x, is_lr))
    else:
        return []

def generate_lr(hr_path, scale):
    hr_paths = get_paths(hr_path)

    lr_path = hr_path.replace("HR", "LR") + "_bicubic/X" + str(scale)

    if not os.path.isdir(lr_path):
        os.makedirs(lr_path)

    for path in hr_paths:
        hr_img_name = path.split('/')[-1]
        lr_img_name = hr_img_name.replace(".", "x" + str(scale) + ".")
        hr_img = Image.open(path).convert('RGB')
        hr_img = modcrop(hr_img, scale)
        w,h = hr_img.size
        lr_img = hr_img.resize((w//scale, h//scale), Image.BICUBIC)
        lr_img.save(lr_path+'/'+lr_img_name)
    print(f"LR images saved to {lr_path+'/'}")

def save(state_dict, model_name, epoch, trained_path, best=False):
    if not os.path.exists(trained_path):
        os.makedirs(trained_path)

    if best:
        torch.save(state_dict, f"{trained_path}/{model_name.lower()}_best_{epoch}.pt")
    else:
        torch.save(state_dict, f"{trained_path}/{model_name.lower()}_{epoch}.pt")

def tensor_to_image(tensor):
    tensor = tensor.numpy()
    if np.ndim(tensor) == 3:
        image_array = np.transpose(tensor, (1, 2, 0)) * 255
    else:
        image_array = tensor * 255

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image_array.squeeze())

def show_images(images, labels, figsize=(12, 7)):
    fig, axes = plt.subplots(2, len(images)//2, figsize=figsize)
    axes = axes.flatten()
    for i in range(len(images)):
        axes[i].imshow(images[i])
        if i < len(images)//2:
            axes[i].set_title(labels[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def prep_showcase(model, lr_tensor, gt_tensor, scale, chop_size=60000):
    if model.get_name() == "ESRT":
        hr_tensor = model.process_image(lr_tensor, scale, chop_size)
    else:
        hr_tensor = model.process_image(lr_tensor)

    lr_img = tensor_to_image(lr_tensor)
    gt_img = tensor_to_image(gt_tensor)
    hr_img = tensor_to_image(hr_tensor)

    w,h = hr_img.size
    wl,hl = lr_img.size

    if abs(lr_img.size[0] - gt_img.size[0]) > 10 and \
       abs(lr_img.size[0] - gt_img.size[0]) > 10:
        bic_img = lr_img.resize((wl * scale, hl * scale), Image.BICUBIC)
    else:
        bic_img = lr_img

    gt_img = gt_img.crop((0,0,w,h))
    bic_img = bic_img.crop((0,0,w,h))

    return [gt_img, bic_img, hr_img]

def get_avg_metrics(model, dataset, scale, chop_size=60000):
    psnr_sum_up = 0
    ssim_sum_up = 0
    psnr_sum_bic = 0
    ssim_sum_bic = 0

    results_path = f"./Results/{dataset.get_name()}/{model.get_name()}/X{str(scale)}"
    img_names = list(map(lambda x: x.split('/')[-1].replace("_GT", "").replace(".", f"x{str(scale)}."), dataset.get_paths()[0]))

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    for i, [lr, gt] in enumerate(dataset):
        gt_img, bic_img, hr_img = prep_showcase(model, lr, gt, scale, chop_size)

        hr_img.save(results_path + "/" + img_names[i])

        psnr_sum_bic += compute_psnr(np.array(gt_img), np.array(bic_img))
        psnr_sum_up += compute_psnr(np.array(gt_img), np.array(hr_img))
        ssim_sum_bic += compute_ssim(np.array(gt_img), np.array(bic_img), channel_axis=2)
        ssim_sum_up += compute_ssim(np.array(gt_img), np.array(hr_img), channel_axis=2)
    
    print(f"Upscaled images saved to {results_path}")

    return psnr_sum_bic/len(dataset), ssim_sum_bic/len(dataset), psnr_sum_up/len(dataset), ssim_sum_up/len(dataset)

def easy_crop(img, relative_pos=(50, 50), crop_size=64):
    w_percent = max(min(100, relative_pos[0]), 0) / 100
    h_percent = max(min(100, relative_pos[1]), 0) / 100

    w, h = img.size

    crop_size //= 2

    patch_center_x = w * w_percent
    patch_center_y = h * h_percent

    left = patch_center_x - crop_size
    upper = patch_center_y - crop_size
    right = patch_center_x + crop_size
    lower = patch_center_y + crop_size

    border = 3
    draw = ImageDraw.Draw(img)
    draw.rectangle([left-border, upper-border, right+border, lower+border], outline="red", width=border)

    return img.crop((left, upper, right, lower))

def show_comparison_picture(model, dataset, img_id, relative_crop_pos=(50, 50), crop_size=64, scale=3, other_model=True):
    lr_tensor, gt_tensor = dataset[img_id]

    showcase_images = [gt_img, bic_img, hr_img] = prep_showcase(model, lr_tensor, gt_tensor, scale)

    psnr_bic = compute_psnr(np.array(gt_img), np.array(bic_img))
    psnr_up = compute_psnr(np.array(gt_img), np.array(hr_img))
    ssim_bic = compute_ssim(np.array(gt_img), np.array(bic_img), channel_axis=2)
    ssim_up = compute_ssim(np.array(gt_img), np.array(hr_img), channel_axis=2)
    if other_model:
        other_model = "SRCNN" if model.get_name() == "ESRT" else "ESRT"
        other_model_img = Image.open(get_paths(f"./Results/{dataset.get_name()}/{other_model}/X{str(scale)}")[img_id]).convert("RGB")
        showcase_images.insert(2, other_model_img)

        psnr_other_model = compute_psnr(np.array(gt_img), np.array(other_model_img))
        ssim_other_model = compute_ssim(np.array(gt_img), np.array(other_model_img), channel_axis=2)

    showcase_images.append(easy_crop(gt_img, relative_crop_pos, crop_size))
    showcase_images.append(easy_crop(bic_img, relative_crop_pos, crop_size))
    if other_model:
        showcase_images.append(easy_crop(other_model_img, relative_crop_pos, crop_size))
    showcase_images.append(easy_crop(hr_img, relative_crop_pos, crop_size))

    labels = [
        "Original / PSNR / SSIM",
        f"Bicubic / {round(psnr_bic, 4)} / {round(ssim_bic, 4)}",
        f"{model.get_name()} / {round(psnr_up, 4)} / {round(ssim_up, 4)}"
    ]

    if other_model:
        labels.insert(2, f"{other_model} / {round(psnr_other_model, 4)} / {round(ssim_other_model, 4)}")

    show_images(showcase_images, labels)

def metrics_from_results(gt_path, up_path):
    gt_paths = get_paths(gt_path)
    up_paths = get_paths(up_path)

    if len(gt_paths) == 0:
        print("Error: No images")
        return
    if len(up_paths) == 0:
        print("Error: No result images")
        return

    avg_psnr = 0
    avg_ssim = 0

    for up_path, gt_path in zip(up_paths, gt_paths):
        gt_img = Image.open(gt_path).convert('RGB')
        up_img = Image.open(up_path).convert('RGB')
        
        w,h = up_img.size
        gt_img = gt_img.crop((0,0,w,h))

        avg_psnr += compute_psnr(np.array(gt_img).transpose(2,0,1), np.array(up_img).transpose(2,0,1))
        avg_ssim += compute_ssim(np.array(gt_img).transpose(2,0,1), np.array(up_img).transpose(2,0,1), channel_axis=0)

    avg_psnr /= len(gt_paths)
    avg_ssim /= len(gt_paths)

    print("Average PSNR: ", avg_psnr)
    print("Average SSIM: ", avg_ssim)

def eval(model, eval_dataloader, device, scale=-1):
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    with torch.no_grad():
        for lr_tensor, hr_tensor in eval_dataloader:
            lr_tensor = lr_tensor.to(device)
            if scale == -1:
                res = model(lr_tensor)
            else:
                res = model.chunks_forward(lr_tensor, scale)

            hr_np = np.array(tensor_to_image(res.detach()[0].cpu()))
            gt_np = np.array(tensor_to_image(hr_tensor.squeeze(0)))

            gt_np = gt_np[:hr_np.shape[0], :hr_np.shape[1], :hr_np.shape[2],]

            psnr = compute_psnr(hr_np, gt_np)
            ssim = compute_ssim(hr_np, gt_np, channel_axis=2)
            psnr_sum += psnr
            ssim_sum += ssim
    
    avg_psnr = round(psnr_sum / len(eval_dataloader), 4)
    avg_ssim = round(ssim_sum / len(eval_dataloader), 4)

    return avg_psnr, avg_ssim

def train(model, train_dataloader, optimizer, criterion, pbar, device):
    model.train()
    epoch_loss = 0
    for lr_tensor, hr_tensor in train_dataloader:
        lr_tensor = lr_tensor.to(device)
        hr_tensor = hr_tensor.to(device)
        optimizer.zero_grad()
        output = model(lr_tensor)
        loss = criterion(output, hr_tensor)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pbar.update(1)
    
    avg_loss = epoch_loss / len(train_dataloader)

    return avg_loss