import os
import re
import cv2
import sys
import argparse
import numpy as np
import json
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "submodules", "lama"))
sys.path.append(os.path.join(os.getcwd(), "submodules", "dust3r"))

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)

# instant mesh
from my_instant_mesh import MyInstantMesh

# lama inpaint
from lama_inpaint import inpaint_img_with_lama

# dust3r
from my_dust3r import run_dust3r

from utils.utils import load_img_to_array, save_array_to_img
from locate.fit_object_pytorch3d import estimate_pose
from locate.pytorch_opt import optimize_mesh_pose, render_depth

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    local_model_path = "pretrained_models/bert-base-uncased"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    if hasattr(args, "text_encoder_type") and os.path.exists(local_model_path):
        args.text_encoder_type = local_model_path 
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def extract_masked_region(image, mask):
    assert image.shape[:2] == mask.shape, "Image and mask dimensions do not match"
    white_background = np.ones_like(image) * 255
    black_background = np.zeros_like(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    white_background[mask == 1] = masked_image[mask == 1]
    black_background[mask == 1] = masked_image[mask == 1]
    return white_background, black_background

def extract_label(input_string):
    match = re.match(r"([a-zA-Z]+)\(\d+\.\d+\)", input_string)
    if match:
        return match.group(1)
    else:
        return "other"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    mask_dir = os.path.join(output_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    for idx, mask in enumerate(mask_list):
        mask_np = mask.cpu().numpy()[0]
        mask_img[mask_np > 0] = value + idx + 1
        cv2.imwrite(os.path.join(mask_dir, f'mask_{idx}.jpg'), (mask_np * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(mask_dir, 'mask.jpg'), (mask_img.numpy() * 255).astype(np.uint8))
    #     mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    #     # save mask
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(mask.cpu().numpy()[0], cmap='gray', aspect='equal')
    #     plt.axis('off')
    #     plt.savefig(os.path.join(mask_dir, f'mask_{idx}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_img.numpy(), aspect='equal')
    # plt.axis('off')
    # plt.savefig(os.path.join(mask_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(mask_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def get_white_mask_area(mask_img):
    mask_array = np.array(mask_img)
    white_threshold = 250
    white_mask = mask_array >= white_threshold
    return white_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser("perception", add_help=True)
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--save_video", action="store_true", help="save video")
    parser.add_argument("--output_dir", type=str, required=False, default="outputs", help="output directory")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)

    # cfg
    config_file = args.dino_config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_path))[0])
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "objects"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "inpaint"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)

    print("Running grounding dino")
    # load grounding dino model
    dino_model = load_model(config_file, grounded_checkpoint, device=device)
    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    print("Running SAM")
    # initialize SAM
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    del(predictor)

    # draw output image
    print("Drawing output image")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    # draw seperated objects
    label_count = {}
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    object_names = []
    for mask, box, phrase in zip(masks, boxes_filt, pred_phrases):
        label = extract_label(phrase)
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1
        masked_white, masked_black = extract_masked_region(image, mask.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.numpy().tolist())
        cropped_white = masked_white[y1:y2, x1:x2]
        cropped_black = masked_black[y1:y2, x1:x2]
        object_names.append(f'{label}{label_count[label]}')
        cv2.imwrite(os.path.join(output_dir, "objects", f'{label}{label_count[label]}.jpg'), cropped_white)
        cv2.imwrite(os.path.join(output_dir, "objects", f'{label}{label_count[label]}_black.jpg'), cropped_black)
    print("objects: ", label_count)

    # inpaint background
    print("Inpaint background")
    torch.set_grad_enabled(True)
    img = load_img_to_array(image_path)
    last_img = img
    for i, mask in enumerate(masks):
        mask = mask[0].cpu().numpy()
        img_inpainted_p = os.path.join(output_dir, "inpaint", f"inpainted_with_mask{i}.jpg")
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device,
            dilation=args.dilate_kernel_size, find_shade=args.lama_find_shade,
            out_path=os.path.join(output_dir, "mask", f"mask_{i}_final.jpg"))
        last_img = inpaint_img_with_lama(
            last_img, mask, args.lama_config, args.lama_ckpt, device=device,
            dilation=args.dilate_kernel_size, find_shade=args.lama_find_shade,
            out_path=os.path.join(output_dir, "mask", f"mask_{i}_final.jpg"))
        save_array_to_img(img_inpainted, img_inpainted_p)
        save_array_to_img(last_img, os.path.join(output_dir, "inpaint", f"inpainted_with_mask{i}_cumu.jpg"))
    final_path = os.path.join(output_dir, "inpaint", f"inpainted_all.jpg")
    last_img = cv2.resize(last_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    save_array_to_img(last_img, final_path)
    
    # depth estimation
    print('Depth estimation')
    depth_img_path = [# os.path.join(output_dir, "inpaint", "inpainted_all.jpg"), 
                      os.path.join(output_dir, "raw_image.jpg")]
    out_path = os.path.join(output_dir, "depth", "pcd")
    os.makedirs(out_path, exist_ok=True)
    run_dust3r(args.dust3r_ckpt, depth_img_path, out_path, 
                device=args.device, min_conf_thr=args.dust3r_confidence)

    # reconstruct object
    print("Generating object mesh")
    mesh_model = MyInstantMesh(args.ins_config,
                               input_path=os.path.join(output_dir, "objects"),
                               output_path=output_dir,
                               seed=args.seed)
    mesh_model.multiview_generation(diffusion_steps=args.diffusion_steps)
    mesh_model.reconstruction(args.scale, args.view, args.distance,
                              args.export_texmap, args.save_video)
    
    # registrate object
    print('Registrate object')
    pcd_file = os.path.join(output_dir, "depth", "pcd", "raw_image.ply")
    raw_img = os.path.join(output_dir, "raw_image.jpg")
    matrices = []
    for box, name in zip(boxes_filt, object_names):
        out = os.path.join(output_dir, 'objects', name)
        os.makedirs(out, exist_ok=True)
        mesh_file = os.path.join(output_dir, 'meshes', f'{name}.obj')
        ref_img = os.path.join(output_dir, 'objects', f'{name}_black.jpg')
        M, plane_model, fov = estimate_pose(mesh_file, pcd_file, ref_img, raw_img, box.numpy(), out, num_samples=6, num_ups=1)
        matrices.append(M.tolist())
    print(matrices, plane_model, fov)
    data = {"matrices": matrices,
        "plane_model": plane_model.tolist(),
        "fov": fov.tolist(),
        "object_names": object_names}
    output_path = os.path.join(output_dir, 'transform.json')
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # optimize pose
    torch.set_grad_enabled(True)
    matrices_optimized = []
    ply_file = os.path.join(output_dir, 'depth', 'pcd', 'raw_image.ply')
    with open(os.path.join(output_dir, 'transform.json'), 'r') as json_file:
        data = json.load(json_file)
    fov = np.array(data['fov'])
    fov_x = np.arctan(fov[2]) * 2
    matrices = np.array(data['matrices'])
    img_size = np.array([size[1], size[0]])
    ref_depth = render_depth(ply_file, fov_x, img_size)
    mask_img = Image.open(os.path.join(output_dir, 'mask', 'mask_0.jpg')).convert('L')
    resized_mask = mask_img.resize((img_size[1], img_size[0]), Image.LANCZOS)
    white_mask = get_white_mask_area(resized_mask)
    for mask, name, matrix in zip(masks, object_names, matrices):
        optimized_pose = optimize_mesh_pose(
            mesh_file=os.path.join(output_dir, 'meshes', f'{name}.obj'),
            ref_depth=ref_depth,
            mask=white_mask,
            pre_transform_matrix=matrix,
            fov_x=fov_x,
            img_size=img_size,
            num_iterations=300,
            lr=0.001,
            # save_dir='locate/pics_sgd' ## you can turn this on to save the intermediate results
        )
        matrices_optimized.append(optimized_pose.tolist())
    data["matrices_optimized"] = matrices_optimized
    with open(os.path.join(output_dir, 'transform.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)