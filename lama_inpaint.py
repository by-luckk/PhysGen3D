import os
import sys
import cv2
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from submodules.lama.saicinpainting.evaluation.utils import move_to_device
from submodules.lama.saicinpainting.evaluation.refinement import refine_predict
from submodules.lama.saicinpainting.training.trainers import load_checkpoint
from submodules.lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils.utils import load_img_to_array, save_array_to_img, dilate_mask

# def mask_with_shade(img, mask, out_dir=None):
#     h, w = img.shape[:2]
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

#     # Pick the binary shadows if the shadow is around the object
#     binary = cv2.bitwise_and(binary, cv2.bitwise_not(mask))

#     # Invert the binary image to focus on black regions
#     binary_inverted = cv2.bitwise_not(binary)

#     # Find all connected components in the inverted binary image
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inverted, connectivity=8)

#     # Find the largest connected component (excluding the background)
#     largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

#     # Create a mask for the largest connected component
#     largest_mask = np.zeros_like(binary_inverted)
#     largest_mask[labels == largest_component] = 255

#     if out_dir is not None:
#         cv2.imwrite(out_dir / 'largest_black_region.png', largest_mask)
#     return largest_mask

def mask_with_shade_new(img, mask, out_dir=None):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Pick the binary shadows if the shadow is around the object
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(mask))

    # Invert the binary image to focus on black regions
    binary_inverted = cv2.bitwise_not(binary)

    # Find all connected components in the inverted binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inverted, connectivity=8)

    # Find the connected component that has the largest intersection with the original mask
    max_intersection = 0
    intersecting_component = 0
    for i in range(1, num_labels):  # Start from 1 to exclude the background
        current_mask = (labels == i).astype(np.uint8) * 255
        intersection = cv2.countNonZero(cv2.bitwise_and(current_mask, mask))
        if intersection > max_intersection:
            max_intersection = intersection
            intersecting_component = i

    # If no intersection is found, return None or handle as needed
    if intersecting_component == 0:
        return mask

    # Create a mask for the component that has the largest intersection with the original mask
    intersecting_mask = np.zeros_like(binary_inverted)
    intersecting_mask[labels == intersecting_component] = 255

    if out_dir is not None:
        cv2.imwrite(str(Path(out_dir) / 'intersecting_region.png'), intersecting_mask)
    return intersecting_mask

def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod=8,
        device="cuda",
        dilation=0,
        out_path=None,
        find_shade=False
):
    assert len(mask.shape) == 2
    mask = mask.astype('uint8')
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if find_shade:
        mask = mask_with_shade_new(img, mask)
    if dilation > 0:
        mask = dilate_mask(mask, dilation)
    if out_path is not None:
        save_array_to_img(mask, out_path)
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['unpad_to_size'] = torch.tensor(batch['image'].shape[-2:]).unsqueeze(1)
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)

    if predict_config.get('refine', False):
        assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
        cur_res = refine_predict(batch, model, **predict_config.refiner)
        cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
    else:
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
            cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model


@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        img: np.ndarray,
        mask: np.ndarray,
        config_p=None,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2 
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch['unpad_to_size'] = unpad_to_size.unsqueeze(0)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res



def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--input_mask_glob", type=str, required=True,
        help="Glob to input masks",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        '--dilate_kernel_size', type=int,
         default=None, help="Dilate kernel size. Default: None"
    )


if __name__ == "__main__":
    """Example usage:
    python lama_inpaint.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --output_dir results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_stem = Path(args.input_img).stem
    mask_ps = sorted(glob.glob(args.input_mask_glob))
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_img_to_array(args.input_img)
    for mask_p in mask_ps:
        mask = load_img_to_array(mask_p)
        print(mask.shape)
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device,
            dilation=args.dilate_kernel_size, find_shade=True)
        save_array_to_img(img_inpainted, img_inpainted_p)