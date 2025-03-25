python perception.py \
  --dino_config configs/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint pretrained_models/groundingdino_swint_ogc.pth \
  --sam_checkpoint pretrained_models/sam_vit_h_4b8939.pth \
  --input_image data/img/teddy.jpg \
  --text_prompt "teddy" \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --ins_config configs/instant-mesh-large.yaml \
  --diffusion_steps 75 \
  --seed 42 \
  --lama_config configs/lama-prediction.yaml \
  --lama_ckpt pretrained_models/big-lama \
  --dilate_kernel_size 50 \
  --export_texmap \
  --device "cuda" \
  --dust3r_confidence 0 \
  --dust3r_ckpt pretrained_models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \

# text propmt like "cup . bottel . computer"

# --lama_find_shade \