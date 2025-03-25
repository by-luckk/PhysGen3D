export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
  --config configs/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint pretrained_models/groundingdino_swint_ogc.pth \
  --sam_checkpoint pretrained_models/sam_vit_h_4b8939.pth \
  --input_image assets/room.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "computer" \
  --device "cuda"