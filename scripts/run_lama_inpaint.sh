python lama_inpaint.py \
    --input_img data/car/raw_image.jpg \
    --input_mask_glob outputs/car/mask/mask.jpg \
    --output_dir outputs \
    --lama_config configs/lama-prediction.yaml \
    --lama_ckpt pretrained_models/big-lama \
    --dilate_kernel_size 10