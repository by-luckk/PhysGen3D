export CUDA_VISIBLE_DEVICES=6
python rendering_sample.py \
-i ./sim_result/sim_result_2024-11-15_09-22-27 \
--path outputs/DSC00531 \
--env data/test/hdr/DSC00531.exr \
-b 0 \
-f \
-e 100 \
-s 1 \
-o render_result/161 \
-M 460 \
-p 20 \
-c \
--shutter-time 0.0

# -i ./sim_result/sim_result_2024-11-11_22-28-27 \
# --path outputs/DSC00543 \
# --env data/test/hdr/DSC00543.exr \

# -i ./sim_result/sim_result_2024-11-12_05-05-51 \
# --path outputs/manyapple \
# --env data/hdr/manyapple.exr \

# -i ./sim_result/sim_result_2024-11-12_10-59-31 \
# --path outputs/sphinx \
# --env data/hdr/apple.exr \

# -i ./sim_result/sim_result_2024-11-12_11-52-00 \
# --path outputs/dog \
# --env data/hdr/potato.exr \

# -i ./sim_result/sim_result_2024-11-12_23-30-58 \
# --path outputs/basketball_on_court \
# --env data/hdr/apple.exr \

# -i ./sim_result/sim_result_2024-11-13_12-52-50 \
# --path outputs/rabbit2 \
# --env data/hdr/panda.exr \