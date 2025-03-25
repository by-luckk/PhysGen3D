export CUDA_VISIBLE_DEVICES=1 \
python simulate/render_particles.py \
-i ./sim_result/sim_result_2024-10-11_21-42-06 \
-b 0 \
-e 200 \
-s 1 \
--gpu-memory 30 \
-o sim_result/1 \
-M 460 \
--shutter-time 0.0
