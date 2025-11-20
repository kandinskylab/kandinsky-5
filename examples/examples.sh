cd ..

# # 5sec, sd, 1 gpu
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_sd.yaml --output_filename ./test-1.mp4

# # 5sec, sd, 2 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 2 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_sd.yaml --output_filename ./test-2.mp4

# # 5sec, sd, 4 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 4 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_sd.yaml --output_filename ./test-4.mp4

# # 5sec, sd, 8 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 8 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_sd.yaml --output_filename ./test-8-sd.mp4

# # 5sec, sd, 1 gpu, magcache
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_sd.yaml --output_filename ./test-1-magcache.mp4 --magcache

# # 5sec, hd, 1 gpu
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_hd.yaml --output_filename ./test-1-hd.mp4 --width 1280 --height 768

# # 5sec, hd, 2 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 2 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_hd.yaml --output_filename ./test-2-hd.mp4 --width 1280 --height 768

# # 5sec, hd, 4 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 4 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_hd.yaml --output_filename ./test-4-hd.mp4 --width 1280 --height 768

# # 5sec, hd, 8 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 8 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_hd.yaml --output_filename ./test-8-hd.mp4 --width 1280 --height 768

# # 5sec, hd, 1 gpu, magcache
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "The bear plays balalaika." --image "./assets/test_image.jpg" --config ./configs/k5_pro_i2v_5s_sft_hd.yaml --output_filename ./test-1-hd-magcache.mp4 --width 1280 --height 768 --magcache --offload

# # 10sec, sd, 1 gpu
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_sd.yaml --output_filename ./test-1-10sec.mp4 --video_duration 10 --offload

# # 10sec, sd, 2 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 2 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_sd.yaml --output_filename ./test-2-10sec.mp4 --video_duration 10

# # 10sec, sd, 4 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 4 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_sd.yaml --output_filename ./test-4-10sec.mp4 --video_duration 10

# # 10sec, sd, 8 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 8 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_sd.yaml --output_filename ./test-8-10sec.mp4 --video_duration 10

# # 10sec, sd, 1 gpu, magcache
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_sd.yaml --output_filename ./test-1-10sec-magcache.mp4 --video_duration 10 --magcache --offload

# # 10sec, hd, 1 gpu
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_hd.yaml --output_filename ./test-1-10sec-hd.mp4 --video_duration 10 --width 1280 --height 768 --offload

# # 10sec, hd, 2 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 2 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_hd.yaml --output_filename ./test-2-10sec-hd.mp4 --video_duration 10 --width 1280 --height 768

# # 10sec, hd, 4 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 4 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_hd.yaml --output_filename ./test-4-10sec-hd.mp4 --video_duration 10 --width 1280 --height 768

# # 10sec, hd, 8 gpus
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 8 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_hd.yaml --output_filename ./test-8-10sec-hd.mp4 --video_duration 10 --width 1280 --height 768

# # 10sec, hd, 1 gpu, magcache
# python -m torch.distributed.launch --nnodes 1 --nproc-per-node 1 test.py --prompt "A dog in red hat" --config ./configs/k5_pro_t2v_10s_sft_hd.yaml --output_filename ./test-1-10sec-hd.mp4 --video_duration 10 --width 1280 --height 768 --magcache --offload

# # Kandinsky 5.0 Video Lite T2V 5s
# python test.py --prompt "A dog in red hat"

# # Kandinsky 5.0 Video Lite T2V 10s
# python test.py --config ./configs/k5_lite_t2v_10s_sft_sd.yaml --prompt "A dog in red hat" --video_duration 10

# # Kandinsky 5.0 Video Lite T2V distill 5s
# python test.py --config ./configs/k5_lite_t2v_5s_distil_sd.yaml --prompt "A dog in red hat"          

# # Kandinsky 5.0 Video Lite T2V distill 10s
# python test.py --config ./configs/k5_lite_t2v_10s_distil_sd.yaml --prompt "A dog in red hat" --video_duration 10

# # Kandinsky 5.0 Video Lite I2V 5s
# python test.py --config ./configs/k5_lite_i2v_5s_sft_sd.yaml --prompt "The Dragon breaths fire." --image "./assets/test_image.jpg" --video_duration 5

# # Kandinsky 5.0 Image Lite T2I
# python test.py --config ./configs/k5_lite_t2i_sft_hd.yaml --prompt "A dog in a red hat" --width=1280 --height=768

# # Kandinsky 5.0 Image Lite I2I
# python test.py --config ./configs/k5_lite_i2i_sft_hd.yaml --image ./assets/cat_in_hat.png --prompt "Replace the cat with a husky, leave the rest unchanged"