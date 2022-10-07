OMP_NUM_THREADS=12               \
CUDA_VISIBLE_DEVICES=4,5,6,7     \
torchrun     --standalone       \
             --nnodes=1         \
             --nproc_per_node=4 \
             tools/train_net.py