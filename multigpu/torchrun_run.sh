OMP_NUM_THREADS=12               \
CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7     \
torchrun     --standalone       \
             --nnodes=1         \
             --nproc_per_node=8 \
              multi_quick.py