# Inference
1. Before running the patch level evaluation, add your HF token firstly please.

2. Copy the model file from LRZ container /dss/dssfs04/lwp-dss-0002/pn25ke/pn25ke-dss-0002/work/TUM_model to TUM_FM/dinov2/downstream/TUM_small.

2. Run the inference_example.py either using single GPU or multi-GPUs.

3. If you are using multi-GPUs, you need to setup os.environ["NCCL_SOCKET_IFNAME"] = "ibp170s0f0" to a proper ifconfig socket. You may need also have a look at the host file.

# Training
### Edited in 1.30.2025 by Jingsong
1. Download the [teacher backbone](https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_checkpoint.pth?download=1) and [student bone](https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_student_checkpoint.pth?download=1)
and [teacher head](https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_dino_head_checkpoint.pth?download=1) 
and [student head](https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_student_dino_head_checkpoint.pth?download=1). Refer to [here](https://github.com/beneroth13/dinov2/tree/main)

2. Rename these files to 'student_dino_head_checkpoint.pth', 'student_checkpoint.pth', 'teacher_dino_head_checkpoint.pth', 'teacher_checkpoint.pth' and put them in TUM_FM/weights/.
3. Run with the following command to run overfit dataset on one cluster:
```bash
torchrun --nproc_per_node=8 --standalone --nnodes=1 --node_rank=
0 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=172.21.134.195:29603 train.py --config-file ../configs/ssl_8GPUS_overfit.yaml
```
4. More details can be found in config file
