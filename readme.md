1. Copy the model file from LRZ container /dss/dssfs04/lwp-dss-0002/pn25ke/pn25ke-dss-0002/work/TUM_model to TUM_FM/dinov2/downstream/TUM_small.

2. Run the inference_example.py either using single GPU or multi-GPUs.

3. If you are using multi-GPUs, you need to setup os.environ["NCCL_SOCKET_IFNAME"] = "ibp170s0f0" to a proper ifconfig socket. You may need also have a look at the host file.
