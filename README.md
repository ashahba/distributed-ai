# distributed-ai
This repository holds some recipes to run AI workloads in distributed mode and on multiple physical machines

## Distibuted TensorFlow with MPI and Horovod
With `TensorFlow v2`, MPI and `Horovod` setting up and running distributed workloads is pretty straight forward.
In this section we cover that setup and run an example for training an small model.

## Prerequisites
Prior to going over setup, ensure you 2 or more Servers/VMs with following specs:
- Each server has passwordless SSH access to the other servers
- Python 3.7 or newer
- Python Pip
- Python Virtualenv
- Python Development tools
- GCC/G++ 8 or newer
- CMake 3 or newer
- OpenMPI(or MPICH)

## Setup:
Create a Python 3 Virtual Environment and install `TensorFlow` and `Horovod` on every node.
Please modify `TF_HVD_DIR` accordingly based on your dataceter recommendation:

```bash
TF_HVD_DIR=/home/<USER>/tf_hvd
virtualenv -p python3 ${TF_HVD_DIR}
PATH=${TF_HVD_DIR}/bin:${PATH}
PYTHONPATH=${TF_HVD_DIR}/lib
```

Now install `intel-tensorflow` or `tensorflow`:

```bash
pip install --no-cache-dir --ignore-installed intel-tensorflow~=2.11

# Customize Horovod installation for TensorFlow
HOROVOD_WITH_MPI=1
HOROVOD_WITH_MXNET=0
HOROVOD_WITH_PYTORCH=0
HOROVOD_WITH_TENSORFLOW=1
pip install --no-cache-dir --ignore-installed horovod~=0.26.1
```

Once both `TensorFlow` and `Horovod` are install successfully, download the following file:
https://github.com/horovod/horovod/blob/v0.26.1/examples/tensorflow2/tensorflow2_mnist.py
and place inside ${TF_HVD_DIR} on both Servers.

# Run the training example:
You can run the example either using `mpirun` or `horovodrun`.

### Using `mpirun`:
```bash
TF_HVD_DIR=/home/<USER>/tf_hvd && \
PYTHONPATH=${TF_HVD_DIR}/lib && \
PATH=${TF_HVD_DIR}/bin:${PATH} \
mpirun --allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 \
    -np 4 \
    -H node-01:2,node02:2 \
    python ${TF_HVD_DIR}/tensorflow2_mnist.py
```

This will generate an output like below:
```
Step #0	Loss: 2.300196
Step #10	Loss: 0.520429
Step #20	Loss: 0.388334
Step #30	Loss: 0.405025
Step #40	Loss: 0.276469
Step #50	Loss: 0.280185
Step #60	Loss: 0.105960
Step #70	Loss: 0.107147
Step #80	Loss: 0.257686
Step #90	Loss: 0.104159
Step #100	Loss: 0.250207
Step #110	Loss: 0.079733
Step #120	Loss: 0.156117
Step #130	Loss: 0.094239
Step #140	Loss: 0.112276
Step #150	Loss: 0.091690
...
...
```

### Using `horovodrun`:
```bash
TF_HVD_DIR=/home/<USER>/tf_hvd && \
PYTHONPATH=${TF_HVD_DIR}/lib && \
PATH=${TF_HVD_DIR}/bin:${PATH} \
horovodrun \
    --verbose \
    -np 4 \
    -H node01:2,node02:4 \
    python ${TF_HVD_DIR}/tensorflow2_mnist.py
```

This will generate an output like below:
```
[1,0]<stdout>:Step #0	Loss: 2.303008
[1,0]<stdout>:Step #10	Loss: 0.631545
[1,0]<stdout>:Step #20	Loss: 0.566792
[1,0]<stdout>:Step #30	Loss: 0.463116
[1,0]<stdout>:Step #40	Loss: 0.252158
[1,0]<stdout>:Step #50	Loss: 0.305903
[1,0]<stdout>:Step #60	Loss: 0.362995
[1,0]<stdout>:Step #70	Loss: 0.224684
[1,0]<stdout>:Step #80	Loss: 0.175049
[1,0]<stdout>:Step #90	Loss: 0.220775
[1,0]<stdout>:Step #100[1,0]<stdout>:	[1,0]<stdout>:Loss: 0.153574
[1,0]<stdout>:Step #110	Loss: 0.199657
[1,0]<stdout>:Step #120	Loss: 0.176949
[1,0]<stdout>:Step #130	Loss: 0.092617
[1,0]<stdout>:Step #140	Loss: 0.112420
[1,0]<stdout>:Step #150	Loss: 0.134965
[1,0]<stdout>:Step #160	Loss: 0.185904
[1,0]<stdout>:Step #170[1,0]<stdout>:	[1,0]<stdout>:Loss: 0.142876
[1,0]<stdout>:Step #180[1,0]<stdout>:	[1,0]<stdout>:Loss: 0.045734
...
...
```
