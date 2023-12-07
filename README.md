# Train and Test
This project is the implementation of Instant-NGP on OpenIllumilation dataset and render same scence under different lighting conditions with same model. 

To implement our method, please follow the instructions in the Environment section to download the code, dataset, and setting environment. Our method is built based on [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl).

## Structure

### Train the model

```bash
conda activate nsrpl
# Train model from scratch
# "dataset.scene": The scene folder name you want to train
# "lightid": the lighting condition you want to choose, example: "[1]" ; "[1, 4]" ; "[1, 4, 8]"
python launch.py --config configs/openill/nerf-oppo_r0.5.yaml --gpu 0 --train dataset.scene=obj_27_pumpkin2 tag=example --lightid "[1, 4]"

# Train model from checkpoint
# "resume": the path to your checkpoint
python launch.py --config configs/openill/nerf-oppo_r0.5.yaml --gpu 0 --train dataset.scene=obj_02_egg tag=example --resume pth_to_ckpt --resume_weights_only --lightid "[1, 4]"
```

### Testing
The training procedure are by default followed by testing, which computes metrics on test data, generates animations and exports the geometry as triangular meshes. If you want to do testing alone, just resume the pretrained model and replace `--train` with `--test`, for example:

```bash
python launch.py --config path/to/your/exp/config/parsed.yaml --resume path/to/your/exp/ckpt/epoch=0-step=20000.ckpt --gpu 0 --test

# example
python launch.py --config /home/ubuntu/frozen_density/instant-nsr-pl/exp/nerf-oppo-r0.5-obj_02_egg/train_2/config/parsed.yaml --resume /home/ubuntu/frozen_density/instant-nsr-pl/exp/nerf-oppo-r0.5-obj_02_egg/train_2_5/ckpt/epoch=0-step=2501.ckpt --gpu 0 --test --lightid "[2]"
```

# Environment Setting
### Download Code
```bash
# Download code
git clone https://github.com/Brack-Wang/instant-nsr-pl.git
cd instant-nsr-pl
mkdir load

```
### Download OpenIllumilation Dataset
   
Take the website of OpenIllumination as the [overview](https://oppo-us-research.github.io/OpenIllumination/) to select object to download.
```bash
# Update huggingface_hub package
pip install --upgrade huggingface_hub

# Change the `--obj_id` with the Object ID. 
# Change the `--local_dir` with target direction 
python open_illumination.py --light lighting_patterns --obj_id 34 --local_dir /home/ubuntu/frozen_density/instant-nsr-pl/load
```
### Create environment
```bash
# Update sudo
sudo apt-get update
sudo apt update

# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
sh Anaconda3-2023.03-Linux-x86_64.sh
export PATH=/home/anaconda3/bin:$PATH
source ~/.bashrc

# Create conda environment
conda create -n nsrpl python=3.8
conda activate nsrpl

# Install pytorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install tinycudann
export CUDA_HOME=/usr/local/cuda
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install wis3d and pytorch3d
pip install -r requirements.txt
pip install https://github.com/zju3dv/Wis3D/releases/download/2.0.0/wis3d-2.0.0-py3-none-any.whl
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Check whether CUDA tookit installed, If not just install as recommanded
nvcc -V
```

# Useful Command

```bash
# copy folder from server to local
scp -r {root_name}@{IP_address}:/{source_folder_path} {local_path}
# copy folder from server to local
scp -r {local_path} {root_name}@{IP_address}:/{source_folder_path} 


```
