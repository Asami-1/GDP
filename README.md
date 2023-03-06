# GDP
Repository related to Cranfield's AAI MSCs GDP

## Steps to use VitPose

Download the weights at https://onedrive.live.com/?authkey=%21AMfI3aaOHafYvIY&id=E534267B85818129%21166&cid=E534267B85818129&parId=root&parQt=sharedby&parCid=D632C8CD854B2F0E&o=OneUp 
Place them at the root of the pose directory.

We use a conda for dependencies.
```
conda create --name vitenv python=3.7.2
conda activate vitenv
```
Install cuda toolkit 11.7
```
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc 
```
Install all the dependencies : 
```
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmdet==2.28.1 timm==0.4.9 einops
```
Install modules from the official repo : 
```
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTpose
pip install -v -e . 
```

You should now be able to run the script.py 
```
cd ..
python script.py
```

You should see a test_labeled file appear.

