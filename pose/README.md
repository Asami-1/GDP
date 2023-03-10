## Steps to use VitPose

Download the weights at https://onedrive.live.com/?authkey=%21AMfI3aaOHafYvIY&id=E534267B85818129%21166&cid=E534267B85818129&parId=root&parQt=sharedby&parCid=D632C8CD854B2F0E&o=OneUp 
Place them at the root of the pose directory.

We use conda for dependencies.
```
conda env create -f environment.yml
conda activate vitenv
```
You should now be able to run the pose_from_videos_folder.py script. Refer to the comments to set up the video  and target folder paths 
```
cd ..
python pose_from_videos_folder.py
```

