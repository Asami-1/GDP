# GDP
Repository related to Cranfield's AAI MSCs GDP

## Steps to use VitPose

Download the weights at https://onedrive.live.com/?authkey=%21AMfI3aaOHafYvIY&id=E534267B85818129%21166&cid=E534267B85818129&parId=root&parQt=sharedby&parCid=D632C8CD854B2F0E&o=OneUp 
Place them at the root of the pose directory.

We use conda for dependencies.
```
conda env create -f environment.yml
conda activate vitenv
```
Install modules from the official repo : 
```
cd ViTpose
pip install -v -e . 
```
You should now be able to run the pose_from_videos_folder.py script. Refer to the comments to set up the video  and target folder paths 
```
cd ..
python pose_from_videos_folder.py
```


# Data Generation part: 

## Folders description : 

  Agressive : contain all agressive videos from the 2 cams
  
  Peace : contain all peace videos from the 2 cams

  data_full arborescence :
  
  	data_full
  
	  	train
		  	conflict
		
		  	peaceful
			
	 	 test
		  	conflict
			
		  	peaceful

(split 75 / 25) 

If we want to use the only 4 first seconds of each videos, there is a script "dataset_gen_fixed_time.py" which creates a new dataset from data_full but containing video_duration*30 json files in each folder.

