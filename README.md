# smurfs_segmentation_classification
Segment and Classify the smurfs

## Setting up Environment
1. Build the image

  ``` docker compose build ```

2. Start the container
   
  ``` docker compose up ```
3. Press the link of Jupyter Notebook appearing in the cmd containing the access token

4. Starting Tensorboard inside the docker container
     
  ```docker exec -it smurfs_container bash```
  
  ```tensorboard --logdir=runs```


