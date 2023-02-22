# AI Engine of the application

1. The engine is dockerized.
2. The engine is implemented with Flask 

It is recommended to install Docker extension for VSCode aswell as Docker Desktop for Windows users.

## How to launch the engine containerized

From the /engine folder of the repository :

1. Build the image from the Dockerfile  
``` docker build -t engine .  ```
* where -t specifies the name of the docker's image (should not be changed) as its first arg and the files that should be loaded in it as its second arg ( ```.``` takes everything from the current directory)

2. Run the container from the image  
 ```docker run --rm --name engine_container -d -p 5000:80 engine    ```
* where ```-p```  specifies the local's port and the docker's port used ( local's:docker's)

3. To kill the container   
``` docker kill engine_container```


## Helpers

### Delete all containers
 ```docker rm $(docker ps -a -q) ```

### Delete all images
 ```docker rmi $(docker images -q) ```

