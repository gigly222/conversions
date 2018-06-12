docker run -d --net=host --ipc=host  --privileged -v /dev:/dev -e DISPLAY=$DISPLAY -v /home/datascientist:/data -v /tmp/.X11-unix:/tmp/.X11-unix --name ncsdk_v2_display -i -t ncsdk:latest /bin/bash

