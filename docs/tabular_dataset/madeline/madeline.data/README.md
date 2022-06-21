# tfrecord to npy
## Build tf2npy image and create tf2npy container
```
pwd # /xxx/.../autogbm/autodl/madeline/madeline.data

docker build -t tf2npy .
docker run -it -v "$(pwd):/app/codalab" --name=tf2npy tf2npy
```
## Exec `tfrecord_to_npy.py` in container
```
pwd # /app/codalab

python tfrecord_to_npy.py
```
