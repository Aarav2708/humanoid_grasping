# Contact-GraspNet  

### Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes   
Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, Dieter Fox  
ICRA 2021    

[paper](https://arxiv.org/abs/2103.14127), [project page](https://research.nvidia.com/publication/2021-03_Contact-GraspNet%3A--Efficient), [video](http://www.youtube.com/watch?v=qRLKYSLXElM)

<p align="center">
  <img src="examples/2.gif" width="640" title="UOIS + Contact-GraspNet"/>
</p>

## Installation

This code has been tested with python 3.7, tensorflow 2.2, CUDA 11.1

1.Create a conda env called zshot using:
  ```shell
  conda env create -f zeroshot_env.yml
  ```
2.Follow the instructions given [here](https://github.com/airo-ugent/airo-mono) and create a conda env called airo-mono.

  Create a conda env called sam_env using
  ```shell
  conda env create -f sam_env.yml
  ```
3.Add weights for the [sam_model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 

4.Create a conda env called contatct_graspnet using:
```shell
conda env create -f contact_graspnet.yml
```
5.Before running the pipeline, check camera using:
```shell
conda activate airo_mono
python3 robot/check_camera.py
```
6.Now, finally run the pipeline using
```shell
bash run_all.sh
```



### Troubleshooting

- Recompile pointnet2 tf_ops:
```shell
sh compile_pointnet_tfops.sh
```

### Hardware
Training: 1x Nvidia GPU >= 24GB VRAM, >=64GB RAM  
Inference: 1x Nvidia GPU >= 8GB VRAM (might work with less)

## Download Models and Data
### Model
Download trained models from [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing) and copy them into the `checkpoints/` folder.
### Test data
Download the test data from [here](https://drive.google.com/drive/folders/1TqpM2wHAAo0j3i1neu3Xeru3_WnsYQnx?usp=sharing) and copy them them into the `test_data/` folder.



## Citation

```
@article{sundermeyer2021contact,
  title={Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```
