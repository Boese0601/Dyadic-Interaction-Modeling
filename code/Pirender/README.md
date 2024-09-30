# PIRenderer


## Getting Started 

To set up the environment, follow the instructions on the [PIRender GitHub page](https://github.com/RenYurui/PIRender).

Please download the VoxCeleb and ViCo datasets and extract them to the `./datasets/` folder. Extract the EMOCA features.

To pretrain the PIRender model on the VoxCeleb dataset, use the following command:

```
python train.py --person_number -1 --single_gpu --seed 42  --config config/face_vox_pretrainA.yaml
```

To fine-tune the PIRender model on the ViCo dataset, use:

```
python train.py --person_number -1 --single_gpu --seed 42  --config config/face.yaml
```

For inference use:
 
```
python inference_newmodel.py 
```


For more detailed information on the implementation, please refer to the ICCV 2021 paper "[PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering](https://arxiv.org/abs/2109.08379)".


## Acknowledgement 

We build our render code base on  [PIRender](https://github.com/RenYurui/PIRender), [imaginaire](https://github.com/NVlabs/imaginaire), [video-preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing).

