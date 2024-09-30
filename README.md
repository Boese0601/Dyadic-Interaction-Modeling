<p align="center">

  <h2 align="center">Dyadic Interaction Modeling for Social Behavior Generation</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=HuuQRj4AAAAJ"><strong>Minh Tran</strong></a><sup>*</sup>
    ·  
    <a href="https://boese0601.github.io/"><strong>Di Chang</strong></a><sup>*</sup>
    ·
    <a href="https://scholar.google.com/citations?user=5w0f0OQAAAAJ&hl=ru"><strong>Maksim Siniukov</strong></a>
    ·
    <a href="https://www.ihp-lab.org/"><strong>Mohammad Soleymani</strong></a>
    <br>
    University of Southern California
    <br>
    <sup>*</sup>Equal Contribution
    <br>
    </br>
        <a href="https://arxiv.org/abs/2403.09069">
        <img src='https://img.shields.io/badge/arXiv-DIM-green' alt='Paper PDF'>
        </a>
        <a href='https://boese0601.github.io/dim/'>
        <img src='https://img.shields.io/badge/Project_Page-DIM-blue' alt='Project Page'></a>
        <!-- <a href='https://youtu.be/VPJe6TyrT-Y'>
        <img src='https://img.shields.io/badge/YouTube-MagicPose-rgb(255, 0, 0)' alt='Youtube'></a> -->
     </br>
    <table align="center">
        <img src="./assets/demo1.gif">
        <img src="./assets/demo2.gif">
    </table>
</p>

*We propose Dyadic Interaction Modeling, a pre-training strategy that jointly models speakers’ and listeners’ motions and learns representations that capture the dyadic context. We then utilize the pre-trained weights and feed multimodal inputs from the speaker into DIM-Listener. DIM-Listener is capable of generating photorealistic videos for the listener's motion.*

## Getting Started 

Clone repo:

```
git clone https://github.com/Boese0601/Dyadic-Interaction-Modeling.git
cd Dyadic-Interaction-Modeling
```

The code is tested with Python == 3.12.3, PyTorch == 2.3.1 and CUDA == 12.5 on 2 x NVIDIA L40S. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies. You may need to change the torch and cuda version in the `requirements.txt` according to your computer.

```
conda create -n dim python=3.12.3
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cudatoolkit=12.1 -c pytorch -c conda-forge
conda activate dim
pip install -r requirements.txt
```

Download the [CANDOR Corpus](https://convokit.cornell.edu/documentation/candor.html) dataset from the official website here. Extract it and put it under the folder 'data/candor_processed/'.

Download the [LM_Listener](https://github.com/sanjayss34/lm-listener?tab=readme-ov-file) dataset from the official website here. Extract it and put it under the folder 'data/lm_listener_data/'.

Download the [BIWI](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) dataset from the official website here. Extract it and put it under the folder 'data/BIWI_data/'.


Download the [ViCo](https://project.mhzhou.com/vico/) dataset from the official website here. Extract it and put it under the folder 'data/vico_processed/', put 'RLD_data.csv' as 'data/RLD_data.csv'. Use the following script to preprocess ViCo dataset.
```
python vico_preprocessing.py
```

## Model Training 
1. Launch the following lines to train VQ-VAE for speaker and listener.
```
python train_vq.py --config config.yaml
python train_vq.py --config config_speaker.yaml
```
2. Launch the following line to pretrain the model on CANDOR dataset.
```
python train_s2s_pretrain.py
```
5. (Optional) Launch the following line to finetune the model on a specific datset.
```
python finetune_s2s_pretrain.py
```

## Model Evaluation 

Launch the following lines to evaluate the model on each of the datasets.
```
python test_s2s_pretrain.py
python test_biwi.py
python test_l2l.py
python test_s2s.py
```

## News
* **[2024.7.04]** Instructions on training and inference are released.
* **[2024.6.23]** Code is fully released. Instructions on training and inference coming soon.
* **[2024.6.23]** Release Dyadic Interaction Modeling project page.
* **[2024.3.27]** Release Dyadic Interaction Modeling paper.




## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{tran2024dyadic,
      title={Dyadic Interaction Modeling for Social Behavior Generation},
      author={Tran, Minh and Chang, Di and Siniukov, Maksim and Soleymani, Mohammad},
      journal={arXiv preprint arXiv:2403.09069},
      year={2024}
}
```

## License

Our code is distributed under the USC research license. See `LICENSE.txt` file for more information.

## Acknowledgement
This work is supported by the National Science Foundation under Grant No. 2211550. The work was also sponsored by the Army Research Office and was accomplished under Cooperative Agreement Number W911NF-20-2-0053. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Office or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

We appreciate the support from [Haiwen Feng](https://scholar.google.com/citations?user=g5co-iIAAAAJ&hl=en), [Quankai Gao](https://zerg-overmind.github.io/) and [Hongyi Xu](https://hongyixu37.github.io/homepage/) for their suggestions and discussions.


