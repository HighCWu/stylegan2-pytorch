# This Anime Doesn't Exist (PyTorch version and PaddlePaddle version)

### Convert weight from aydao's tf checkpoints to pytorch

First, you need to clone this repository:

> git clone https://github.com/HighCWu/stylegan2-pytorch2paddle -b tadne --recursive

Download pretrained weight:

- [Mega ](https://mega.nz/file/nUkWFZgS#EHHrqILumjpTppSXG-QlCOdWaUIVLTDnqPxsXPrI3UQ)(1GB)
- [Google Drive](https://drive.google.com/file/d/1qNhyusI0hwBLI-HOavkNP5I0J0-kcN4C/view) [(backup)](https://drive.google.com/file/d/1A-E_E32WAtTHRlOzjhhYhyyBDXLJN9_H/view)
- Rsync mirror: `rsync -v rsync://78.46.86.149:873/biggan/2020-11-27-aydao-stylegan2ext-danbooru2019s-512px-5268480.pkl ./network-tadne.pkl`

Then you can convert it like this:

> python convert_weight.py network-tadne.pkl

This will create converted network-tadne.pt file.

Converting script is available on [Colab](https://colab.research.google.com/github/HighCWu/stylegan2-pytorch2paddle/blob/tadne/convert_weight.ipynb).

### Convert weight from pytorch to paddlepaddle

Run convert script:

```bash
python convert_weight_torch2pp.py network-tadne.pt
```

Test converted model:

```bash
python generate_pp.py --size 512 --ckpt network-tadne.g_ema
```

Converting script is available on [Colab](https://colab.research.google.com/github/HighCWu/stylegan2-pytorch2paddle/blob/tadne/convert_weight_torch2pp.ipynb).

...

The following is the description of the original repo:

# StyleGAN 2 in PyTorch

Implementation of Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958) in PyTorch

## Notice

I have tried to match official implementation as close as possible, but maybe there are some details I missed. So please use this implementation with care.

## Requirements

I have tested on:

* PyTorch 1.3.1
* CUDA 10.1/10.2

## Usage

First create lmdb datasets:

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

This will convert images to jpeg and pre-resizes it. This implementation does not use progressive growing, but you can create multiple resolution datasets using size arguments with comma separated lists, for the cases that you want to try another resolutions later.

Then you can train model in distributed settings

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

train.py supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

### Convert weight from official checkpoints

You need to clone official repositories, (https://github.com/NVlabs/stylegan2) as it is requires for load official checkpoints.

For example, if you cloned repositories in ~/stylegan2 and downloaded stylegan2-ffhq-config-f.pkl, You can convert it like this:

> python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl

This will create converted stylegan2-ffhq-config-f.pt file.

### Generate samples

> python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT

You should change your size (--size 256 for example) if you train with another dimension.

### Project images to latent spaces

> python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...

### Closed-Form Factorization (https://arxiv.org/abs/2007.06600)

You can use `closed_form_factorization.py` and `apply_factor.py` to discover meaningful latent semantic factor or directions in unsupervised manner.

First, you need to extract eigenvectors of weight matrices using `closed_form_factorization.py`

> python closed_form_factorization.py [CHECKPOINT]

This will create factor file that contains eigenvectors. (Default: factor.pt) And you can use `apply_factor.py` to test the meaning of extracted directions

> python apply_factor.py -i [INDEX_OF_EIGENVECTOR] -d [DEGREE_OF_MOVE] -n [NUMBER_OF_SAMPLES] --ckpt [CHECKPOINT] [FACTOR_FILE]

For example,

> python apply_factor.py -i 19 -d 5 -n 10 --ckpt [CHECKPOINT] factor.pt

Will generate 10 random samples, and samples generated from latents that moved along 19th eigenvector with size/degree +-5.

![Sample of closed form factorization](factor_index-13_degree-5.0.png)

## Pretrained Checkpoints

[Link](https://drive.google.com/open?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO)

I have trained the 256px model on FFHQ 550k iterations. I got FID about 4.5. Maybe data preprocessing, resolution, training loop could made this difference, but currently I don't know the exact reason of FID differences.

## Samples

![Sample with truncation](doc/sample.png)

Sample from FFHQ. At 110,000 iterations. (trained on 3.52M images)

![MetFaces sample with non-leaking augmentations](doc/sample-metfaces.png)

Sample from MetFaces with Non-leaking augmentations. At 150,000 iterations. (trained on 4.8M images)


### Samples from converted weights

![Sample from FFHQ](doc/stylegan2-ffhq-config-f.png)

Sample from FFHQ (1024px)

![Sample from LSUN Church](doc/stylegan2-church-config-f.png)

Sample from LSUN Church (256px)

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
