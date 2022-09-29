**Disclaimer:** This repository has been forked from [this implementation](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion). Please find the instructions to train a
model on a vast.ai instance below.

# Dreambooth with Stable Diffusion

This is an implementation of Google's [Dreambooth](https://arxiv.org/abs/2208.12242) with [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

The repository is based on that of [Textual Inversion](https://github.com/rinongal/textual_inversion).
Note that Textual Inversion only optimizes word ebeddings, while DreamBooth fine-tunes the whole diffusion model.

The implementation makes minimum changes over the official codebase of Textual Inversion.

## Usage

### Preparation
To fine-tune a stable diffusion model, you need to obtain the pre-trained stable diffusion models following their [instructions](https://github.com/CompVis/stable-diffusion#stable-diffusion-v1). Weights can be downloaded on [HuggingFace](https://huggingface.co/CompVis). You can decide which version of checkpoint to use, but I use ```sd-v1-4-full-ema.ckpt```.

We also need to create a set of images for regularization, as the fine-tuning algorithm of Dreambooth requires that. Details of the algorithm can be found in the paper. Note that in the original paper, the regularization images seem to be generated on-the-fly. However, here I generated a set of regularization images before the training. The text prompt for generating regularization images can be ```photo of a <class>```, where ```<class>``` is a word that describes the class of your object, such as ```dog```. The command is

```
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt --prompt "a photo of a <class>" 
```

I generate 8 images for regularization, but more regularization images may lead to stronger regularization and better editability. After that, save the generated images (separately, one image per ```.png``` file) at ```/root/to/regularization/images```.

**Updates on 9/9**
We should definitely use more images for regularization. Please try 100 or 200, to better align with the original paper. To acomodate this, I shorten the "repeat" of reg dataset in the [config file](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/blob/main/configs/stable-diffusion/v1-finetune_unfrozen.yaml#L96).

For some cases, if the generated regularization images are highly unrealistic (happens when you want to generate "man" or "woman"), you can find a diverse set of images (of man/woman) online, and use them as regularization images.

### Training
Training can be done by running the following command

```
python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word <xxx>
```

Detailed configuration can be found in ```configs/stable-diffusion/v1-finetune_unfrozen.yaml```. In particular, the default learning rate is ```1.0e-6``` as I found the ```1.0e-5``` in the Dreambooth paper leads to poor editability. The parameter ```reg_weight``` corresponds to the weight of regularization in the Dreambooth paper, and the default is set to ```1.0```.

Dreambooth requires a placeholder word ```[V]```, called identifier, as in the paper. This identifier needs to be a relatively rare tokens in the vocabulary. The original paper approaches this by using a rare word in T5-XXL tokenizer. For simplicity, here I just use a random word ```sks``` and hard coded it.. If you want to change that, simply make a change in [this file](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/blob/main/ldm/data/personalized.py#L10).

Training will be run for 800 steps, and two checkpoints will be saved at ```./logs/<job_name>/checkpoints```, one at 500 steps and one at final step. Typically the one at 500 steps works well enough. I train the model use two A6000 GPUs and it takes ~15 mins.

### Generation
After training, personalized samples can be obtained by running the command

```
python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --ckpt /path/to/saved/checkpoint/from/training
                                 --prompt "photo of a sks <class>" 
```

In particular, ```sks``` is the identifier, which should be replaced by your choice if you happen to change the identifier, and ```<class>``` is the class word ```--class_word``` for training.

## Results
Here I show some qualitative results. The training images are obtained from the [issue](https://github.com/rinongal/textual_inversion/issues/8) in the Textual Inversion repository, and they are 3 images of a large trash container. Regularization images are generated by prompt ```photo of a container```. Regularization images are shown here:

![](assets/a-container-0038.jpg)

After training, generated images with prompt ```photo of a sks container```:

![](assets/photo-of-a-sks-container-0018.jpg)

Generated images with prompt ```photo of a sks container on the beach```:

![](assets/photo-of-a-sks-container-on-the-beach-0017.jpg)

Generated images with prompt ```photo of a sks container on the moon```:

![](assets/photo-of-a-sks-container-on-the-moon-0016.jpg)

Some not-so-perfect but still interesting results:

Generated images with prompt ```photo of a red sks container```:

![](assets/a-red-sks-container-0021.jpg)

Generated images with prompt ```a dog on top of sks container```:

![](assets/a-dog-on-top-of-sks-container-0023.jpg)


## Run a training session on a vast.ai instance

Setting up the environment, training the model, and downloading/uploading data for one session should cost around $1.
Make sure to follow the instructions below closely to avoid losing time on the virtual machine.

1. Prepare the training and regularization data in advance. Since inference doesn't require that much memory, 
regularization images can be generated locally if you have GPU machine with about 10GB memory. Use the command below:

```
python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --ckpt /path/to/saved/checkpoint/from/training
                                 --prompt "photo of a sks <class>" 
```

2. Essential points for the data:
     - Having square images for both is important. Otherwise, the resulting images would come out distorted.
     - Using as little as 8 images for both training and regularization works; however, I use +50 training and +100 regularization images.
You can go even higher with the numbers.
     - If the generated regularization images do not represent the class well enough, you can simply gather images from the web.
3. Register/log in to a vast.ai account.
4. Under Client/Create section, select an instance with at least one Nvidia RTX A6000. Things to pay attention to while choosing an instance:
   - Select a PyTorch image (pytorch/pytorch) in Instance Configuration.
   - Download and upload speeds (Inet Up/Down) should be high enough. +100mbps is ideal.
   - Select On-Demand option on top of the list. 
5. Go to Client/Instances section, and once the instance goes online, click on Open. It should load the Jupyter dashboard on your browser. 
Here, click on New/Terminal to open a terminal session.
6. Clone the repository: `git clone https://github.com/zanilzanzan/DreamBooth.git`.
7. Go back to Jupyter dashboard and open the file `DreamBooth/setup_env.sh` and enter your HuggingFace username and password information on the second line.
The environment setup process downloads the Stable Diffusion model `sd-v1-4-full-ema.ckpt` from HuggingFace, and you need to provide your credentials
to be able to access the file. If you don't have an HuggingFace account, please go ahead and create one. Note: If there is a more secure download method,
please let me know. Save the edited file (ctrl+s).
8. Head back to the terminal session, cd into the project directory: `cd DreamBooth`, and run the environment setup script: `bash setup_env.sh`.
This will download the necessary files, install required software, and set up the conda environment. Installation takes a while, please be patient.
9. Once the script finishes, activate the conda environment: `conda activate ldm`.
10. Make sure that the training and regularization data are transferred. If you chose to generate the regularization images on the VM, plese run the
inference command provided in the first step.
11. Now you can start training with the following command:

```
python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml
               -t
               --gpus 0,
               --actual_resume ./weights/sd-v1-4-full-ema.ckpt 
               -n <whatever you want this training to be called> 
               --data_root <the relative path to your training images> 
               --reg_data_root <the relative path to your regularization images> 
               --class_word <the word you used to get for regularization data>
```
11. Points to keep in mind, before starting a training session:
    - Modify `max_steps` in the config file `configs/stable-diffusion/v1-finetune_unfrozen.yaml` for the total number of iterations.
Some found that 1000 steps is the sweet spot, but for portraits 3000 steps created wonders in my case.
    - In the current implementation, the latest checkpoint is saved in every 500 iterations. The checkpoint can be found 
in `logs/<experiment_name>/checkpoints/` as `last.ckpt`.
12. After training is done, switch to Jupyter dashboard, find your model file, and simply download it. If your file doesn't 
appear where it is supposed to, you can move the model file to the main project directory using the terminal: 
`mv /workspace/DreamBooth/logs/<experiment_name>/checkpoints/last.ckpt /workspace/DreamBooth`. Then it should be visible.

