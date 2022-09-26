# Dreambooth on Stable Diffusion

This is an implementtaion of Google's [Dreambooth](https://arxiv.org/abs/2208.12242) with [Stable Diffusion](https://github.com/CompVis/stable-diffusion). The original Dreambooth is based on [Imagen](https://imagen.research.google/) text-to-image model. However, neither the model nor the pre-trained weights of Imagen is available. To enable people to fine-tune a text-to-image model with a few examples, I implemented the idea of Dreambooth on Stable diffusion.

This code repository is based on that of [Textual Inversion](https://github.com/rinongal/textual_inversion). Note that Textual Inversion only optimizes word ebedding, while dreambooth fine-tunes the whole diffusion model.

The implementation makes minimum changes over the official codebase of Textual Inversion. In fact, due to lazyness, some components in Textual Inversion, such as the embedding manager, are not deleted, although they will never be used here.

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


## Run a training session on a VAST.AI instance

**Important note:** The instructions were directly taken from [this comment](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/issues/4#issuecomment-1246140407). I have recently experimented with the procedure. In a few days, I will be summarizing all the required steps and update the code for a smooth training. Please stay tuned.

Training session on VAST.AI costs around $1

1- Prepare your training and regularization data in advance

2- Pick an (when finding an instance, make sure to select a PyTorch instance config) instance with at least 1 A6000 (cheapest that meets the VRAM reqs, I've found - and 1 is good to start with since you might be spending more time figuring out how to set it up than actually training it). Make sure the download (and upload) speeds are decent, like >100mbps

3. Go in and open a terminal session

4. Clone this repository: `git clone https://github.com/zanilzanzan/DreamBooth.git`

5. cd into the directory cd Dreambooth-Stable-Diffusion and make a conda environment for it `conda env create -f environment.yaml` this will take a lil' while

6. While that's happening, create a new terminal instance and pull down the SD EMA model, make sure you're in the project directory (Dreambooth-Stable...). Easiest way to do this is download it from hugging face `wget --http-user=USERNAME --http-password=PASSWORD https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt`, you should use an API key for a more secure method of doing so, but just for the ease of use for anyone unfamiliar

7. While these two things are going on, you can use the time to go and upload your training/regularization data into some new subfolders in the project. Something like /Dreambooth-Stable-Diffusion/training or something

8. When the conda environment is set up, initialize conda with conda init bash, then reset the terminal with reset, or create a new terminal session

9. Navigate to the project directory again (Dreambooth-Stable...) if you aren't there already, and activate the environment with conda activate ldx < or whatever the environment was called

10. You should be ready to train! `python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume sd-v1-4-full-ema.ckpt -n <whatever you want this training to be called> --gpus 0, --data_root <the relative path to your training images> --reg_data_root <the relative path to your regularization images> --class_word <the word you used to get for regularization data>`. Note that this will map your training to the default `sks` prompt, so go change that in the personalized.py file if you want.

12. Let it roll, should be a bit over 1 it/s on 1 A6000.

13. After that's all done, all that's left is to download your model. I've run into issues with the vast.ai frontend and can't actually navigate into the /logs/xx/checkpoints folder, so if you hit this, try moving it out. In a terminal, cd into the checkpoints folder and do a mv final.ckpt .. to move it back a directory, so it should now be selectable and you can download it!

14. Alternatively, if you have a bad net connection, you could upload it to google drive to save some time running the instance - look into the 'gdrive' github repo to install this on the ubuntu CLI.
