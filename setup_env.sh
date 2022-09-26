mkdir weights && cd ./weights
wget --http-user=<USERNAME> --http-password=<PASSWORD> https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
cd ..
apt -y install gcc unzip zip
conda env create -f environment.yaml
conda init bash
reset
exec bash
