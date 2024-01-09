set -x
set -e
mkdir pretrained_models
cd pretrained_models

mkdir stable-diffusion-v1-5
cd stable-diffusion-v1-5
wget -O v1-5-pruned-emaonly.safetensors https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors?download=true
mkdir scheduler text_encoder tokenizer unet
cd scheduler
wget -O scheduler_config.json https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/scheduler/scheduler_config.json?download=true
cd ../text_encoder
wget -O config.json https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json?download=true
wget -O pytorch_model.bin https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin?download=true
cd ../tokenizer
wget -O merges.txt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt?download=true
wget -O special_tokens_map.json https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/special_tokens_map.json?download=true
wget -O tokenizer_config.json https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_config.json?download=true
wget -O vocab.json https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json?download=true
cd ../unet
wget -O diffusion_pytorch_model.bin https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true
wget -O config.json https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json?download=true
cd ..
cd ..

mkdir sd-vae-ft-mse
cd sd-vae-ft-mse
wget -O config.json https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true
wget -O diffusion_pytorch_model.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors?download=true
cd ..

git lfs install
git clone https://huggingface.co/zcxu-eric/MagicAnimate
