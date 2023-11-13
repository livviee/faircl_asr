# Fair Continual Learning of ASR Project

1. Clone this repository.
2. Create a docker container with docker image "cloud9sm/fair_cl:latest"
3. Make sure CommonVoice dataset is mounted to your docker container
4. Make sure you have train.csv, test.csv, dev.csv in your save_folder in order to avoid data preprocessing
5. Make sure hyperparameters in yaml file is well adjusted to your setting
6. Check the availability of four GPUs
7. Run the command: CUDA_VISIBLE_DEVICES=[],[],[],[] python3 train_final.py hparams/OF_YOUR_SETTING.yaml --data_parallel_backend --tqdm_colored_bar --grad_accumulation_factor 4
   
