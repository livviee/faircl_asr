# Fair Continual Learning for ASR Project


## How to run
1. Clone this repository.
2. Create a docker container with docker image `cloud9sm/fair_cl:latest`
3. Make sure CommonVoice dataset is mounted to your docker container
4. Make sure you have `train.csv`, `test.csv`, `dev.csv` in your `save_folder` in order to avoid data preprocessing
5. Make sure hyperparameters in yaml file is well adjusted to your setting
6. Check the availability of four GPUs
7. Run the command:
   
`CUDA_VISIBLE_DEVICES=[],[],[],[] python3 train_final.py hparams/OF_YOUR_SETTING.yaml --data_parallel_backend --tqdm_colored_bar --grad_accumulation_factor 4`
   
## Hparams
### Training Settings

1. multi-lingual model : train with `1_train_en+de.yaml`
2. fine-tuned model : first train en model with `2_train_en_first.yaml` and then fine-tune with de using `3_train_de_second.yaml`
3. naive memory replay model : first train en model with `2_train_en_first.yaml` and then train it with de + 1 percent of en using `4_train_de_second_with_naive_mem_replay.yaml`

### What you should care / modify
- You should change `!PLACEHOLDER` in regard to your own setting.
  - Set `output_folder` and `data_folder` to of your own corresponding directory.
- Preprocessed `train.csv`, `test.csv`, `dev.csv` should be in your `save_folder` before start training in order to avoid data_preprocessing


## train_final.py
### What you should modify
- If you are going to use wandb, uncomment wandb command lines 
