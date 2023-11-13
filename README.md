# Fair Continual Learning of ASR Project


## How to run
1. Clone this repository.
2. Create a docker container with docker image "cloud9sm/fair_cl:latest"
3. Make sure CommonVoice dataset is mounted to your docker container
4. Make sure you have train.csv, test.csv, dev.csv in your save_folder in order to avoid data preprocessing
5. Make sure hyperparameters in yaml file is well adjusted to your setting
6. Check the availability of four GPUs
7. Run the command: `CUDA_VISIBLE_DEVICES=[],[],[],[] python3 train_final.py hparams/OF_YOUR_SETTING.yaml --data_parallel_backend --tqdm_colored_bar --grad_accumulation_factor 4`
   
## Hparams
### Train Settings

1. multi-lingual model : "1_train_en+de.yaml"
2. fine-tuned model : first train en model with "2_train_en_first.yaml" and then fine-tune with "3_train_de_second.yaml"
3. naive memory replay model : first train en model with "2_train_en_first.yaml" and then train with "4_train_de_second_with_naive_mem_replay.yaml"

### What you should care / modify
- You should change <!PLACEHOLDER> in regard to your own setting.
- Check output_folder is set right
- You should have the preprocessed train.csv, test.csv, dev.csv in your save_folder before start training in order to avoid data_preprocessing
