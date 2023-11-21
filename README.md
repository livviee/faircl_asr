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

  

\
## Sample Selection Methods for Memory Replay
### Entropy-based Sample Selection
- Implementation of Entropy-based Sample Selection inspired by: `Entropy-based Sample Selection for Online Continual Learning (2021)`
   (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9287846)
- `entropy_sample_selection.py` : indirectly measures distances of features with cosine similarities.
- `entropy_sample_selection2.py` : faster version of above by removing and adding samples in bulk.
- corresponding hparam file : `5_entropy_based_sample_selection.yaml`

### Information-theoretic Sample Selection
- Implementation of Information-theoretic Sample Selection inspired by: `Information-Theoretic Online Memory Selection for Continual Learning (2022)`
  (https://arxiv.org/pdf/2204.04763.pdf)
- `info_theory_sample_selection.py` : threshold-based method. measures criteria M with surprise and learnability metric.
- corresponding hparam file : `6_info_theory_sample_selection.yaml`

  


