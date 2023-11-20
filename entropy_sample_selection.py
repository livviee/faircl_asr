#!/usr/bin/env python3
"""
Implementation of "Entropy-based Sample Selection for Online Continual Learning (2021)"
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9287846

In order to find the minimum distance feature,
cosine similarity of features was used instead of measuring direct distances.

"""
import sys
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from mySentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main, if_main_process
import warnings
import yaml
import sentencepiece as spm
import wandb
from mySchedulers import MyIntervalScheduler
from train_final import ASR

import csv
from torch.utils.data import DataLoader
#from speechbrain.dataio.dataloader import LoopedLoader
#from speechbrain.dataio.dataset import DynamicItemDataset
import itertools
import random
import os

logger = logging.getLogger(__name__)

           
# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        #csv: "ID", "duration", "wav-경로", "spk_id", "wrd", "age", "gender", "accents"
        datasets, ["id", "duration", "wav", "spk_id", "wrd", "age", "gender", "accents",
                   "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data
    
    
    
    
def create_csv(csv_file, reservoir):
    """    
    csv_file : str
        new csv file name 
    """

    # Stream into a .tmp file, and rename it to the real path at the end.
    csv_file_tmp = csv_file + ".tmp"

    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        #csv_writer.writerow(["ID", "wav", "spk_id", "wrd", "age", "gender", "accents"])
        csv_writer.writerow(["ID", "duration", "wav", "spk_id", "wrd", "age", "gender", "accents"])
        
        
        final_dict = reservoir.group_dict
        
        for group in final_dict:
            for sample_object in final_dict[group].values():
                csv_writer.writerow(
                    [
                        sample_object.id,
                        sample_object.duration,
                        sample_object.wav,
                        sample_object.spk_id,
                        sample_object.wrd,
                        sample_object.age,
                        sample_object.gender,
                        sample_object.accents
                    ]
                )
    
    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    


        
#@dataclass        
class Sample:
    def __init__(self, id, duration, wav, spk_id, wrd, age, gender, accents, sig, tokens_bos, tokens_eos, tokens, feats, softmax, loss):
        self.id = id
        self.duration = duration
        self.wav = wav # path
        self.spk_id = spk_id
        self.wrd = wrd
        self.age = age
        self.gender = gender
        self.accents = accents
        self.sig = sig
        self.tokens_bos = tokens_bos
        self.tokens_eos = tokens_eos
        self.tokens = tokens
        
        self.feats = feats
        self.softmax = softmax
        self.loss = loss
        self.measure_M = 0
        self.distance = 0
        self.similarity = 0
        
    def add_distance(self, distance_val):
        self.distance += distance_val
        
    def add_similarity(self, similarity_val):
        self.similarity += similarity_val
        
        
def dict_to_string(dictionary):
    string_representation = ""
    for key, value in dictionary.items():
        string_representation += str(key) + ": " + str(value) + ", "
    string_representation = string_representation.rstrip(", ")
    return string_representation

class Reservoir:
    def __init__(self, size, attribute, cardinality=None, ):
        
        if attribute == "age":
            self.groups = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]
        elif attribute == "gender":
            self.groups = ["female", "male", "other"]
            
        self.attribute = attribute    
        self.count_k_i = {i:0 for i in self.groups}
        
        if not cardinality == None:
            self.cardinality = cardinality # dictionary
        else:
            self.cardinality = None
            
        self.size = size
        self.group_dict = {i:dict() for i in self.groups} # 각 group의 Sample 객체 dict를 저장
        self.max_M_sample = None
        self.majority_group = None
        
    
    def find_majority_group(self):
        self.majority_group = max(self.count_k_i, key=self.count_k_i.get)
        return self.majority_group
    
    def update_count_k_i(self):
        """updates count_k_i & updates majority_group"""
        for key, value in self.group_dict.items():
            self.count_k_i[key] = len([item for item in value if item])
        self.find_majority_group()
        
    def delete_sample(self, group, i_d):
        self.group_dict[group].pop(i_d)
        self.update_count_k_i()
    
    def add_sample(self, group, i_d, sample_object):
        self.group_dict[group][i_d] = sample_object
        self.update_count_k_i()
        
    def __str__(self):
        info = "attribute: " + self.attribute +\
                "\ncount_k_i: " + self.count_k_i +\
                "\ncardinality: " + self.cardinality +\
                "\nsize (current saved samples): " + self.size +\
                "\ngroup_dict: " + dict_to_string(self.group_dict) +\
                "\nmax_M_sample: " + dict_to_string(self.max_M_sample)
        return info
        

def make_sample_object(batch, asr):
    batch = batch.to(asr.device)
    
    wavs, wav_lens = batch.sig
    wavs, wav_lens = wavs.to(asr.device), wav_lens.to(asr.device)
        
    i_d = batch.id[0]
    duration = batch.duration[0]
    path = batch.wav[0]
    spk_id = batch.spk_id[0]
    wrd = batch.wrd[0]
    age = batch.age[0]
    gender = batch.gender[0]
    accents = batch.accents[0]
    tokens_bos, _ = batch.tokens_bos
    tokens_eos, _ = batch.tokens_eos
    tokens, tokens_lens = batch.tokens
    
    # Forward pass
    feats = asr.modules.wav2vec2(wavs, wav_lens)
    logits = asr.modules.ctc_lin(feats)
    softmax = asr.hparams.log_softmax(logits) # p_ctc

    # Evaluate
    loss = asr.hparams.ctc_cost(softmax, tokens, wav_lens, tokens_lens)

    return Sample(i_d, 
                duration, 
                path, 
                spk_id, 
                wrd, 
                age, 
                gender, 
                accents, 
                batch.sig, 
                tokens_bos, 
                tokens_eos, 
                batch.tokens,
                feats,
                softmax,
                loss)
    


    
def append_batch_to_group_dict(batch, reservoir, asr, attribute):
    
    batch = batch.to(asr.device)
    
    wavs, wav_lens = batch.sig
    wavs, wav_lens = wavs.to(asr.device), wav_lens.to(asr.device)
        
    i_d = batch.id[0]
    duration = batch.duration[0].item()
    path = batch.wav[0]
    spk_id = batch.spk_id[0]
    wrd = batch.wrd[0]
    age = batch.age[0]
    gender = batch.gender[0]
    accents = batch.accents[0]
    tokens_bos, _ = batch.tokens_bos
    tokens_eos, _ = batch.tokens_eos
    tokens, tokens_lens = batch.tokens
    
    # Forward pass
    feats = asr.modules.wav2vec2(wavs, wav_lens)
    logits = asr.modules.ctc_lin(feats)
    softmax = asr.hparams.log_softmax(logits) # p_ctc

    # Evaluate
    loss = asr.hparams.ctc_cost(softmax, tokens, wav_lens, tokens_lens)

    
    # id, duration, wav, spk_id, wrd, age, gender, accents, sig, tokens_bos, tokens_eos, tokens
    
    if attribute == "age":
        reservoir.group_dict[age][i_d] = Sample(i_d, 
                                                duration, 
                                                path, 
                                                spk_id, 
                                                wrd, 
                                                age, 
                                                gender, 
                                                accents, 
                                                batch.sig, 
                                                tokens_bos, 
                                                tokens_eos, 
                                                batch.tokens,
                                                feats,
                                                softmax,
                                                loss)
    elif attribute == "gender":
        reservoir.group_dict[gender][i_d] = Sample(i_d, 
                                                    duration, 
                                                    path, 
                                                    spk_id, 
                                                    wrd, 
                                                    age, 
                                                    gender, 
                                                    accents, 
                                                    batch.sig, 
                                                    tokens_bos, 
                                                    tokens_eos, 
                                                    batch.tokens,
                                                    feats,
                                                    softmax,
                                                    loss)
    reservoir.update_count_k_i()
    
def init_reservoir(reservoir, asr, train_loader):
    
    size = reservoir.size
    attribute = reservoir.attribute
    
    print("\ninit_reservoir\n")
    
    for i in range(size):
        batch = next(train_loader)
        
        if attribute == "age":
            next_group = batch.age[0]
        elif attribute == "gender":
            next_group = batch.gender[0]
            
        if(next_group != ''):
            append_batch_to_group_dict(batch, reservoir, asr, attribute)
            #print(str(i+1)+"th append ", batch.id[0])    
    
    print("end of init_reservoir\n")
    
    return train_loader


def find_min_dist_sample_in_majority_group(reservoir):
    
    majority_group_dict = reservoir.group_dict[reservoir.majority_group]
    cos_sim= torch.nn.CosineSimilarity(dim=1)
    
    two_combis = list(itertools.combinations(majority_group_dict.values(), 2))
    
    for set_ in two_combis:
                
        feature1 = set_[0].feats.squeeze()
        feature2 = set_[1].feats.squeeze()
        
        feature1_length = feature1.shape[0]
        feature2_length = feature2.shape[0]

        if feature1_length > feature2_length:
            feature1 = feature1.narrow(0,0,feature2_length)
        elif feature1_length < feature2_length:
            feature2 = feature2.narrow(0,0,feature1_length)
        
        similarity = cos_sim(feature1, feature2)
        

        if not isinstance(set_[0].similarity, int) and not isinstance(similarity, int):
            if (set_[0].similarity.shape[0] > similarity.shape[0]):
                set_[0].similarity = set_[0].similarity[:similarity.shape[0]]
                set_[0].distance = set_[0].distance[:similarity.shape[0]]
            elif (set_[0].similarity.shape[0] < similarity.shape[0]):
                similarity = similarity[:set_[0].similarity.shape[0]]
         
        
        distance = 1 - similarity
        set_[0].add_similarity(similarity)
        set_[1].add_similarity(similarity)
        set_[0].add_distance(distance)
        set_[1].add_distance(distance)
    
    id_list = list(majority_group_dict.keys())
    object_list = list(majority_group_dict.values())

    similarity_list = list()
    
    for i in range(len(object_list)):
        if not isinstance(object_list[i].similarity, int):
            sim = torch.sum(object_list[i].similarity)
        else:
            sim = 0
        similarity_list.append(sim)
    
    sum_similarity = sum(similarity_list)
    #print(sum_similarity) #tensor(7314.4727, device='cuda:0', grad_fn=<AddBackward0>)
    
    
    if sum_similarity != 0:
        prob_list = [val / sum_similarity for val in similarity_list]
    else:
        print("\n\n\n sum_similarity == 0 -> random dropping \n\n\n")
        prob = 1 / len(majority_group_dict)
        prob_list = [prob] * len(majority_group_dict)
    #print(prob_list)
    
    random_choice = random.choices(population=range(len(id_list)),
                                weights=prob_list,
                                k=1)[0]
    selected_id = id_list[random_choice]
    selected_object = object_list[random_choice]
    
    """
    # find sample with minimum distance : no randomness
    distance_list = list()
    #dist_list = list() # for debugging

        
    for i in range(len(object_list)):
        torch.sum(1-object_list[i].distance)
        distance_list.append()
        #dist_list.append(torch.sum(object_list[i].distance).detach().cpu()) # for debugging
    
    
    #print("distance list")
    #print(dist_list)
    #print("\n\n")
    
    min_idx = distance_list.index(min(distance_list))
    min_dist_id = id_list[min_idx]
    
    #print("\nmin dist index: ", str(min_idx), " id: ", min_dist_id)
    """
    
    # reset distances
    for sample_object in object_list:
        sample_object.distance = 0
        
    # reset similarites
    for sample_object in object_list:
        sample_object.similarity = 0
    
    
    return selected_id, selected_object
    
    
def entropy_based_data_selection(asr, size, attribute, train_loader, csv_file):
    """
    size: Reservoir size
    attribute: age / gender
    
    """
    
    reservoir = Reservoir(size, attribute)
    train_loader = iter(train_loader)
    
    train_loader = init_reservoir(reservoir, asr, train_loader)
    
    next_batch = next(train_loader)
    

    times = 0
    while next_batch is not None:
        
        if attribute == "age":
            next_group = next_batch.age[0]
        elif attribute == "gender":
            next_group = next_batch.gender[0]
            
        if(next_group != ''):
            min_dist_id, min_dist_object = find_min_dist_sample_in_majority_group(reservoir)
            
            if attribute == "age":
                min_dist_group = min_dist_object.age
            elif attribute == "gender":
                min_dist_group = min_dist_object.gender
            
            # replace samples
            append_batch_to_group_dict(next_batch, reservoir, asr, attribute)
            reservoir.delete_sample(min_dist_group, min_dist_id)
        
        next_batch = next(train_loader)
        
        times+=1
        if(times % 1000 == 0) or times < 30:
            #print(reservoir.group_dict)
            print(reservoir.count_k_i)
        
        
        if(times % 5000 == 0):
            dir_name, base_name = os.path.split(csv_file)
            without_ext, ext = base_name.split(".")
            
            csv_file_ = dir_name + "/" + without_ext + "_" + str(times) + "." + ext
            create_csv(csv_file_, reservoir)
            
    
    print("Sample selection done.\n\nFinal Group Dictionary")
    print(reservoir.group_dict)
    
    return reservoir
    



if __name__ == "__main__":
    
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    args = {
        "seed": hparams["seed"],
        "peak_lr": hparams["peak_lr"],
        "epochs": hparams["number_of_epochs"],
        "batch_size": hparams["batch_size"],
        "num_workers": hparams["num_workers"]
    }
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    

    
    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"], # 1
		eos_id=hparams["eos_index"], # 2
		pad_id=hparams["pad_index"], # 3
		unk_id=hparams["unk_index"], # 4
        bos_piece=hparams["bos_piece"], # <bos>
		eos_piece=hparams["eos_piece"], # <eos>
    )
   
    # Defining scheduler 
    lr_annealing_model = MyIntervalScheduler(lr_initial = hparams["peak_lr"],
                                            n_warmup_steps = hparams["tenth_step"],
                                            anneal_steps = hparams["half_step"],
                                            anneal_rates = hparams["anneal_rate"])
    
    
    # Create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)


    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer
    asr_brain.lr_annealing_model = lr_annealing_model
        
    train_loader = asr_brain.make_dataloader(train_data, 
                                             stage=sb.Stage.TRAIN, 
                                             **hparams["dataloader_options"])
    
    
    
    size = 10000
    attribute = "age"
    csv_file = hparams["selected_sample_csv"]
    
    
    entropy_based_data_selection(asr_brain, size, attribute, train_loader, csv_file)
    
    
    print("end of main")
    
    

    
    
