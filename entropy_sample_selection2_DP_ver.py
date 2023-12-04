#!/usr/bin/env python3

"""
For post-processing method
- Uses DP
- with torch.no_grad() in sample selection

For in-processing method
- Saveable reservoir -> not yet..
- Uses DDP
- with torch.no_grad() in sample selection

Inspired by "Entropy-based Sample Selection for Online Continual Learning (2021)"
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9287846

calculates distances within random subset of the majority group
In order to measure distances,
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
import itertools
import random
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import pandas as pd
import gc
#from torch.nn.parallel import DistributedDataParallel as DDP

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
        datasets, ["id", "duration", "wav", "spk_id", "wrd", "age", "gender", "accents",
                   "sig", "tokens"],
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

        csv_writer.writerow(["ID"])
        
        
        final_dict = reservoir.group_dict
        
        for group in final_dict:
            for sample_object in final_dict[group].values():
                csv_writer.writerow(
                    [
                        sample_object.id,
                    ]
                )
    
    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    


        
#@dataclass        
class Sample:
    def __init__(self, id, duration, age, gender, accents, feats, measure_M=None, distance=None, similarity=None):
        self.id = id
        self.duration = duration
        self.age = age
        self.gender = gender
        self.accents = accents
        self.feats = feats
        
        if measure_M is not None and distance is not None and similarity is not None:
            self.measure_M = measure_M
            self.distance = distance
            self.similarity = similarity
        else:    
            self.measure_M = 0
            self.distance = 0
            self.similarity = 0
    
    def return_list(self):
        return_list = [self.id, self.duration, self.age, self.gender, self.accents, self.feats, self.measure_M, self.distance, self.similarity]
        return return_list
        
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

@sb.utils.checkpoints.register_checkpoint_hooks
class Reservoir:
    def __init__(self, size, attribute, cardinality=None):
        
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
        self.current_total_samples = 0
        self.init_phase = True
        self.group_dict = {i:dict() for i in self.groups} # 각 group의 Sample 객체 dict를 저장
        self.max_M_sample = None
        self.majority_group = None
        self.second_majority_group = None
    
    def load_group_dict(self, flattened_group_dict):
        nested_group_dict = convert_nested(flattened_group_dict)
        
        for group, group_sample_dict in nested_group_dict.items():
            for key, value in group_sample_dict.items():
                self.group_dict[group][key] = Sample(*value)
                
    def reinit(self):
        self.count_k_i = {i:0 for i in self.groups}
        self.current_total_samples = 0
        self.init_phase = True
        self.group_dict = {i:dict() for i in self.groups}
        self.max_M_sample = None
        self.majority_group = None
        self.second_majority_group = None
        
    def find_majority_group(self):
        self.majority_group = max(self.count_k_i, key=self.count_k_i.get) # max값 갖는 group중 dict 의 앞부분에 있는 group 선택
        return self.majority_group
    
    def find_second_majority_group(self):
        
        max1 = max(self.count_k_i.values())
        max2 = 0
        for v in self.count_k_i.values():
            if(v>max2 and v<max1):
                    max2 = v
        self.second_majority_group = [k for k,v in self.count_k_i.items() if v == max2][0]
        
        return self.second_majority_group
    
    def update_current_total_samples(self):
        total_samples = 0
        for value in self.count_k_i.values():
            total_samples += value
        self.current_total_samples = total_samples
        
    def update_phase(self):
        if self.init_phase and (self.current_total_samples == self.size):
            self.init_phase = False
        
    def update_count_k_i(self):
        """updates count_k_i & updates majority_group"""
        for key, value in self.group_dict.items():
            self.count_k_i[key] = len([item for item in value if item])
        self.find_majority_group()
        self.find_second_majority_group()
        self.update_current_total_samples()
        self.update_phase()
        wandb_log(self)
        
    def delete_sample(self, group, i_d):
        self.group_dict[group].pop(i_d)
        self.update_count_k_i()
        gc.collect()
    
    def delete_samples(self, group, i_ds):
        for i_d in i_ds:
            self.group_dict[group].pop(i_d)
        self.update_count_k_i()
        gc.collect()
    
    def add_sample(self, group, i_d, sample_object, update=True):
        self.group_dict[group][i_d] = sample_object
        if update:
            self.update_count_k_i()
        
        
    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        group_dict = {i:dict() for i in self.groups}
        
        for group, group_sample_dict in self.group_dict.items():
            for key, value in group_sample_dict.items():
                group_dict[group][key] = value.return_list()
        
        #flattened_group_dict = pd.json_normalize(group_dict, sep='#').to_dict(orient='records')[0]
        
        save_dict = {
            "count_k_i": self.count_k_i,
            "init_phase": self.init_phase,
            #"group_dict": group_dict,
            "max_M_sample": self.max_M_sample,
            "majority_group": self.majority_group,
            "second_majority_group": self.second_majority_group,
        }
        
        #print(save_dict)
        
        
        with open(path, "w") as w:
            w.write(yaml.dump(save_dict, sort_keys=False)) # 여기서 group_dict를 저장을 못한다..
            
        #print("yaml written!!")
        
                

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        with open(path) as f:
            save_dict = yaml.safe_load(f)
        self.count_k_i = save_dict["count_k_i"]
        self.init_phase = save_dict["init_phase"]
        #self.load_group_dict(save_dict["group_dict"])
        self.max_M_sample = save_dict["max_M_sample"]
        self.majority_group = save_dict["majority_group"]
        self.second_majority_group = save_dict["second_majority_group"]
        
        
        

def insert(dct, lst):
    for x in lst[:-2]:
        dct[x] = dct = dct.get(x, dict())
    dct.update({lst[-2]: lst[-1]})
    
def convert_nested(dct):
    # empty dict to store the result
    result = dict()
 
    # create an iterator of lists 
    # representing nested or hierarchical flow
    lists = ([*k.split("#"), v] for k, v in dct.items())
 
    # insert each list into the result
    for lst in lists:
        insert(result, lst)
    return result

    
def append_batch_to_group_dict(times, batch, reservoir, asr, attribute, init, n_diff=None):
        
    prev_times = times
    
    batch = batch.to(asr.device)
    batch_size = len(batch.id)
    
    wavs, wav_lens = batch.sig
    wavs = wavs.to(asr.device)
    wavs_lens = wav_lens.to(wavs.device)
    
    i_d = batch.id
    duration = batch.duration
    age = batch.age
    gender = batch.gender
    accents = batch.accents
    
    tokens, tokens_lens = batch.tokens

    with torch.no_grad():
        # Forward pass
        feats = asr.modules.wav2vec2(wavs, wav_lens)
     
    if attribute == "age":
        for i in range(batch_size):
            if age[i] == '':
                continue
            elif ((not init) and times < prev_times + n_diff) or (init and times < reservoir.size):
                reservoir.group_dict[age[i]][i_d[i]] = Sample(i_d[i],
                                                            duration[i].item(), 
                                                            age[i], 
                                                            gender[i], 
                                                            accents[i], 
                                                            feats[i],
                                                            )
                times += 1
            elif ((not init) and times == prev_times + n_diff) or (init and times == reservoir.size):
                break
                
    elif attribute == "gender":
        for i in range(batch_size):
            if gender[i] == '':
                continue
            elif (not init) or (init and times < reservoir.size):
                reservoir.group_dict[gender[i]][i_d[i]] = Sample(i_d[i],
                                                            duration[i].item(), 
                                                            age[i], 
                                                            gender[i], 
                                                            accents[i], 
                                                            feats[i],
                                                            )
                times += 1
            elif times == reservoir.size:
                break
    reservoir.update_count_k_i()
    
    return times

# for in-processing sample selection
def append_batch_to_group_dict2(batch, feats, softmax, loss, asr):
    
    reservoir = asr.reservoir
    init = asr.reservoir.init_phase
    
    attribute = asr.attribute
    n_diff = asr.n_diff
    
    batch_size = len(batch.id)
    
    i_d = batch.id
    duration = batch.duration
    age = batch.age
    gender = batch.gender
    accents = batch.accents

    
    appended_num_list = list()
    
    times = 0
    
    if attribute == "age":
        for i in range(batch_size):
            if age[i] == '':
                continue
            elif ((not init) and times < n_diff) or (init and reservoir.current_total_samples < reservoir.size):
                sample_object = Sample(i_d[i],
                                    duration[i].item(), 
                                    age[i], 
                                    gender[i], 
                                    accents[i], 
                                    feats[i],
                                    )

                reservoir.add_sample(age[i], i_d[i], sample_object, True)
                times += 1
                appended_num_list.append(i)
            elif ((not init) and times == n_diff) or (init and reservoir.current_total_samples == reservoir.size):
                break
                
    elif attribute == "gender":
        for i in range(batch_size):
            if gender[i] == '':
                continue
            elif ((not init) and times < n_diff) or (init and reservoir.current_total_samples < reservoir.size):
                sample_object = Sample(i_d[i],
                                    duration[i].item(), 
                                    age[i], 
                                    gender[i], 
                                    accents[i], 
                                    feats[i],
                                    )

                reservoir.add_sample(age[i], i_d[i], sample_object, True)
                times += 1
                appended_num_list.append(i)
            elif ((not init) and times == n_diff) or (init and reservoir.current_total_samples == reservoir.size):
                break
    
    return times, appended_num_list

def init_reservoir(reservoir, asr, train_loader):
    
    size = reservoir.size
    attribute = reservoir.attribute
    
    print("\ninit_reservoir\n")
    
    times = 0
    
    while(True):
        batch = next(train_loader)
        
        times = append_batch_to_group_dict(times, batch, reservoir, asr, attribute, init=True)

        if times == size:
            break
        
    print("end of init_reservoir\n")
    
    return train_loader

def random_subset_of_dict(dictionary, n_size):
    
    new_dict = dictionary
    random_choices = random.sample(dictionary.keys(), k=n_size)

    for key in list(dictionary.keys()):
        if key not in random_choices:
            #new_dict.pop(key)
            del new_dict[key]
    
    return new_dict
    


def find_min_dist_sample_in_majority_group(reservoir, n_diff, asr):
    
    majority_group_dict = reservoir.group_dict[reservoir.majority_group]
    
    n_size = reservoir.count_k_i[reservoir.majority_group]
    
    if n_size > 1000:
        n_size = 1000
    
    majority_group_dict = random_subset_of_dict(majority_group_dict, n_size)
    
    cos_sim= torch.nn.CosineSimilarity(dim=1)
    cos_sim = nn.DataParallel(cos_sim)
    
    two_combis = list(itertools.combinations(majority_group_dict.values(), 2))
    
    set_length_not_matching = set()
    
    for set_ in two_combis:
                
        feature1 = set_[0].feats.squeeze().to(asr.device)
        feature2 = set_[1].feats.squeeze().to(asr.device)
        
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
         
        with torch.no_grad():
            distance = 1 - similarity
        set_[0].add_similarity(similarity)
        set_[1].add_similarity(similarity)
        set_[0].add_distance(distance)
        set_[1].add_distance(distance)
    
    id_list = list(majority_group_dict.keys())
    object_list = list(majority_group_dict.values())

    similarity_list = list()
    
    with torch.no_grad():
        for i in range(len(object_list)):
            if not isinstance(object_list[i].similarity, int):
                sim = torch.sum(object_list[i].similarity)
            else:
                sim = 0
            similarity_list.append(sim)
        
        sum_similarity = sum(similarity_list)
    
        if sum_similarity != 0:
            prob_list = [val / sum_similarity for val in similarity_list]
        else:
            print("\n\n\n sum_similarity == 0 -> random dropping \n\n\n")
            prob = 1 / len(majority_group_dict)
            prob_list = [prob] * len(majority_group_dict)

    prob_list_ = list()
    

    try:
        random_choices = list(WeightedRandomSampler(weights=prob_list,
                                                    num_samples=n_diff,
                                                    replacement=False))
    except:
        random_choices = np.random.choice(len(prob_list), n_diff, replace=False)

    selected_id = list()
    selected_object = list()
    
    for i in random_choices:
        selected_id.append(id_list[i])
        selected_object.append(object_list[i])

    
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
    

def find_min_dist_sample_in_majority_group2(reservoir, n_diff, asr):
    
    majority_group_dict = reservoir.group_dict[reservoir.majority_group]
    
    n_size = reservoir.count_k_i[reservoir.majority_group]
    
    if n_size > 1000:
        n_size = 1000
    
    majority_group_dict = random_subset_of_dict(majority_group_dict, n_size)
    
    cos_sim= torch.nn.CosineSimilarity(dim=1)
    
    two_combis = list(itertools.combinations(majority_group_dict.values(), 2))
    
    set_length_not_matching = set()
    
    for set_ in two_combis:
                
        feature1 = set_[0].feats.squeeze().to(asr.device)
        feature2 = set_[1].feats.squeeze().to(asr.device)
        
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
         
        with torch.no_grad():
            distance = 1 - similarity
        set_[0].add_similarity(similarity)
        set_[1].add_similarity(similarity)
        set_[0].add_distance(distance)
        set_[1].add_distance(distance)
    
    id_list = list(majority_group_dict.keys())
    object_list = list(majority_group_dict.values())

    similarity_list = list()
    
    with torch.no_grad():
        for i in range(len(object_list)):
            if not isinstance(object_list[i].similarity, int):
                sim = torch.sum(object_list[i].similarity)
            else:
                sim = 0
            similarity_list.append(sim)
        
        sum_similarity = sum(similarity_list)
    
        if sum_similarity != 0:
            prob_list = [val / sum_similarity for val in similarity_list]
        else:
            print("\n\n\n sum_similarity == 0 -> random dropping \n\n\n")
            prob = 1 / len(majority_group_dict)
            prob_list = [prob] * len(majority_group_dict)

    prob_list_ = list()
    
    try:
        random_choices = list(WeightedRandomSampler(weights=prob_list,
                                                    num_samples=n_diff,
                                                    replacement=False))
    except:
        random_choices = np.random.choice(len(prob_list), n_diff, replace=False)

    selected_id = list()
    selected_object = list()
    
    if len(random_choices) == 0:
        return None, None
    
    for i in random_choices:
        selected_id.append(id_list[i])
        selected_object.append(object_list[i])

    
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
    asr.modules.wav2vec2 = nn.DataParallel(asr.modules.wav2vec2)
    asr.modules.ctc_lin = nn.DataParallel(asr.modules.ctc_lin)
    
    asr.modules.eval()
    
    
    if asr.checkpointer is not None:
        asr.checkpointer.recover_if_possible(
            device=torch.device(asr.device)
        )
            
    
    reservoir = Reservoir(size, attribute)

    train_loader = iter(train_loader)
    

    
    train_loader = init_reservoir(reservoir, asr, train_loader)
    print(reservoir.count_k_i)
    
    times = size
    

    next_batch = next(train_loader)
    
    while next_batch is not None:

        n_diff = reservoir.count_k_i[reservoir.majority_group] - reservoir.count_k_i[reservoir.second_majority_group]
        
        if (n_diff == 0):
            min_dist_ids, min_dist_objects = find_min_dist_sample_in_majority_group(reservoir, 1, asr)
        else:
            min_dist_ids, min_dist_objects = find_min_dist_sample_in_majority_group(reservoir, n_diff, asr)
        
        if attribute == "age":
            min_dist_group = min_dist_objects[0].age
        elif attribute == "gender":
            min_dist_group = min_dist_objects[0].gender
            
        reservoir.delete_samples(min_dist_group, min_dist_ids)    
        
        prev_times = times
        
        while True:
            
            times = append_batch_to_group_dict(times, next_batch, reservoir, asr, attribute, 
                                                     init=False, n_diff=n_diff)
            try:
                next_batch = next(train_loader)    
            except StopIteration:
                next_batch = None
                print("reached the end of the dataloader")
                break
            
            if (times - prev_times) == n_diff:
                break
            

        
        if(times % 1000 == 0) or times < reservoir.size + 30:
            print(reservoir.count_k_i)
        
        
        if(times % 5000 == 0):
            dir_name, base_name = os.path.split(csv_file)
            without_ext, ext = base_name.split(".")
            
            csv_file_ = dir_name + "/" + without_ext + "_" + str(times) + "." + ext
            create_csv(csv_file_, reservoir)
            
    
    # Save the samples in the final reservoir in the csv file
    dir_name, base_name = os.path.split(csv_file)
    without_ext, ext = base_name.split(".")
    
    csv_file_ = dir_name + "/" + without_ext + "_FINAL." + ext
    create_csv(csv_file_, reservoir)
    
    print("Sample selection done.\n\nFinal Group Dictionary")
    print(reservoir.group_dict)
    
    return reservoir
    

def wandb_log(reservoir):
    #wandb.log({"female" : reservoir.count_k_i["male"]})
    #wandb.log({"male" : reservoir.count_k_i["female"]})
    #wandb.log({"other" : reservoir.count_k_i["other"]})
	
    wandb.log({"teens" : reservoir.count_k_i["teens"]})
    wandb.log({"twenties" : reservoir.count_k_i["twenties"]})
    wandb.log({"thrities" : reservoir.count_k_i["thirties"]})
    wandb.log({"fourties" : reservoir.count_k_i["fourties"]})
    wandb.log({"fifties" : reservoir.count_k_i["fifties"]})
    wandb.log({"sixties" : reservoir.count_k_i["sixties"]})
    wandb.log({"seventies" : reservoir.count_k_i["seventies"]})
    wandb.log({"eighties" : reservoir.count_k_i["eighties"]})
    wandb.log({"nineties" : reservoir.count_k_i["nineties"]})
    
    
if __name__ == "__main__":
    
    wandb.init(project='Entropy_sample_selection2 DP ver.')
    wandb.run.save()
    
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    args = {
        "seed": hparams["seed"],
        "batch_size": hparams["batch_size"],
        "num_workers": hparams["num_workers"]
    }
    wandb.config.update(args)

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
    attribute = hparams["attribute"]
    csv_file = hparams["selected_sample_csv"]
    
    
    entropy_based_data_selection(asr_brain, size, attribute, train_loader, csv_file)
    
    
    print("end of main")
    
    

    
    

    
    
    

    
    
