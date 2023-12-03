#!/usr/bin/env python3

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
from scipy.stats import norm
#from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

           
# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]
    
    coreset_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["coreset_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        coreset_data = coreset_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        coreset_data = coreset_data.filtered_sorted(
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

    datasets = [coreset_data, train_data, valid_data, test_data]

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
        "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "duration", "age", "gender", "sig", "tokens"],
    )
    return coreset_data, train_data, valid_data, test_data
    
    
    
def dataio_prepare2(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]
    
    coreset_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["coreset_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        coreset_data = coreset_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        coreset_data = coreset_data.filtered_sorted(
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
        
        
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv2"], replacements={"data_root": data_folder},
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

    datasets = [coreset_data, train_data, valid_data, test_data]

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
        "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "duration", "age", "gender", "sig", "tokens"],
    )
    return coreset_data, train_data, valid_data, test_data
    
    
def create_csv(csv_file, id_list):
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
        
        for id in id_list:
            csv_writer.writerow([id])
    
    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    


def cos_similarity(f1, f2, asr):
    cos_similarity = torch.nn.CosineSimilarity(dim=1)
    
    f1, f2 = f1.to(asr.device), f2.to(asr.device)
    f1_length = f1.shape[0]
    f2_length = f2.shape[0]
    
    if f1_length > f2_length:
        f1 = f1.narrow(0,0,f2_length)
    elif f1_length < f2_length:
        f2 = f2.narrow(0,0,f1_length)
    
    cos_sim = torch.mean(cos_similarity(f1, f2)).to(asr.device)

    return cos_sim
    
    
def calculate_minibatch_similarity(batch_grad, asr):
    similarity = torch.zeros(batch_grad.shape[0]).to(asr.device)
    batch_mean_grad = torch.mean(batch_grad, dim=0)
    for i in range(batch_grad.shape[0]):
        cos_sim = cos_similarity(batch_grad[i], batch_mean_grad, asr)
        similarity[i] = cos_sim
    return similarity


def calculate_sample_diversity(batch_grad, asr):
    diversity = torch.zeros(batch_grad.shape[0]).to(asr.device)
    N = batch_grad.shape[0]
    for i in range(batch_grad.shape[0]):
        for j in range(i+1,batch_grad.shape[0]):
            cos_sim = cos_similarity(batch_grad[i], batch_grad[j], asr)
            diversity[i] += cos_sim
            diversity[j] += cos_sim
    diversity *= (-1/(N-1))
    return diversity


def calculate_coreset_affinity(batch_grad, coreset_grad, asr):
    affinity = torch.zeros(batch_grad.shape[0]).to(asr.device)
    coreset_mean_grad = torch.mean(coreset_grad, dim=0)
    for i in range(batch_grad.shape[0]):
        cos_sim = cos_similarity(batch_grad[i], coreset_mean_grad, asr)
        affinity[i] = cos_sim
    return affinity

def calculate_duration_score(batch, asr):
    duration_score = torch.zeros(len(batch.id)).to(asr.device)
    duration = batch.duration
    for i in range(len(batch.id)):
        duration_prob = torch.Tensor([norm.cdf(x=duration[i].detach().cpu().numpy(), 
                                loc=asr.duration_mean, 
                                scale=asr.duration_std)]).to(asr.device)
        duration_score[i] = duration_prob
    return duration_score

def select_top_k(measure_M, k, batch):

    values, ids = measure_M.topk(k)
    top_k_ids = list()
    
    for i in ids:
        top_k_ids.append(batch.id[i])
    
    return top_k_ids


def calculate_grad(batch, asr):
    batch = batch.to(asr.device)
    
    wavs, wav_lens = batch.sig # is_leaf == True, requires_grad == False
    wavs, wav_lens = wavs.to(asr.device), wav_lens.to(asr.device)


    # Forward pass
    feats = asr.modules.wav2vec2(wavs, wav_lens)
    feats.requires_grad_()
    logits = asr.modules.ctc_lin(feats)
    p_ctc = asr.hparams.log_softmax(logits)
    
    tokens, tokens_lens = batch.tokens
    tokens, tokens_lens = tokens.to(asr.device), tokens_lens.to(asr.device)

    losses = asr.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
    loss = losses.mean()
    
    feats.retain_grad()
    loss.backward()
    
    return feats.grad


def find_coreset_candidates_for_batch(asr, batch, B_C, k=None, final=False):
    batch_grad = calculate_grad(batch, asr)
    coreset_grad = calculate_grad(B_C, asr)
    
    minibatch_similarity = calculate_minibatch_similarity(batch_grad, asr)
    sample_diversity = calculate_sample_diversity(batch_grad, asr)
    coreset_affinity = calculate_coreset_affinity(batch_grad, coreset_grad, asr)
    duration_score = calculate_duration_score(batch, asr)
    
    if not final:
        measure_M = minibatch_similarity + sample_diversity + asr.tau * coreset_affinity + asr.duration_coef * duration_score
    else:
        measure_M = minibatch_similarity + sample_diversity + asr.tau_star * coreset_affinity + asr.duration_coef * duration_score
    
    top_2_ids = select_top_k(measure_M, 2, batch)
    if k is not None:
        top_k_ids = select_top_k(measure_M, k, batch)
        return top_2_ids, top_k_ids
    else:
        return top_2_ids
    


def add_to_coreset_candidate(asr, top_k_ids):
    if asr.coreset_candidate is None:
        asr.coreset_candidate = top_k_ids
    else:
        for id in top_k_ids:
            asr.coreset_candidate.append(id)
            
                        
def concat_batch(batch, top_k_ids, B_C, asr):
    
    #ids = batch.id
    # ["id", "duration", "age", "gender", "sig", "tokens"],
    for i in range(len(batch.id)):
        if batch.id[i] in top_k_ids:
            B_C.id.append(batch.id[i])
            B_C.duration = torch.cat([B_C.duration, batch.duration[i].reshape(1)])
            B_C.age.append(batch.age[i])
            B_C.gender.append(batch.gender[i])
                        
            # concat PaddedBatch with right padding
            if B_C.sig.data.shape[1] < batch.sig.data[i].shape[0]:
                B_C.sig = B_C.sig._replace(data = torch.nn.functional.pad(B_C.sig.data, (0, batch.sig.data[i].shape[0]-B_C.sig.data.shape[1])).to(asr.device))
                B_C.sig = B_C.sig._replace(data = torch.cat([B_C.sig.data, torch.unsqueeze(batch.sig.data[i], 0).to(asr.device)]))
            elif B_C.sig.data.shape[1] > batch.sig.data[i].shape[0]:
                B_C.sig = B_C.sig._replace(data = torch.cat([B_C.sig.data, torch.unsqueeze(torch.cat([batch.sig.data[i], torch.zeros(B_C.sig.data.shape[1]-batch.sig.data[i].shape[0]).to(asr.device)]), 0)]))
            else:
                B_C.sig = B_C.sig._replace(data = torch.cat([B_C.sig.data, torch.unsqueeze(batch.sig.data[i], 0).to(asr.device)]))
            B_C.sig = B_C.sig._replace(lengths = torch.cat([B_C.sig.lengths, torch.unsqueeze(batch.sig.lengths[i], 0).to(asr.device)]))
            
            if B_C.tokens.data.shape[1] < batch.tokens.data[i].shape[0]:
                B_C.tokens = B_C.tokens._replace(data = torch.nn.functional.pad(B_C.tokens.data, (0, batch.tokens.data[i].shape[0]-B_C.tokens.data.shape[1])))
                B_C.tokens = B_C.tokens._replace(data = torch.cat([B_C.tokens.data, torch.unsqueeze(batch.tokens.data[i], 0).to(asr.device)]))
            elif B_C.tokens.data.shape[1] > batch.tokens.data[i].shape[0]:
                B_C.tokens = B_C.tokens._replace(data = torch.cat([B_C.tokens.data, torch.unsqueeze(torch.cat([batch.tokens.data[i], torch.zeros(B_C.tokens.data.shape[1]-batch.tokens.data[i].shape[0]).to(asr.device)]), 0).to(asr.device)]))
            else:
                B_C.tokens = B_C.tokens._replace(data = torch.cat([B_C.tokens.data, torch.unsqueeze(batch.tokens.data[i], 0).to(asr.device)]))
            B_C.tokens = B_C.tokens._replace(lengths = torch.cat([B_C.tokens.lengths, torch.unsqueeze(batch.tokens.lengths[i], 0).to(asr.device)]))
            
    return B_C

    
def add_to_real_coreset(asr, top_k_ids):
    if asr.real_coreset is None:
        asr.real_coreset = top_k_ids
    elif len(asr.real_coreset) < asr.hparams.final_k : # 10000
        for id in top_k_ids:
            asr.real_coreset.append(id)
    else:
        return False
    return True
            
def select_coreset_from_candidates(asr):
    
    batch = next(asr.train_loader) # 64
    B_C = next(asr.coreset_loader2) # 12
    
    while True:        
        #find_coreset_candidates_for_batch(asr, batch, B_C, k=None, final=False)
        top_2_ids = find_coreset_candidates_for_batch(asr, batch, B_C, final=True) # select 2 out of 32 (batch_size)

        is_added = add_to_real_coreset(asr, top_2_ids)
        
        if not is_added:
            break
    
        try:
            batch = next(asr.train_loader)
            B_C = next(asr.coreset_loader2)
        except StopIteration:
            batch = None
            print("reached the end of the dataloader")
            break
    
    # save real_coreset to csv

    dir_name, base_name = os.path.split(asr.csv_file)
    without_ext, ext = base_name.split(".")
    
    csv_file_ = dir_name + "/" + without_ext + "_FINAL_REAL_CORESET." + ext
    create_csv(csv_file_, asr.real_coreset)
    
