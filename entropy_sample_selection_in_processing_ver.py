#!/usr/bin/env python3

import os
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
from entropy_sample_selection2_DP_ver import *
from speechbrain.core import *

logger = logging.getLogger(__name__)


# Define training procedure

class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
    
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.modules.wav2vec2(wavs, wav_lens)
        logits = self.modules.ctc_lin(feats) # x
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens, feats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens, feats = predictions
        
        ids = batch.id
        tokens, tokens_lens = batch.tokens
        tokens, tokens_lens = tokens.to(self.device), tokens_lens.to(self.device)

        losses = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = losses.mean()
        
        if stage != sb.Stage.TRAIN:
            # Decode token terms to words

            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=-1
            )
            
            """
            # Beam Search Decoding
                
            p_ctc = p_ctc.detach().cpu().numpy()
            sequence = self.beam_search_decoder.decode_batch(pool=None, logits_list=p_ctc,
                                                             beam_width=self.hparams.beam_size)
            # pool: multiprocessing pool for parallel execution

            """
            
            predicted_words = self.tokenizer(sequence, task="decode_from_list")

            
            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")
            
            
            
            # Print at Validation stage
            
            print("target / greedy predicted words:\n")
            for i in range(2):
                print(target_words[i])
                print(predicted_words[i])
                print("\n\n")
            

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return p_ctc, feats, losses, loss
    
    
    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        
        
        should_step = self.step % self.grad_accumulation_factor == 0
        
        is_freeze_step = self.optimizer_step <= self.hparams.freeze_steps
        
        # Managing automatic mixed precision
        # TOFIX: CTC fine-tuning currently is unstable
        # This is certainly due to CTC being done in fp16 instead of fp32
            
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                p_ctc, feats, losses, loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.wav2vec2.freeze:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    if not self.hparams.wav2vec2.freeze:
                        if self.optimizer_step >= self.hparams.warmup_steps:
                            self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                
                
                
                
        else:
            # This is mandatory because HF models have a weird behavior with DDP
            # on the forward pass
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            p_ctc, feats, losses, loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)


            if self.sample_selection:
                # Entropy-based sample selection
                with torch.no_grad():
                    if not self.reservoir.init_phase and self.n_diff is None:
                        self.n_diff = self.reservoir.count_k_i[self.reservoir.majority_group] - self.reservoir.count_k_i[self.reservoir.second_majority_group]

                        if (self.n_diff == 0):
                            min_dist_ids, min_dist_objects = find_min_dist_sample_in_majority_group2(self.reservoir, 1, self)
                        else:
                            min_dist_ids, min_dist_objects = find_min_dist_sample_in_majority_group2(self.reservoir, self.n_diff, self)
                        
                        if self.attribute == "age":
                            min_dist_group = min_dist_objects[0].age
                        elif self.attribute == "gender":
                            min_dist_group = min_dist_objects[0].gender
                            
                        self.reservoir.delete_samples(min_dist_group, min_dist_ids)
                    
                    times, appended_num_list = append_batch_to_group_dict2(batch, feats, p_ctc, losses, self) # self.attribute, reservoir, asr
                    
                    if not self.reservoir.init_phase: 
                        if times == self.n_diff:
                            self.n_diff = None
                        elif self.n_diff is not None:               
                            self.n_diff -= times
                    
                weight = torch.ones(len(batch.id)).to(self.device)
                for i in range(len(batch.id)):
                    if i in appended_num_list:
                        weight[i] = (1+self.lambda_star)
                losses = losses * weight
                loss = torch.mean(losses)
                
                        
            wandb.log({"Training loss": loss})
            
            
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
                
                
            if should_step: ## accumulation done
                if self.check_gradients(loss):
                    if is_freeze_step:
                        self.modules.wav2vec2.freeze = True
                        
                        for param in self.modules["wav2vec2"].parameters():
                            param.requires_grad = False

                    else:
                        self.modules.wav2vec2.freeze = False
                        
                        for param in self.modules["wav2vec2"].parameters():
                            param.requires_grad = True
                    
                    self.model_optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
        
        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

            
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``, meant for calculating and logging metrics.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        
        # after each step (after accumulated enough gradient and finally updated optimizer)
        if should_step:
            
            old_lr, new_lr = self.lr_annealing_model(
                    self.optimizer_step, self.model_optimizer
                )

            sb.nnet.schedulers.update_learning_rate(
                    self.model_optimizer, new_lr
                )
            
            wandb.log({"Learning rate": old_lr})
            

        
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            p_ctc, feats, losses, loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if epoch >= self.hparams.sample_selection_epoch:
            self.sample_selection = True
        else:
            self.sample_selection = False
        
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            
            # save csv file
            dir_name, base_name = os.path.split(self.csv_file)
            without_ext, ext = base_name.split(".")
            
            csv_file_ = dir_name + "/" + without_ext + "_EPOCH_" + str(epoch) + "." + ext
            create_csv(csv_file_, self.reservoir)
            
            self.reservoir.reinit()
            
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
            
            ckpt_name = "_END OF EPOCH_" + str(epoch)
            self.checkpointer.save_checkpoint(name = ckpt_name)
            
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the model optimizer"

        # model optim instantiation
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            #self.checkpointer.add_recoverable("reservoir", self.reservoir)
            
    def zero_grad(self, set_to_none=False):
        self.model_optimizer.zero_grad(set_to_none)
    
    
    




if __name__ == "__main__":

    wandb.init(project='In-process_Entropy_sample_selection')
    
    #wandb.init(id="usual-armadillo-30", resume=True)
    wandb.run.save()
    
        
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

    checkpointer = hparams["checkpointer"]
    
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer, #hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer
    
    asr_brain.lr_annealing_model = lr_annealing_model

    
    asr_brain.reservoir = Reservoir(asr_brain.hparams.reservoir_size, asr_brain.hparams.attribute)
    asr_brain.n_diff = None
   
    
    asr_brain.csv_file = asr_brain.hparams.selected_sample_csv
    asr_brain.attribute = asr_brain.hparams.attribute
    asr_brain.lambda_star = asr_brain.hparams.lambda_star
    asr_brain.sample_selection = False
    
    
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
    
    
    # Test
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
    

    
    
