import json
from typing import Any, Dict, Mapping
import torch
import numpy as np
from numpy import ndarray
from dateutil import parser
from metrics import re_score
from torch.optim import AdamW
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from sentence_transformers import SentenceTransformer
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.optimization import get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

def get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps, warmup_init_lr=-1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        lr (:obj:`float`, `optional`, defaults to 1e-7):
            The  LR.
        warmup_init_lr ():
            The initial lr. Defaults to LR.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """
    warmup_end_lr = optimizer.defaults["lr"]
    if warmup_init_lr < 0:
        warmup_init_lr = 0 if num_warmup_steps > 0 else warmup_end_lr

    # linearly warmup for the first args.warmup_updates
    lr_step = (warmup_end_lr - warmup_init_lr) / num_warmup_steps

    # then, decay prop. to the inverse square root of the update number
    decay_factor =  num_warmup_steps ** 0.5

    # initial learning rate
    lr = warmup_init_lr

    # optimizer.set_lr(lr)

    def lr_lambda(current_step: int):
        """Update the learning rate after each update."""
        if current_step < num_warmup_steps:
            lr = warmup_init_lr + current_step * lr_step
            lr = lr/warmup_end_lr
        else:
            lr = decay_factor * current_step ** -0.5
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup
}

def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the left, pad the last position.
    [[1, 2, 4, 5]       -> [[2, 4, 5, pad_token_id
     [5, 6, 7, 4]]      ->   6, 7, 4, pad_token_id]]
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    return shifted_input_ids

def extract_triplets(text) -> list[dict]:
    triplets: list[dict] = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


class BaseLightningModule(pl.LightningModule):
    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, test_ids: list[str] = None, sent_entailer=None, *args, **kwargs):
        """
        REBEL experiment lightning module https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
        
        Parameters
        ----------
        - conf the hydra configuration contained in the conf folder
        - config the facebook/BART/REBEL configuration
        - tokenizer the BART/REBEL tokenizer
        - model the BART/REBEL model
        - wandb_run_name, used to log the different training metrics
        """
        super().__init__(*args, **kwargs)
        self.strict_loading = False
        
        self.relations: list[str] = []
        """list of relation labels of all ontologies in ontology_paths."""
                
        for ontology_path in conf.ontology_paths:
            with open(conf.repo_path + ontology_path) as ont_f:
                ontology = json.load(ont_f)
                for rel in ontology["relations"]:
                    if rel not in self.relations: 
                        self.relations.append(rel['label'])          # avoid duplicates in score computation
        
        self.relations = list(set(self.relations))
        print("self.relations, passed to re_score :", self.relations)      
                
        self.save_hyperparameters(conf)                     # adds conf dict to self.hparams
        self.tokenizer: BartTokenizerFast = tokenizer
        self.model: BartForConditionalGeneration = model
        
        self.config = config
        """`facebook/BART` json configuration file"""
        
        # for the model, the <pad> token id is 1, but the data collator pads the batches with -100, 
        # and -100 indices are ignored by pytorch for loss computation for the tokenizer,
        # whenever we decode indices, we have to replace -100 by config.pad_token_id, via torch.where() 
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.val_preds: list[dict] =  []
        self.test_preds: list[dict] = []
        
        self.test_ids = test_ids
        """list of test_ids to be used when writing `self.test_preds` to file, dataloader doesn't keep ids."""
    
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the inputs through the model and retrieve the loss and logits in output.
        To be called in `training_step()`, `validation_step()` and `test_step()` functions.
        
        Parameters
        ----------
        
        - inputs, dictionary with keys `input_ids`, `attention_mask`, `labels`, where :
            - inputs['input_ids'].size()      is (batch_size, max_length_of_tokenized_sent_of_batch)
            - inputs['attention_mask'].size() is (batch_size, max_length_of_tokenized_sent_of_batch)
            - inputs['labels'].size()         is (batch_size, max_length_of_tokenized_linearized_triples_of_batch)
            
          `input_ids` are the indices of the tokens of the tokenized sentences, `attention_mask` is a binary vector
          used to ignore <pad> tokens in attention, and `labels` are the linearized tokenized target triples.
        
        Returns
        -------
        - a dictionary with a `logits` and `loss` keys, where `loss` is the cross entropy loss over the batch, and 
        `logits` is a tensor of size (batch_size, max_target_sequence_length, vocabulary_size) 
        """
        original_labels = inputs.pop("labels")
        """See BART_INPUTS_DOCSTRING in modeling_bart.py, shifting left gets rid of first <s> token, decoder_input_ids 
        is the decoder's teacher forcing baseline. From the DOCSTRING : 
        > "For translation and summarization training, `decoder_input_ids` should be provided. If no `decoder_input_ids` 
        > is provided, the model will create this tensor by shifting the `input_ids` to the right for denoising pre-training 
        > following the paper."
        So decoder_input_ids is used during training, from Jurafsky Chapter 13 : 
        > As in that case, we use teacher forcing in the decoder. Recall that in teacher forcing, at each time step in 
        > decoding y_{t+1} we force the system to use the gold target token from training as the next input x_{t+1}, rather than 
        > allowing it to rely on the (possibly erroneous) decoder output y_t (autoregressivaly generated target token n°t), 
        here x is `decoder_input_ids` !
        """
        inputs["decoder_input_ids"] = torch.where(original_labels != -100, original_labels, self.config.pad_token_id)
        
        outputs = self.model(
            input_ids = inputs['input_ids'], 
            attention_mask = inputs['attention_mask'], 
            decoder_input_ids = inputs['decoder_input_ids'], 
            use_cache = False,
            return_dict=True
        )
        """outputs is a dictionary with a `logits` key, where outputs['logits'] is of size (batch_size, 
        max_length_of_tokenized_linearized_triples_of_batch, vocabulary_size)"""
        
        logits = outputs['logits']
        
        labels = shift_tokens_left(original_labels, -100)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)) 
        inputs["labels"] = original_labels
        return {'loss': loss, 'logits': logits}
    
    def training_step(self, batch: dict) -> dict[str, torch.Tensor]:        
        forward_output = self.forward(batch)
        self.log('loss', forward_output['loss'])
        return forward_output['loss']
    
    def inference_step(self, batch: dict) -> dict:
        """Compute loss of model over batch without training and return predictions.
        
        Parameters
        ----------
        - batch, a dictionary with `labels`, `input_ids` and `attention_mask` keys.
        
        Return
        ------
        - dict, with `loss`, `predictions` and `labels` keys.
        """
        with torch.no_grad():
            forward_output = self.forward(batch)
            
        outputs = {'loss': forward_output['loss'].mean().detach()}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch)
        
        return outputs
        
    def validation_step(self, batch: dict) -> dict:
        """Compute validation loss and predictions for batch, return `loss`, `predictions`, `labels`."""
        outputs = self.inference_step(batch)
        self.log('val_loss', outputs['loss'])
        
        #print("Predicted triples list length for batch", len(outputs['predictions'][0]))
        self.val_preds.append(outputs)
        return outputs        
        
    def test_step(self, batch: dict[str, torch.Tensor]) -> None:
        """Same as `validation_step()` but on test data."""
        outputs = self.inference_step(batch)
        self.log('test_loss', outputs['loss'])
        
        #print("outputs =", outputs)
        self.test_preds.append(outputs)
        return outputs        
    
    def generate_triples(self, batch: dict[str, torch.Tensor]) -> tuple:
        """generate the triples for a batch of encoded natural language sentences"""
        # yields a torch long tensor of size (num_return_sequences x B x max_length), 
        # where max_length is generate() parameter
        gen_kwargs = {
            "max_length": self.hparams.max_target_length
            if self.hparams.max_target_length is not None else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams,
            "num_return_sequences": self.hparams.num_return_sequences
        }
        """model inference arguments, passed to `generate()`"""
        
        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),      # tensor of encoded input sentences size B x max_length_of_sentence_in_batch (collator padded)
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache=True,                               # speeds up decoding
            **gen_kwargs
        )
        """outputs a tensor of size num_return_sequences*batch_size x max_length_of_linearized_triples_output_in_batch"""
        
        # yields of length batch_size, containing the padded ground truth linearized triples
        gt_decoded_labels: list[str] = self.tokenizer.batch_decode(
            torch.where(batch['labels'] != -100, batch['labels'], self.config.pad_token_id), 
            skip_special_tokens=False
        )
        
        # yields a flattened list of size num_return_sequences*batch_size
        # containing the linearized triple outputs, we need to fuse beams for a same target together. 
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # list of length batch_size containing original input sentences
        original_sentences = []
        # list of length batch_size, containing every triple list per sample
        # where every return sequence for a same sample was fused into a single list, duplicate triples removed.  
        final_beam_preds: list[list[dict]] = []
        
        # i will change value batch_size times, batch_idx=i//self.hparams.num_return_sequences
        for i in range(0, len(decoded_preds), self.hparams.num_return_sequences):
            
            # list of size num_return_sequences for sample i/num_return_sequences] of batch
            preds_of_sample: list[str] = decoded_preds[i:i+self.hparams.num_return_sequences]
            triples_for_sample: set[str] = set()
            
            # for every return sequence, add triples to a same set, fuse them together 
            for ret_seq in preds_of_sample:
                triples: list[dict] = extract_triplets(ret_seq)                 # get it's triples
                for triple in triples:
                    triples_for_sample.add(json.dumps(triple, sort_keys=True).lower())  # lower() is essential, multiple beams will vary on capitals !!
            
            # make a numpy array to be able to use array slicing with another array
            triplets: np.array[dict] = np.array([json.loads(dic_trip) for dic_trip in triples_for_sample])  # extract the clean array of triples
            final_beam_preds.append(triplets)
            
        return final_beam_preds, [extract_triplets(rel.replace("<sub>", "<subj>")) for rel in gt_decoded_labels]
    
    def on_validation_epoch_end(self):
        for pred in self.val_preds:
                for llm_triples, gt in zip(pred['predictions'], pred['labels']):
                    
                    #print("------ llm triples ----------")
                    for pred in llm_triples:
                        
                        pred['head'] = pred['head'].lower()
                        pred['tail'] = pred['tail'].lower()
                        pred['type'] = pred['type'].lower()
                        
                        if pred['type'] == 'publication date' or pred['type'] == 'inception' or pred['type'] == 'start time':
                            try:
                                dt = parser.parse(pred['tail'])     # parse iso string as simple date   
                                pred['tail'] = dt.strftime('%d %B %Y').lower()  # 01 January 2020
                            except:
                                pass
                        #print(pred)
                    
                    #print("------ ground truth ----------")
                    for pred in gt:
                        pred['head'] = pred['head'].lower()
                        pred['tail'] = pred['tail'].lower()
                        pred['type'] = pred['type'].lower()
                        #print(pred)

        # BUG : re_score does not normalize to lowercase
        pred_relations = [item for pred in self.val_preds for item in pred['predictions']]
        gt_relations = [item for pred in self.val_preds for item in pred['labels']]
            
        scores, precision, recall, f1 = re_score(
            pred_relations,
            gt_relations,
            relation_types=self.relations
        )
        
        self.log("val_prec_micro", precision)
        self.log("val_recall_micro", recall)
        self.log("val_F1_micro", f1)
        
        # empty validation predictions list, making space for new batch.
        self.val_preds.clear()
        
    def on_test_epoch_end(self):
        # BUG : re_score does not normalize to lowercase, so we do that here.
        for pred in self.test_preds:
                for llm_triples, gt in zip(pred['predictions'], pred['labels']):
                    for pred in llm_triples:
                        pred['head'] = pred['head'].lower()
                        pred['tail'] = pred['tail'].lower()
                        pred['type'] = pred['type'].lower()
                        
                        if pred['type'] == 'publication date':
                            try:
                                dt = parser.parse(pred['tail'])     # parse iso string as simple date   
                                pred['tail'] = dt.strftime('%d %B %Y').lower()  # 01 January 2020
                            except:
                                pass
                    
                    for pred in gt:
                        pred['head'] = pred['head'].lower()
                        pred['tail'] = pred['tail'].lower()
                        pred['type'] = pred['type'].lower()

        # we just save the file here without ids to avoid technical debt,
        # if you want to fully evaluate properly, re-use the model via a
        # an extra adpater in utils that loads from a trained checkpoint.
        with open(self.hparams.repo_path + self.hparams.output_file_path, 'a') as out_f:
            idx = 0
            for pred in self.test_preds:
                res: list = []
                for llm_triples in pred['predictions']:
                    triples: list = []
                    for triple in llm_triples:
                        if triple['type'] == 'publication date':
                            try:
                                dt = parser.parse(pred['tail'])     # parse iso string as simple date   
                                triple['tail'] = dt.strftime('%d %B %Y').lower()  # 01 January 2020
                            except:
                                pass
                        if triple['type'] in self.relations:   # filter out triples not in ontology, NOTE : not useful, this can be removed
                            triples.append(triple)
                    res.append(triples)
                pred['predictions'] = res
                
                for llm_triples, gt in zip(pred['predictions'], pred['labels']):
                    out_f.write(json.dumps({
                        'id': self.test_ids[idx],
                        'predictions': list(llm_triples),
                        'labels':list(gt),
                    }) + "\n")
                    
                    idx += 1
        
        pred_relations = [item for pred in self.test_preds for item in pred['predictions']]
        
        #print("pred relations\n\n", pred_relations)
        gt_relations = [item for pred in self.test_preds for item in pred['labels']]
        
        scores, precision, recall, f1 = re_score(
            pred_relations,
            gt_relations,
            relation_types=[rel['label'] for rel in self.relations]
        )
        
        self.log("test_prec_micro", precision)
        self.log("test_recall_micro", recall)
        self.log("test_F1_micro", f1)
        
        self.test_preds.clear()
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
            "eps": self.hparams.adam_epsilon,
        }
    
        optimizer_kwargs["lr"] = self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "inverse_square_root":
            # args = {"warmup_updates": self.hparams.warmup_steps, "lr": [self.hparams.learning_rate]}
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler
    