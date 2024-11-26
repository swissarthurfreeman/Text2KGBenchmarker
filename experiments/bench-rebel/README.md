## REBEL/BART Architecture

For the sake of reproducing this repository, it makes sense to try to understand as much as possible of what's
going under the hood. This will allow us to overcome technical debt and better understand what we're doing. 
Rebel is based on Facebook's BART encoder/decoder architecture, it's architecture can be displayed as follows :

```python
>>> import torch
>>> from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
>>> model = AutoModelForSeq2SeqLM.from_pretrained('/home/gordon/Documents/edu/gordon_ms/Project/Benchmarker/experiments/rebel/Rebel-large')
>>> model
BartForConditionalGeneration(
  (model): BartModel(
    (shared): BartScaledWordEmbedding(50272, 1024, padding_idx=1)
    (encoder): BartEncoder(
      (embed_tokens): BartScaledWordEmbedding(50272, 1024, padding_idx=1)
      (embed_positions): BartLearnedPositionalEmbedding(1026, 1024)
      (layers): ModuleList(
        (0-11): 12 x BartEncoderLayer(
          (self_attn): BartSdpaAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layernorm_embedding): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): BartDecoder(
      (embed_tokens): BartScaledWordEmbedding(50272, 1024, padding_idx=1)
      (embed_positions): BartLearnedPositionalEmbedding(1026, 1024)
      (layers): ModuleList(
        (0-11): 12 x BartDecoderLayer(
          (self_attn): BartSdpaAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (activation_fn): GELUActivation()
          (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): BartSdpaAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layernorm_embedding): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (lm_head): Linear(in_features=1024, out_features=50272, bias=False)
)
>>> type(model)
<class 'transformers.models.bart.modeling_bart.BartForConditionalGeneration'>
```

This is implemented within Huggingface's transformers package, which implements a whole bunch of classical models, including BERT and XLNet, full list is available [here](https://github.com/huggingface/transformers/tree/main/src/transformers/models).

### Layer by Layer

First layer, `BartScaledWordEmbedding(50272, 1024, padding_idx=1)` is an instance of `nn.Embedding`, a simple
lookup table that stores embeddings of a fixed dictionary and size. Here, this is BART facebook's dictionary
size of 50'272 tokens, and every token has an embedding of size 1024. This layer is used to retrieve embeddings
for batches of indices like so : 

```python
>>> from transformers.models.bart.modeling_bart import BartScaledWordEmbedding
>>> e = BartScaledWordEmbedding(50272, 1024, padding_idx=1)
>>> e(torch.LongTensor([[0, 1, 2], [0, 1, 2]])) # input B x seqlen where seqlen is the number of tokens of seq 
tensor([[[-0.9930, -1.2196, -0.6536,  ..., -0.1884,  0.6079,  0.4055],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.2018, -0.5649,  1.3997,  ...,  0.9354,  0.9541, -0.8603]],

        [[-0.9930, -1.2196, -0.6536,  ..., -0.1884,  0.6079,  0.4055],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.2018, -0.5649,  1.3997,  ...,  0.9354,  0.9541, -0.8603]]],
       grad_fn=<MulBackward0>)
# output is [2, 3, 1024] e.g. B x N x 1024, N must be the same for all sequences.
```

`padding_idx` means entries at position 1 do not contribute to gradients, here the embedding of the 2nd token of every batch is not updated. `padding_idx` is the index
in the vocabulary of the `pad` token used in BART, this is a BART implementation
specific detail, no need to go into full details.

The second layer, `BartLearnedPositionalEmbedding` is also an instance of `nn.Embedding`, which does some BART specific hack. Then follows a stack of `BartEncoderLayer`s, which combine two self attention layers seperated by `GELU` and end on a `LayerNorm` layer, note that dimensionality is never reduced, the decoder does the same thing, and the model has an additional language modeling head,

```python
(lm_head): Linear(in_features=1024, out_features=50272, bias=False)
```

in charge of taking these rich embeddings and output a distribution of logits over the vocabulary to predict the next token. Note that this is a seq2seq model, trained with Masked Language Modeling (MLM) to reconstruct an initial, corrupted sentence with masked tokens. As is classical with pytorch models, the input is always batched, and the model expects an input of batched token indicies (B, seqlen) and will output a tensor of rich contextual embeddings of size (B, seqlen, 1024), the **first sequence embedding** [will be fed](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1660) to the language modeling head, 

```python
output = self.model(input_ids, ...) # output is Tuple | Seq2SeqModelOutput
lm_logits = self.lm_head(outputs[0])modeljust is a wrapper for `F.cross_entropy(input, target)` where the input `(B, seqlen, 50272)` are predicted logits and target `(B*seqlen)` are the class indices. 


## Tokenizer

REBEL's tokenizer can be loaded the following way : 

```python
>>> tokenizer = AutoTokenizer.from_pretrained('/home/gordon/Documents/edu/gordon_ms/Project/Benchmarker/experiments/rebel/Rebel-large', use_fast=True, additional_special_tokens=[<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>'])
```

it's usage is as follows, 

```python
>>> sentence = '<triplet> This Must Be the Place <subj> Talking Heads <obj> performer <subj> Speaking in Tongues <obj> part of <triplet> Talking Heads <subj> new wave <obj> genre <triplet> Speaking in Tongues <subj> Talking Heads <obj> performer'
>>> res = tokenizer([sentence])
{'input_ids': [[0, 50267, 152, 8495, 1456, 5, 6067, 1437, 50266, 14920, 23376, 1437, 50265, 12576, 1437, 50266, 3580, 11, 17922, 3663, 1437, 50265, 233, 9, 1437, 50267, 14920, 23376, 1437, 50266, 92, 4605, 1437, 50265, 11581, 1437, 50267, 3580, 11, 17922, 3663, 1437, 50266, 14920, 23376, 1437, 50265, 12576, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
>>> len(res['input_ids'][0])
49                              # longer because of subword tokenization
>>> len(sentence.split())
35
>>> tokenizer.convert_ids_to_tokens(0)
'<s>'
>>> tokenizer.convert_ids_to_tokens(res['input_ids'][0])
['<s>', '<triplet>', 'This', 'Must', 'Be', 'the', 'Place', '', '<subj>', 'Talking', 'Heads', '', '<obj>', 'performer', '', '<subj>', 'Speaking', 'in', 'Tong', 'ues', '', '<obj>', 'part', 'of', '', '<triplet>', 'Talking', 'Heads', '', '<subj>', 'new', 'wave', '', '<obj>', 'genre', '', '<triplet>', 'Speaking', 'in', 'Tong', 'ues', '', '<subj>', 'Talking', 'Heads', '', '<obj>', 'performer', '</s>']
```

So the output of a tokenization of `K` sentence is a dictionary with two keys `input_ids`, `attention_mask` of length `K` where position `i` contains the list of token indices for sentence `i` in batch.

## LightningDataModule, DataSet Scripts

A lightning data module is a simple interface which provides a method to prepare data, and retrieve validation, training and test dataloaders, that yield appropriately formatted batches which can be directly used within gradient updating loops. The file `lightning_data_module.py` contains such a class, it gets instantiated by providing it the tokenizer and the model, the idea being that we want to pre-process our data by doing the tokenization of the sentences and the linearized triples before hand. We first start with a call to `load_dataset` which will use the corresponding dataset script `wkdata_synth_movie.py` to load the data from the files to batches. This file provides a class which has a function `_generate_examples()` that yields a [generator](https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python) that will generate tuples with the id and a dictionary of the sentence and it's linearized triples.  


This script is executed when `load_dataset()` is called in `lightning_data_module.py`'s constructor, 

```python
self.datasets: dict[str, Dataset] = load_dataset(
    conf.dataset_script_path, 
    data_files={'train': conf.train_file, 'dev': conf.val_file, 'test': conf.test_file}, 
    trust_remote_code=True
)
```

and `load_dataset` will call `_generate_examples()` method the and get a `Dataset` instance for every split. At this stage, the data hasn't been tokenized yet, this is done within the `prepare_data()` method of the lightning data module.

```python
def prepare_data(self) -> None:
  self.train_dataset = self.datasets['train'].map(
      self.preprocess_function,
      batched=True,
      remove_columns=self.datasets['train'].column_names,
      load_from_cache_file=False
  )
  ...
...

def preprocess_function(self, batch: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    inputs = batch[self.text_key]
    targets = batch[self.target_key]

    model_inputs = self.tokenizer(inputs, max_length=1024, padding=False, truncation=True)

    with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(targets, max_length=1024, padding=False, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]      
    return model_inputs
```

the `preprocess_function()` is applied to the whole dataset, here batch is of size 1000, and is an instance of `LazyBatch` HG class, which in practices functions as a dict, whith a `'sent'` and `'triples'` keys containing a list of sentences and a corresponding list of linearized triples. The data is then ran through the tokenizer, including the sentences and targets, and are trunkated to a maximum of 1024 tokens. The output is a dictionary with a `'labels'` and `'input_ids'` list of vocabulary indices, corresponding to the tokenization result, if we add these lines before the `return`, 

```python
print(self.tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0]), 
      self.tokenizer.convert_ids_to_tokens(model_inputs["labels"][0]))
```

we'd get,
```python
['<s>', 'We', 'ĠLive', 'Ġin', 'ĠPublic', 'Ġdives', 'Ġdeep', 'Ġinto', 'Ġthe', ..., 'Ġaward', '-', 'winning', 'Ġfilm', '.', '</s>'] 
['<s>', '<triplet>', 'ĠWe', 'ĠLive', 'Ġin', 'ĠPublic', 'Ġ', '<subj>', 'ĠO', 'nd', 'i', 'ĠTim', 'oner', 'Ġ', '<obj>', 'Ġdirector', 'Ġ', '<subj>', ..., '</s>']
```

Note that the weird character `'Ġ'` stands for a space, see this hugging face [issue](https://github.com/huggingface/transformers/issues/22306).

### LightningDataModule Usage

Using the class can be done like so, 

```python
pl_data_module: BaseLightningDataModule = BaseLightningDataModule(conf, tokenizer, model)
pl_data_module.prepare_data()

val_dataloader = pl_data_module.val_dataloader()

for batch in iter(val_dataloader):
    
    print(type(batch), batch.keys(), len(batch['input_ids']), type(batch['input_ids']), len(batch['input_ids'][1]))
    print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
    break

```

The first print will yield,

```
<class 'transformers.tokenization_utils_base.BatchEncoding'> 
dict_keys(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']) 
24
<class 'torch.Tensor'>
116
```
where 24 is the batch size, e.g. every value of this dictionary is a Tensor of size (24, 116), where every line contains the tokenized sentence with padding added by the `DataCollatorForSeq2Seq`. The first, `'input_ids'` contains the indices for 

The output of the second print will look like,
```python
['<s>', 'We', 'ĠLive', 'Ġin', 'ĠPublic', 'Ġdives', ..., 'winning', 'Ġfilm', '.', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', ..., '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
```

Note that the length of 116 corresponds here to the length of the longest sentence *within the batch*, the data collator will pad up to that size for every sentence in the batch to have the same size, making it fittable within a matrix that can be fed to the model. Note that in the collator's constructor documentation it's written, for the `padding` parameter, `padding - True or 'longest' (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).` Recall that the data collator is passed to the DataLoader constructor of the pytorch data module,

```python
def val_dataloader(self) -> DataLoader:
  return DataLoader(
      dataset=self.train_dataset,
      batch_size=self.conf.val_batch_size,
      collate_fn=self.data_collator,
      pin_memory=True
  )
```

the dataloader will apply the collation logic over the provided dataset, adding the mapping to the mix. We could propably do this ourselves in the pre-processing and omit the `collate_fn` method, but might as well use stuff that already exists. The `<pad>` token itself, is part of the vocabulary of the tokenizer, and, if we look at the first 10 tokens,

```python
for i in range(10):
    print(tokenizer.convert_ids_to_tokens(i))

# yields
<s> <pad> </s> <unk> . Ġthe , Ġto Ġand Ġof
```

The first 4 tokens are facebook BART specific tokens. They're used to delimit padding tokens, end and start of sequences and the unknown token. Recall BART is a seq2seq model, when we give it a new sequence to translate, it gets padded with `<s>`, and the model's output gets terminated by having the model generate the `</s>` token. 


## BaseLightningModule

`*args` is a list of arguments that are passed to the constructor but not explicitely listed, without name, `*kwargs` is a dictionary of arguments passed that aren't known to the constructor, but that were provided a name, see [this post](https://stackoverflow.com/questions/3394835/use-of-args-and-kwargs) for more details. Hugging face makes heavy use of this to pass arguments from one base class to other said base class might use.

