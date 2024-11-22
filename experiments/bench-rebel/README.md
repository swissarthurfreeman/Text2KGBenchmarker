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
lm_logits = self.lm_head(outputs[0])
lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
...
labels = labels.to(lm_logits.device)
loss_fct = CrossEntropyLoss()
masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
```

here `lm_logits` appears to be of size `(B, seqlen, 50272)` and the loss is then computed between that and the labels variable, which is of size `(B, seqlen)` where each entry is an index in the vocabulary. Recall that `nn.CrossEntropyLoss` really just is a wrapper for `F.cross_entropy(input, target)` where the input `(B, seqlen, 50272)` are predicted logits and target `(B*seqlen)` are the class indices. 



