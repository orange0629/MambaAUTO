import torch
import torch.nn as nn
from transformers import MambaForCausalLM, MambaConfig


class crossMultiheadAttention(nn.Module):
    '''
    The cross attention block between modalities in the MambaAUTO model. Align the time series modality with the causal LLM modality.
    Note that the vocab embed size is not necessarily to be the same as the llm size. This is a flexible design.
    Inputs:
    patch_embed: the time series modality. [batch_size x n_patch x patch_embed_size]
    vocab_embed: the causal LLM modality. [batch_size x n_vocab x vocab_embed_size]
    Outputs:
    out: the aligned time series modality. [batch_size x n_patch x llm_size]
    '''

    def __init__(self, d_k, nhead, patch_embed_size, vocab_embed_size, llm_size, dropout = 0.1):
        super(crossMultiheadAttention, self).__init__()

        self.W_k = nn.Linear(vocab_embed_size, d_k * nhead)
        self.W_v = nn.Linear(vocab_embed_size, d_k * nhead)
        self.W_q = nn.Linear(patch_embed_size, d_k * nhead)

        self.W_o = nn.Linear(d_k * nhead, llm_size)

        self.nhead = nhead
        self.patch_embed_size = patch_embed_size
        self.vocab_embed_size = vocab_embed_size
        self.llm_size = llm_size
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_embed, vocab_embed):
        bs, n_patch, _ = patch_embed.size()
        _, n_vocab, _ = vocab_embed.size()

        k = self.W_k(vocab_embed).view(bs, n_vocab, self.nhead, self.d_k).transpose(1, 2) # [batch_size x nhead x n_vocab x d_k]
        v = self.W_v(vocab_embed).view(bs, n_vocab, self.nhead, self.d_k).transpose(1, 2) # [batch_size x nhead x n_vocab x d_k]
        q = self.W_q(patch_embed).view(bs, n_patch, self.nhead, self.d_k).transpose(1, 2) # [batch_size x nhead x n_patch x d_k]

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5 # [batch_size x nhead x n_patch x n_vocab]
        attn = torch.nn.functional.softmax(attn, dim = -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # [batch_size x nhead x n_patch x d_k]
        out = out.transpose(1, 2).reshape(bs, n_patch, self.nhead * self.d_k) # [batch_size x n_patch x nhead * d_k]

        out = self.W_o(out) # [batch_size x n_patch x llm_size]

        return out


class MambaAUTO(nn.Module):
    
    def __init__(self, configs) -> None:
        super(MambaAUTO, self).__init__()

        # ---general config---
        if torch.cuda.is_available():            
            if configs.use_multi_gpu:
                self.device = f"cuda:{configs.local_rank}"
            else:
                self.device = f"cuda:{configs.gpu}"
        else:
            self.device = "cpu"

        print(self.device)

        # ---load the pre-trained model---

        self.model_name = configs.model

        model_name_lookup = {
            "Mamba-130m": "state-spaces/mamba-130m-hf",
            "Mamba-370m": "state-spaces/mamba-370m-hf",
            "Mamba-790m": "state-spaces/mamba-790m-hf",
            "Mamba-1.4b": "state-spaces/mamba-1.4b-hf",
            "Mamba-2.8b": "state-spaces/mamba-2.8b-hf"
            }


        self.mamba = MambaForCausalLM.from_pretrained(
            model_name_lookup[self.model_name],
            device_map = self.device,
            output_hidden_states = True
        )

        # ---params---
        self.token_len = configs.token_len # define the token length. How much timestamps in a token.
        self.patch_embed_size = configs.patch_embed_size # the size of the patch embedding. How many dimensions in a token after passing patch embedder.

        self.vocab_embedding = self.mamba.get_input_embeddings().weight # get the pre-trained vocab embedding, in shape [vocab_size x vocab_embed_size]
        self.vocab_embedding.requires_grad = False # freeze the vocab embedding
        self.vocab_size = self.vocab_embedding.size(0) # get the vocab size
        self.vocab_embed_size = self.vocab_embedding.size(1) # get the vocab embedding size

        # just for test
        self.patch_embed_size = self.vocab_embed_size

        self.probing_size = configs.probing_size # the size of the vocabs after probing. probing_size = n_vocab in the cross attention block. Like 1000, since mamba have ~50000 tokens by default.

        self.instanceNorm = nn.InstanceNorm1d(num_features = self.token_len) # instance normalization the input time series
        self.llm_size = configs.llm_size # the size of the causal LLM, Mamba used here, e.g. 4096

        # just for test
        self.llm_size = self.vocab_embed_size

        # ---blocks---
        # freeze the model
        for name, param in self.mamba.named_parameters():
            param.requires_grad = False

        # cross multihead attention block
        self.crossMultiheadAttention = crossMultiheadAttention(
            d_k = configs.d_k, # ? maybe 128? 256?
            nhead = configs.nhead,
            patch_embed_size = self.patch_embed_size,
            vocab_embed_size = self.vocab_embed_size,
            llm_size = self.llm_size
        )

        # patch embedder
        self.patchEmbedder = nn.Linear(self.token_len, self.patch_embed_size) # project [batch_size x n_patch x token_len] to [batch_size x n_patch x patch_embed_size]

        # linears
        self.linearProbe = nn.Linear(self.vocab_size, self.probing_size) # project [vocab_size x vocab_embed_size] to [probing_size x vocab_embed_size], need T twice
        self.outputLinear = nn.Linear(self.llm_size, self.token_len) # project [batch_size x n_patch x llm_size] to [batch_size x n_patch x token_len]

    def forecast(self, x_enc, x_mark_enc = None, x_dec = None, x_mark_dec = None):
        '''
        x_enc: [batch_size x seq_len x nvar]
        x_mark_enc: [batch_size x seq_len x nvar]
        '''

        # position holder
        if not x_mark_enc:
            x_mark_enc = torch.zeros_like(x_enc)

        if not x_dec:
            x_dec = torch.zeros_like(x_enc)

        if not x_mark_dec:
            x_mark_dec = torch.zeros_like(x_enc)

        # ---preprocess---
        # instance norm
        means = torch.mean(x_enc, dim = 1, keepdim = True).detach() # keepdim so that we can broadcast, mean in shape [batch_size x 1 x nvar]
        x_enc = x_enc - means

        stdev = torch.sqrt(
            torch.var(x_enc, dim = 1, unbiased = False, keepdim = True) + 1e-5
        ) # stdev in shape [batch_size x 1 x nvar]

        x_enc /= stdev

        bs, seq_len, n_vars = x_enc.shape

        x_enc = x_enc.permute(0, 2, 1)

        # --channel independence--
        # equivalent to concate features one after another
        x_enc = x_enc.reshape(bs * n_vars, -1) # [batch_size * nvar x seq_len]

        # --patching--
        fold_out = x_enc.unfold(dimension = -1, size = self.token_len, step = self.token_len) # [batch_size * nvar x n_patch x token_len]
        n_patch = fold_out.size(1)

        # --patch embedding--
        print(fold_out.device)
        patch_embed = self.patchEmbedder(fold_out) # [batch_size * nvar x n_patch x patch_embed_size]

        # --cross attention--
        vocab_embed = self.linearProbe(self.vocab_embedding.T).T # [probing_size x vocab_embed_size]
        vocab_embed = vocab_embed.unsqueeze(0).expand(bs * n_vars, -1, -1) # [batch_size * nvar x probing_size x vocab_embed_size]
        patch_embed = self.crossMultiheadAttention(patch_embed, vocab_embed) # [batch_size * nvar x n_patch x llm_size]

        # --send to LLM--
        outputs = self.mamba(
            inputs_embeds = patch_embed,
        ).hidden_states[-1] # mamba returns (loss, logits, cache_params, hidden_states), the last hidden_state in shape [batch_size * nvar x n_patch x llm_size]

        # --output linear--
        dec_out = self.outputLinear(outputs) # [batch_size * nvar x n_patch x token_len]

        # concat and reshape
        dec_out = dec_out.reshape(bs, n_vars, -1) # [batch_size x nvar x n_patch * token_len]

        dec_out = dec_out.permute(0, 2, 1) # [batch_size x n_patch * token_len x nvar]

        # --denormalize--
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, n_patch * self.token_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, n_patch * self.token_len, 1)

        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)




            
        