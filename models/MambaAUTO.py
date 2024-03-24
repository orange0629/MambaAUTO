import torch
import torch.nn as nn
from transformers import MambaForCausalLM, MambaConfig


class crossMultiheadAttention(nn.Module):
    '''
    The cross attention block between modalities in the MambaAUTO model. Align the time series modality with the causal LLM modality.
    Note that the vocab embed size is not necessarily to be the same as the llm size.
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

        # general config
        if torch.cuda.is_available():            
            if configs.use_multi_gpu:
                self.device = f"cuda:{configs.local_rank}"
            else:
                self.device = f"cuda:{configs.gpu}"
        else:
            self.device = "cpu"

        print(self.device)

        # params
        self.token_len = configs.token_len # define the token length. How much timestamps in a token.
        self.vocab_embedding = self.mamba.get_input_embeddings().weight # get the pre-trained vocab embedding
        self.instanceNorm = nn.InstanceNorm1d(num_features = self.token_len) # instance normalization the input time series
        self.llm_size = configs.llm_size # the size of the causal LLM

        # blocks
        self.mamba = MambaForCausalLM.from_pretrained(

        )

        self.crossMultiheadAttention = crossMultiheadAttention(
            d_k = 64, # ? maybe 128? 256?
            nhead = 8,
            patch_embed_size = 64,
            vocab_embed_size = 4096,
            llm_size = 4096
        )

        self.outputLinear = nn.Linear(self.llm_size, self.token_len)

        