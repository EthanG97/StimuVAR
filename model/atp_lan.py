"""
Main file containing core ATP class code and some example utilities using ATP.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


@dataclass
class ATPConfig:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder).
    '''
    # ATPEncoder params
    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 128
    d_model_ff: int = 128
    enc_dropout: float = 0.1
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 4096  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)
    use_pe: bool = False  # controls whether to use positional encodings in the ATPEncoder
    soft_inference: bool = False  # controls whether to use soft (continuous) selection during ATP evaluation

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ATPEncoder(nn.Module):
    """
    The multimodal transformer encoder for the ATP model. For analysis purposes, the ATP encoder
    does not use any positional information (no positional encodings + transformer / self-attention)
    and is generally kept low-capacity. If the goal is raw accuracy (not analysis), you can relax these constraints.
    """
    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig with parameters for the (transformer-based, atemporal) encoder for ATP.
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.d_model = config.d_model
        self.dropout = nn.Dropout(p=config.enc_dropout)
        self.use_pe = config.use_pe
        
        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)
        if self.use_pe:
            self.pe = PositionalEncoding(self.d_model, dropout=config.enc_dropout)

    def forward(self, x_inputs: torch.tensor, len_query):
        """
        x_inputs: torch.tensor of shape (L, N, D)
        """
        L, N, D = x_inputs.size()  # (sequence_length, batch_size, d_model)
        assert D == self.d_model, "inputs dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        if self.use_pe:
            x_query = self.pe(x_encoded[:len_query])
            x_cands = self.pe(x_encoded[len_query:])
            x_encoded = torch.cat([x_query, x_cands], dim=0)
        # x_encoded += self.modality_encoding(x_encoded)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)
        return x_encoded


class ATPSelectorModel(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language 
    encoding and outputs a (discrete) selection over the input frames, to help analyze 
    downstream discriminative video-language tasks.
    """
    
    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(config, **kwargs)
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_query: Optional[torch.tensor] = None,
                x_txt_cands: Optional[torch.tensor] = None,
                **kwargs):
        """
        Performs the ATP selection operation on the input embeddings.
        Returns selected (unmodified) visual embeddings and selection mask.
        x_vis_seq: torch.tensor of shape (N, L, D_in) with visual embeddings of size D_in
        x_txt_query: torch.tensor of shape (N, D_in) with optional query text embeddings
        x_txt_cands: torch.tensor of shape (N, L_cands, D_in) with optional add'l text embeddings
        (optional) temperature: used when config.use_ste is set to False; set as keyword argument. Default = 1.0.
        """        
        N, L, D = x_vis_seq.size()  # (batch_size, sequence_length, feature_dimension)
        x_vis_seq = x_vis_seq.permute(1, 0, 2)    # make (L, N, D); sequence first
    
        # n_text_feats = self.atp_encoder.modality_encoding.n_text_feats
        n_text_feats = 0
        
        # combine inputs into one multimodal sequence
        x_inputs = []
        if x_txt_cands is not None:
            x_inputs.append(x_txt_cands.permute(1,0,2))
            n_text_feats += x_txt_cands.shape[1]
            
        x_inputs.append(x_vis_seq)
        x_inputs = torch.cat(x_inputs, dim=0)
        
        # embed the input sequence to the (smaller) model dimension (d_model) with modality encodings.
        x_encoded = self.embedding(self.dropout(x_inputs))
        x_atp_encoded = self.atp_encoder(x_encoded, n_text_feats)[n_text_feats:]
        
        # obtain selection scores (logits)
        x_logits = self.logits(self.dropout(x_atp_encoded))

        # get selection mask over the visual inputs (frames)
        if self.training:
            # obtain differentiable selection mask during training.
            if self.config.use_ste:  # gumbel softmax straight-through estimator; hard selection
                selection_mask = F.gumbel_softmax(x_logits, dim=0, hard=True)
            else:  # softmax with temperature; soft selection
                selection_mask = F.softmax(x_logits / kwargs.get("temperature", 1.0), dim=0)
        else:
            if self.config.soft_inference:
                selection_mask = F.softmax(x_logits / kwargs.get("temperature", 1.0), dim=0)
            # ATP always performs hard (discrete) selection during evaluation.
            else:
                selection_index_argmax = x_logits.max(dim=0, keepdim=True)[1]
                selection_mask = torch.zeros_like(x_logits, memory_format=torch.contiguous_format).scatter_(
                    dim=0, index=selection_index_argmax, value=1.0)

        # use mask to perform selection
        # selected_frames = (selection_mask * x_vis_seq).sum(dim=0)
        
        return selection_mask