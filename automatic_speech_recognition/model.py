# MIT License
# code by Soohwan Kim @sooftware

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import Tensor
from typing import Tuple, Optional, Any


class Listener(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_dim: int,
            hidden_state_dim: int = 512,
            dropout_p: float = 0.2,
            num_layers: int = 3,
            bidirectional: bool = True,
            rnn_type: str = 'lstm',
    ) -> None:
        super(Listener, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=input_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        conv_outputs = nn.utils.rnn.pack_padded_sequence(inputs.transpose(0, 1), input_lengths.cpu())
        outputs, hidden_states = self.rnn(conv_outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.transpose(0, 1)

        return outputs


class MultiHeadAttention(nn.Module):
    """
    Applies a multi-headed scaled dot mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Args:
        d_model (int): dimension of model
        num_heads (int): The number of heads. (default: )
    Inputs: query, value
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
    Returns: context, attn
        - **context**: tensor containing the attended output features from the decoder.
        - **attn**: tensor containing the attention values
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)

        context = torch.bmm(attn, value)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn


class Speller(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            max_length: int = 150,
            hidden_state_dim: int = 1024,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3,
    ) -> None:
        super(Speller, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_state_dim << 1, num_classes)

    def forward_step(
            self,
            input_var: torch.Tensor,
            hidden_states: Optional[torch.Tensor],
            encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)
        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
            self,
            encoder_outputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        logits = list()
        hidden_states, attn = None, None

        targets, batch_size, max_length = self.validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            step_outputs, hidden_states, attn = self.forward_step(
                input_var=targets,
                hidden_states=hidden_states,
                encoder_outputs=encoder_outputs,
            )

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                logits.append(step_output)

        else:
            input_var = targets[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=input_var,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                )
                logits.append(step_outputs)
                input_var = logits[-1].topk(1)[1]

        return torch.stack(logits, dim=1)

    def validate_args(
            self,
            targets: Optional[Any] = None,
            encoder_outputs: torch.Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, int, int]:
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:  # inference
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                targets = targets.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1  # minus the start of sequence symbol

        return targets, batch_size, max_length


class ListenAttendSpell(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 1,
            hidden_state_dim: int = 512,
            encoder_bidirectional: bool = True,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.2,
            max_length: int = 128,
            num_attention_heads: int = 4,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(ListenAttendSpell, self).__init__()
        self.encoder = Listener(
            input_dim=input_dim,
            num_layers=num_encoder_layers,
            hidden_state_dim=hidden_state_dim,
            dropout_p=dropout_p,
            bidirectional=encoder_bidirectional,
            rnn_type=rnn_type,
        )
        decoder_hidden_state_dim = hidden_state_dim << 1 \
            if encoder_bidirectional \
            else hidden_state_dim
        self.decoder = Speller(
            num_classes=num_classes,
            max_length=max_length,
            hidden_state_dim=decoder_hidden_state_dim,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            num_heads=num_attention_heads,
            dropout_p=dropout_p,
            num_layers=num_decoder_layers,
            rnn_type=rnn_type,
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[Tensor, dict]:
        encoder_outputs = self.encoder(inputs, input_lengths)
        result = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)
        return result

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
