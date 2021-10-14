import copy
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel, gelu
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig


class DistilBertLMHead(nn.Module):
    """DistilBert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

    def forward(self, features, **kwargs):
        x = self.vocab_transform(features)  # (bs, seq_length, dim)
        x = gelu(x)  # (bs, seq_length, dim)
        x = self.vocab_layer_norm(x)  # (bs, seq_length, dim)
        x = self.vocab_projector(x)

        return x


class InvariantDistilBertConfig(DistilBertConfig):
    model_type = "invariant-distilbert"

    def __init__(self, envs=1, **kwargs):
        """Constructs InvariantDistilBertConfig."""
        super().__init__(**kwargs)
        self.envs = envs


class InvariantDistilBertForMaskedLM(DistilBertPreTrainedModel):
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, model=None):  # , model, envs):
        super().__init__(config)

        self.config = config
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.encoder = DistilBertModel(config)
        self.encoder.to('cuda')

        if len(config.envs) == 0:
            self.envs = ['erm']
        else:
            self.envs = config.envs

        self.lm_heads = {}
        for env_name in self.envs:
            self.lm_heads[env_name] = DistilBertLMHead(config)

        if model is not None:
            self.encoder = copy.deepcopy(model.distilbert)
            self.lm_heads = {}
            for env_name in self.envs:
                self.lm_heads[env_name] = DistilBertLMHead(config)
                self.lm_heads[env_name].vocab_transform = copy.deepcopy(model.vocab_transform)
                self.lm_heads[env_name].vocab_layer_norm = copy.deepcopy(model.vocab_layer_norm)
                self.lm_heads[env_name].vocab_projector = copy.deepcopy(model.vocab_projector)
                # self.register_parameter(env_name + '-head', self.lm_heads[env_name])

        for env_name, lm_head in self.lm_heads.items():
            self.__setattr__(env_name + '_head', self.lm_heads[env_name])

        self.encoder.to('cuda')
        for _, lm_head in self.lm_heads.items():
            lm_head.to('cuda')

        self.n_environments = len(self.lm_heads)

    def print_lm_w(self):
        for env, lm_h in self.lm_heads.items():
            print(lm_h.dense.weight)

    def init_head(self):
        for env_name in self.envs:
            self.lm_heads[env_name] = DistilBertLMHead(config)
            self.lm_heads[env_name].to('cuda')

    def init_base(self):
        self.encoder.init_weights()
        self.init_head()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)
        # self.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        for env, lm_head in self.lm_heads.items():
            return lm_head.vocab_projector

    def set_output_embeddings(self, new_embeddings):
        for env, lm_head in self.lm_heads.items():
            lm_head.decoder = new_embeddings

    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="roberta-base",
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     mask="<mask>",
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.n_environments == 1:
            lm_head = list(self.lm_heads.values())[0]
            prediction_scores = lm_head(sequence_output)
        else:
            prediction_scores = 0.
            for env, lm_head in self.lm_heads.items():
                prediction_scores += 1. / self.n_environments * lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
