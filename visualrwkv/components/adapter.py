import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration, ViTModel
from transformers import T5Tokenizer
from dataclasses import dataclass
import contextlib
from typing import Optional


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@dataclass
class AdapterOutput:
    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None


class AdapterPretrain(nn.Module):
    def __init__(
        self,
        vit_model: str = "google/vit-base-patch16-224-in21k",
        t5_model: str = "t5-small",
        num_task_embeddings: int = 32,
        feature_size: int = 256,
        is_freeze_vit: bool = True,
        max_length: int = 64,
    ):
        super(AdapterPretrain, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model, model_max_length=512)
        self.tokenizer.bos_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_task_embeddings = num_task_embeddings
        self.vit = ViTModel.from_pretrained(vit_model)
        self.is_freeze_vit = is_freeze_vit
        if is_freeze_vit:
            self.freeze_vit()

        model = T5ForConditionalGeneration.from_pretrained(t5_model)
        # get hidden size from T5 model
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        # projector from vit to t5 hidden size
        self.t5_proj = nn.Linear(self.vit.config.hidden_size, self.hidden_size)
        # project from t5 hidden size to embedding size
        self.vision_proj = nn.Linear(self.hidden_size, feature_size)
        self.text_proj = nn.Linear(self.hidden_size, feature_size)
        # layer norm for vision
        self.ln_vision = LayerNorm(self.hidden_size)

        dummy_decoder = list(
            nn.Sequential(*list(model.decoder.children())[1:]).children()
        )

        # use the T5 decoder
        self.list_decoder = nn.Sequential(*list(dummy_decoder[0]))
        self.residue_decoder = nn.Sequential(*list(dummy_decoder[1:]))

        ## We use the embeddings of T5 for encoding the tokenized words
        self.language_emb = nn.Embedding.from_pretrained(model.shared.weight)
        # create task embeddings
        self.task_embs = nn.Parameter(
            torch.zeros(1, num_task_embeddings, self.hidden_size)
        )
        self.task_embs.data.normal_(mean=0.0, std=1e-4)
        self.temperature = nn.Parameter(0.07 * torch.ones([]))
        # task head
        self.itm_head = nn.Linear(self.hidden_size, 2)
        # begin of response token
        self.bor_token = "response:"
        # end of response token
        self.eor_token = "\n\n"

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def freeze_vit(self):
        for name, param in self.vit.named_parameters():
            param.requires_grad = False
        self.vit.eval()

    def encode_image(self, image):
        with self.maybe_autocast():
            visual_embs = self.ln_vision(
                self.t5_proj(self.vit(image).last_hidden_state)
            )
        return visual_embs.float()

    @torch.no_grad()
    def forward_task_embs(self, image):
        visual_embs = self.encode_image(image)
        task_embs_img = self.task_embs.expand(image.shape[0], -1, -1)
        for layer in self.list_decoder:
            task_embs_img = layer(task_embs_img, encoder_hidden_states=visual_embs)[0]
        task_embs_img = self.residue_decoder(task_embs_img)
        return task_embs_img