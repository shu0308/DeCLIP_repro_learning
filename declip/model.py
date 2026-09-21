from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ClassPrompt(nn.Module):
    def __init__(self, text_width: int, visual_width: int, length: int):
        super().__init__()
        self.text = nn.Parameter(torch.empty(2, length, text_width))
        self.visual = nn.Parameter(torch.empty(2, length, visual_width))
        nn.init.normal_(self.text, std=0.02)
        nn.init.normal_(self.visual, std=0.02)


class DeCLIP(nn.Module):
    """Paper-faithful DP + AST on a frozen OpenCLIP ViT model.

    Sign index 0 is the positive prompt and sign index 1 is the negative prompt.
    The same visual prompt is injected into every one of the final n blocks and
    discarded after each block, matching equations (6)-(7) of arXiv v3.
    """

    def __init__(self, clip_model, tokenizer, class_names: Sequence[str], prompt_length=4, prompted_blocks=5, pool_temperature=1.0, pooling="attention", exclude_cls=False, semantic_init=False):
        super().__init__()
        self.clip = clip_model
        self.class_names = list(class_names)
        self.prompt_length = prompt_length
        self.prompted_blocks = prompted_blocks
        self.pool_temperature = pool_temperature
        if pool_temperature <= 0:
            raise ValueError("pool_temperature must be positive")
        self.pooling = pooling
        self.exclude_cls = exclude_cls
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        visual = self.clip.visual
        if not hasattr(visual, "transformer") or len(visual.transformer.resblocks) < prompted_blocks:
            raise TypeError("DeCLIP requires an OpenCLIP VisionTransformer")
        text_width = self.clip.token_embedding.embedding_dim
        visual_width = visual.class_embedding.numel()
        self.prompts = nn.ModuleList([ClassPrompt(text_width, visual_width, prompt_length) for _ in class_names])
        placeholder = " ".join(["X"] * prompt_length)
        tokens = tokenizer([f"{placeholder} {name.replace('_', ' ')}." for name in class_names])
        self.register_buffer("prompt_tokens", tokens, persistent=True)
        if semantic_init:
            init_tokens = tokenizer(["a photo of a", "not a photo of"]).to(self.clip.token_embedding.weight.device)
            with torch.no_grad():
                contexts = self.clip.token_embedding(init_tokens)[:, 1 : 1 + prompt_length]
                for module in self.prompts:
                    module.text.copy_(contexts)

    def prompt_parameters(self, class_indices: Iterable[int]):
        return [p for idx in class_indices for p in self.prompts[idx].parameters()]

    def freeze_except(self, class_indices: Iterable[int]):
        active = set(class_indices)
        for idx, module in enumerate(self.prompts):
            module.requires_grad_(idx in active)

    def _encode_text(self, class_indices, sign):
        token_ids = self.prompt_tokens[class_indices]
        dtype = self.clip.transformer.get_cast_dtype()
        x = self.clip.token_embedding(token_ids).to(dtype)
        contexts = torch.stack([self.prompts[i].text[sign] for i in class_indices]).to(dtype)
        x[:, 1 : 1 + self.prompt_length] = contexts
        x = x + self.clip.positional_embedding.to(dtype)
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = self.clip.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), token_ids.argmax(dim=-1)]
        projection = self.clip.text_projection
        x = projection(x) if isinstance(projection, nn.Linear) else x @ projection
        return F.normalize(x, dim=-1)

    def _shared_visual_prefix(self, images):
        v = self.clip.visual
        x = v.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)
        split = len(v.transformer.resblocks) - self.prompted_blocks
        with torch.no_grad():
            for block in v.transformer.resblocks[:split]:
                x = block(x)
        return x

    def _encode_visual(self, shared, class_idx, sign):
        v = self.clip.visual
        x = shared
        prompt = self.prompts[class_idx].visual[sign].to(x.dtype)
        for block in v.transformer.resblocks[-self.prompted_blocks:]:
            repeated = prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = block(torch.cat([repeated, x], dim=1))[:, self.prompt_length :]
        x = v.ln_post(x)
        if v.proj is not None:
            x = x @ v.proj
        if self.exclude_cls:
            x = x[:, 1:]
        return F.normalize(x, dim=-1)

    @staticmethod
    def _token_similarity(visual_tokens, text_feature):
        return torch.einsum("bld,d->bl", visual_tokens, text_feature)

    def _aggregate(self, similarity):
        if self.pooling == "attention":
            weights = F.softmax(self.pool_temperature * similarity, dim=-1)
            pooled = (weights * similarity).sum(-1)
        elif self.pooling == "mean":
            pooled = similarity.mean(-1)
        elif self.pooling == "max":
            pooled = similarity.max(-1).values
        elif self.pooling == "logsumexp":
            pooled = torch.logsumexp(self.pool_temperature * similarity, dim=-1) / self.pool_temperature
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return pooled * self.clip.logit_scale.exp()

    def _aggregate_pair(self, positive_similarity, negative_similarity):
        if self.pooling != "attention":
            return self._aggregate(positive_similarity), self._aggregate(negative_similarity)
        # DualCoOp class-specific region aggregation: the positive branch
        # selects relevant regions and the same weights aggregate both signs.
        weights = F.softmax(20.0 * positive_similarity, dim=-1)
        scale = self.clip.logit_scale.exp()
        positive = (weights * positive_similarity).sum(-1) * scale
        negative = (weights * negative_similarity).sum(-1) * scale
        return positive, negative

    def forward(self, images, class_indices: Sequence[int], ast_temperature=1.0):
        shared = self._shared_visual_prefix(images)
        return self.forward_from_shared(shared, class_indices, ast_temperature)

    def forward_from_shared(self, shared, class_indices: Sequence[int], ast_temperature=1.0):
        class_indices = list(class_indices)
        positive_texts = self._encode_text(class_indices, 0)
        negative_texts = self._encode_text(class_indices, 1)
        positive_scores, negative_scores = [], []
        for local_idx, class_idx in enumerate(class_indices):
            positive_visual = self._encode_visual(shared, class_idx, 0)
            negative_visual = self._encode_visual(shared, class_idx, 1)
            positive_similarity = self._token_similarity(positive_visual, positive_texts[local_idx])
            negative_similarity = self._token_similarity(negative_visual, negative_texts[local_idx])
            positive, negative = self._aggregate_pair(positive_similarity, negative_similarity)
            positive_scores.append(positive)
            negative_scores.append(negative)
        logits = torch.stack(
            [torch.stack(positive_scores, dim=1), torch.stack(negative_scores, dim=1)], dim=-1
        ) / float(ast_temperature)
        return logits

    @staticmethod
    def ast_temperature(task_number: int, seen_class_count: int, strength=16.0):
        # The paper uses t=1 for the base task.
        if task_number <= 1:
            return 1.0
        return max(strength * task_number / seen_class_count, 1.0)

