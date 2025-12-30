# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only ColQwen3 multimodal embedding model compatible with HuggingFace weights."""

import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.configs.colqwen3 import ColQwen3Config
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import (
    Qwen3LLMModel,
    Qwen3VLMoeVisionModel,
)
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_int_env_var

logger = logging.getLogger(__name__)


@dataclass
class ColQwen3EmbeddingOutput:
    """Multi-vector embedding output for ColQwen3.

    Each request produces per-token embeddings (not pooled).
    Output shape per request: [seq_len, embed_dim]
    """

    embeddings: Union[torch.Tensor, List[torch.Tensor]]


class ColQwen3(nn.Module):
    """ColQwen3 multimodal embedding model for retrieval tasks.

    This model outputs per-token embeddings (multi-vector) rather than
    pooled single-vector embeddings. The embeddings are L2-normalized
    for use with late-interaction retrieval methods like MaxSim.

    Architecture:
        - Vision encoder: Qwen3VLMoeVisionModel
        - Language model: Qwen3LLMModel (without lm_head)
        - Projection layer: Linear(hidden_size -> embed_dim)
        - L2 normalization on output embeddings
    """

    # Weight mapping for checkpoint conversion from HuggingFace format
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # Map VLM nesting from HuggingFace checkpoint
            "vlm.model.visual.": "visual.",
            "vlm.model.language_model.": "model.",
            "vlm.model.": "model.",
            # Legacy/alternative mappings
            "model.visual.": "visual.",
            "model.language_model.": "model.",
            # Projection layer mapping
            "vlm.embedding_proj_layer.": "embedding_proj_layer.",
            "custom_text_proj.": "embedding_proj_layer.",
            "embedding_proj_layer.": "embedding_proj_layer.",
        },
    )

    def __init__(
        self,
        config: ColQwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()

        # Check for data parallel encoding
        server_args = get_global_server_args()
        self.use_data_parallel = (
            server_args.mm_enable_dp_encoder if server_args else False
        )

        # Initialize vision encoder (Qwen3VL vision model)
        self.visual = Qwen3VLMoeVisionModel(
            config.vision_config,
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        # Initialize language model (without lm_head since we only need embeddings)
        self.model = Qwen3LLMModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # Embedding projection layer: hidden_size -> embed_dim
        self.embedding_proj_layer = nn.Linear(
            config.text_config.hidden_size,
            config.embed_dim,
            bias=True,
        )

        # Check if MRoPE is enabled
        self.is_mrope_enabled = "mrope_section" in getattr(
            config.text_config, "rope_scaling", {}
        ) or "mrope_section" in getattr(config, "rope_scaling", {})

        # Deepstack configuration (for Qwen3-VL compatibility)
        self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
        self.use_deepstack = {Modality.IMAGE: True, Modality.VIDEO: True}

    def separate_deepstack_embeds(self, embedding):
        """Separate main embeddings from deepstack embeddings."""
        if self.num_deepstack_embeddings == 0:
            return embedding, None

        separate_index = self.config.text_config.hidden_size
        input_embeds = embedding[:, :separate_index]
        input_deepstack_embeds = embedding[:, separate_index:]
        return input_embeds, input_deepstack_embeds

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input IDs to accommodate multimodal tokens."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract image features using the vision encoder."""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        max_patches_per_call = get_int_env_var("SGLANG_VLM_MAX_PATCHES_PER_VIT", 0)
        max_images_per_call = get_int_env_var("SGLANG_VLM_MAX_IMAGES_PER_VIT", 0)

        if max_patches_per_call == 0 and max_images_per_call == 0:
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values,
                    image_grid_thw.tolist(),
                    rope_type="rope_3d",
                )
            else:
                return self.visual(pixel_values, grid_thw=image_grid_thw)

        # Handle chunked processing for large inputs
        grid_thw_list = image_grid_thw.tolist()
        patches_per_image = [int(math.prod(g)) for g in grid_thw_list]
        num_images = len(patches_per_image)

        cum_patches = [0]
        for p in patches_per_image:
            cum_patches.append(cum_patches[-1] + p)
        total_patches = cum_patches[-1]

        assert pixel_values.size(0) == total_patches

        all_chunk_embeds: List[torch.Tensor] = []
        img_start = 0

        while img_start < num_images:
            img_end = img_start
            patches_in_chunk = 0
            images_in_chunk = 0

            while img_end < num_images:
                next_patches = patches_per_image[img_end]

                if (
                    max_patches_per_call > 0
                    and patches_in_chunk + next_patches > max_patches_per_call
                ):
                    break

                if max_images_per_call > 0 and images_in_chunk + 1 > max_images_per_call:
                    break

                patches_in_chunk += next_patches
                images_in_chunk += 1
                img_end += 1

            if img_end == img_start:
                img_end = img_start + 1
                patches_in_chunk = patches_per_image[img_start]
                images_in_chunk = 1

            patch_start = cum_patches[img_start]
            patch_end = cum_patches[img_end]
            pixel_chunk = pixel_values[patch_start:patch_end]
            grid_chunk = image_grid_thw[img_start:img_end]

            if self.use_data_parallel:
                chunk_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_chunk,
                    grid_chunk.tolist(),
                    rope_type="rope_3d",
                )
            else:
                chunk_embeds = self.visual(pixel_chunk, grid_thw=grid_chunk)

            all_chunk_embeds.append(chunk_embeds)
            img_start = img_end

        return torch.cat(all_chunk_embeds, dim=0)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract video features using the vision encoder."""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            return self.visual(pixel_values, grid_thw=video_grid_thw)

    def get_input_embeddings(self):
        """Get the input embedding layer."""
        return self.model.embed_tokens

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        """Check if LoRA should be applied to a module."""
        return bool(self._lora_pattern.match(module_name))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = True,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> EmbeddingPoolerOutput:
        """Forward pass for ColQwen3 embedding model.

        Args:
            input_ids: Flattened input token IDs
            positions: Position IDs (may be 3D for MRoPE)
            forward_batch: Batch information including multimodal inputs
            get_embedding: Must be True for embedding models
            pp_proxy_tensors: Pipeline parallel proxy tensors

        Returns:
            EmbeddingPoolerOutput with per-token embeddings (multi-vector)
        """
        assert (
            get_embedding
        ), "ColQwen3ForRetrieval is only used for embedding generation"

        # Use MRoPE positions if enabled
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        # Process multimodal inputs and get hidden states
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # Handle pipeline parallel intermediate output
        if not self.pp_group.is_last_rank:
            return hidden_states

        # Project to embedding dimension
        embeddings = self.embedding_proj_layer(hidden_states)

        # L2 normalize per token
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Split batch into per-request embeddings
        if forward_batch.extend_seq_lens is not None:
            seq_lens = forward_batch.extend_seq_lens.tolist()
            embeddings_list = []
            start_idx = 0
            for seq_len in seq_lens:
                embeddings_list.append(embeddings[start_idx : start_idx + seq_len])
                start_idx += seq_len
            return EmbeddingPoolerOutput(embeddings=embeddings_list)
        else:
            return EmbeddingPoolerOutput(embeddings=embeddings)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights from checkpoint."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Skip lm_head weights (not used in embedding model)
            if "lm_head" in name:
                continue

            # Transform checkpoint weight names to model weight names
            # Checkpoint format: vlm.model.language_model.* or vlm.model.visual.*
            # Model format: model.* or visual.*
            if name.startswith("vlm.model.language_model."):
                name = name.replace("vlm.model.language_model.", "model.")
            elif name.startswith("vlm.model.visual."):
                name = name.replace("vlm.model.visual.", "visual.")
            elif name.startswith("vlm.model."):
                name = name.replace("vlm.model.", "model.")

            # Handle visual module weight renaming
            is_visual = "visual" in name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if is_visual:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if is_visual:
                    # Adapt to VisionAttention naming
                    name = name.replace("attn.qkv.", "attn.qkv_proj.")

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = ColQwen3
