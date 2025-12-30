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
"""Configuration for ColQwen3 multimodal embedding model."""

from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig


class ColQwen3Config(PretrainedConfig):
    r"""
    Configuration class for ColQwen3 multimodal embedding model.

    ColQwen3 is a multi-vector embedding model based on Qwen3-VL that outputs
    per-token embeddings (Seq_Len x embed_dim) with L2 normalization for
    ColPali-style late-interaction retrieval.

    Args:
        vision_config (`Union[PretrainedConfig, dict]`, *optional*):
            The config object or dictionary of the vision backbone.
        text_config (`Union[PretrainedConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
        embed_dim (`int`, *optional*, defaults to 320):
            The dimension of the output embedding vectors.
        padding_side (`str`, *optional*, defaults to "left"):
            The side on which to pad tokens.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the video prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index for vision content.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index for vision content.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for weight initialization.
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie the word embeddings.
    """

    model_type = "colqwen3"
    sub_configs = {
        "vision_config": Qwen3VLVisionConfig,
        "text_config": Qwen3VLTextConfig,
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        embed_dim=320,
        padding_side="left",
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        initializer_range=0.02,
        tie_word_embeddings=True,
        **kwargs,
    ):
        # Initialize vision config
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        # Initialize text config
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        # ColQwen3-specific parameters
        self.embed_dim = embed_dim
        self.padding_side = padding_side
        self.initializer_range = initializer_range

        # Vision token IDs
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # Get hidden_size from text_config for convenience
        self.hidden_size = self.text_config.hidden_size

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self):
        """Return the text configuration."""
        return self.text_config
