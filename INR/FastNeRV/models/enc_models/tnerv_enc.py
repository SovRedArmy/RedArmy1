from models import register, ParamEnc
from utils import split_str

@register('tnerv_enc')
class TNeRVEnc(ParamEnc):
    """T-NeRV：ParamEnc 的逐层注入版本（对比实验）。
    把 z_tokens 按 decoder['z_tokens'] 拆成多段，在解码器 Transformer 的前若干层逐层注入
    （配合 ViT.forward 的 z_token_list 分支），可选 sequential_decode 与 cumsum_z。"""
    def __init__(self, tokenizer, encoder, decoder, kl_layer, vq_layer):
        self.z_token_list = split_str(decoder['z_tokens'])
        decoder['z_tokens'] = sum(self.z_token_list)
        super().__init__(tokenizer, encoder, decoder, kl_layer, vq_layer)
        self.cumsum_z = decoder['cumsum_z']
        self.sequential_decode = decoder['sequential_decode']
