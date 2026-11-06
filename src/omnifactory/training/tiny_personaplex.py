"""Random native 17-stream PersonaPlex fixture, never a speech-quality model."""

from omnifactory.adapters.personaplex_model import DELAYS, PersonaPlexTrainingModel


def tiny_personaplex(**overrides):
    params = dict(
        delays=list(DELAYS),
        n_q=16,
        dep_q=16,
        card=32,
        text_card=64,
        existing_text_padding_id=3,
        dim=32,
        num_heads=2,
        num_layers=2,
        hidden_scale=4,
        context=32,
        causal=True,
        norm="rms_norm_f32",
        gating="silu",
        positional_embedding="rope",
        depformer_dim=16,
        depformer_dim_feedforward=32,
        depformer_num_heads=2,
        depformer_num_layers=2,
        depformer_causal=True,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_pos_emb="none",
        depformer_gating="silu",
    )

    params.update(overrides)
    return PersonaPlexTrainingModel(**params)


def tiny_mimi():
    import torch

    from omnifactory._vendor.personaplex.models.compression import MimiModel
    from omnifactory._vendor.personaplex.models.loaders import _seanet_kwargs, _transformer_kwargs
    from omnifactory._vendor.personaplex.modules import SEANetDecoder, SEANetEncoder
    from omnifactory._vendor.personaplex.modules.transformer import ProjectedTransformer
    from omnifactory._vendor.personaplex.quantization import SplitResidualVectorQuantizer

    seanet = {**_seanet_kwargs, "dimension": 16, "n_filters": 4}
    transformer = {
        **_transformer_kwargs,
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
        "input_dimension": 16,
        "output_dimensions": [16],
    }
    encoder = SEANetEncoder(**seanet)
    model = MimiModel(
        encoder,
        SEANetDecoder(**seanet),
        SplitResidualVectorQuantizer(
            dimension=16, n_q=8, bins=2048, input_dimension=16, output_dimension=16
        ),
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        encoder_frame_rate=24000 / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=ProjectedTransformer(**transformer),
        decoder_transformer=ProjectedTransformer(**transformer),
    )
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_(std=0.1)
                module._initialized.fill_(1)
    return model.eval().requires_grad_(False)
