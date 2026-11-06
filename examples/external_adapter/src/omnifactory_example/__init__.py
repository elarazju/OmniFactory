"""Lazy entry point. Discovery needs only OmniFactory's base dependencies."""


def adapter_spec():
    from dataclasses import replace

    from omnifactory.adapters.runtime import adapter_spec as builtin_spec

    # Reuse an established native recipe to illustrate packaging and hook overrides.
    # A new architecture supplies its own ModelSpec, loader and native exporter.
    source = builtin_spec("qwen2_5_omni")
    return replace(
        source,
        name="example_qwen_fixture",
        implementation="omnifactory_example.native:ExampleAdapter",
        loaders={"understanding_sft": source.loaders["understanding_sft"]},
        loss_components={"understanding_sft": source.loss_components["understanding_sft"]},
        batch_capabilities={"understanding_sft": source.batch_capabilities["understanding_sft"]},
        outputs=("text",),
        audio_unit=None,
        named_speakers=False,
        validator=None,
    )
