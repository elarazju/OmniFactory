"""CPU/device-neutral reference execution of the native MiniCPM Token2Wav modules.

Uses official pretrained assets and the stepaudio2-minicpmo implementation.
The upstream wrapper hard-codes CUDA; this wrapper retains its offline math.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from omnifactory.adapters.speech import S3Tokenizer, file_sha256
from omnifactory.errors import OmniFactoryError

ASSET_FILES = ("speech_tokenizer_v2_25hz.onnx", "campplus.onnx", "flow.pt", "flow.yaml", "hift.pt")
ASSET_REVISION = "eb4141de24550a3a1a4082f718be58f46b761728"


def fetch_assets(output):
    import shutil

    from huggingface_hub import hf_hub_download

    from omnifactory.training.artifacts import new_directory, write_json

    with new_directory(output) as root:
        hashes = {}
        for name in ASSET_FILES:
            source = hf_hub_download(
                "openbmb/MiniCPM-o-4_5", "assets/token2wav/" + name, revision=ASSET_REVISION
            )
            shutil.copy2(source, root / name)
            hashes[name] = file_sha256(root / name)
        write_json(
            root / "omnifactory_assets.json",
            {"model": "openbmb/MiniCPM-o-4_5", "revision": ASSET_REVISION, "sha256": hashes},
        )
    return {"assets": str(output.resolve()), "revision": ASSET_REVISION, "sha256": hashes}


def verify_assets(root):
    root = Path(root)
    metadata = json.loads((root / "omnifactory_assets.json").read_text())
    if set(metadata["sha256"]) != set(ASSET_FILES):
        raise OmniFactoryError(
            "E_SPEECH_ASSETS", "asset manifest does not cover native decoder files"
        )
    for name in ASSET_FILES:
        if file_sha256(root / name) != metadata["sha256"][name]:
            raise OmniFactoryError("E_SPEECH_ASSETS", f"changed decoder asset: {name}")
    return metadata


def load_native_flow(root, *, device="cpu"):
    """Load the pinned flow module without the vocoder or ONNX sessions."""
    import yaml
    from stepaudio2.cosyvoice2.flow.decoder_dit import DiT
    from stepaudio2.cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
    from stepaudio2.cosyvoice2.flow.flow_matching import CausalConditionalCFM
    from stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2

    root = Path(root)
    # Whitelisted constructors only; no !apply/import/exec from arbitrary YAML.
    classes = {
        "cosyvoice2.flow.flow.CausalMaskedDiffWithXvec": CausalMaskedDiffWithXvec,
        "cosyvoice2.flow.flow_matching.CausalConditionalCFM": CausalConditionalCFM,
        "cosyvoice2.flow.decoder_dit.DiT": DiT,
        "cosyvoice2.transformer.upsample_encoder_v2.UpsampleConformerEncoderV2": UpsampleConformerEncoderV2,
    }

    class Loader(yaml.SafeLoader):
        pass

    def construct(loader, suffix, node):
        if suffix not in classes:
            raise OmniFactoryError("E_DECODER_CONFIG", f"unsupported decoder class {suffix}")
        return classes[suffix](**loader.construct_mapping(node, deep=True))

    Loader.add_multi_constructor("!new:", construct)
    flow = yaml.load((root / "flow.yaml").read_text(), Loader=Loader)["flow"]
    flow.load_state_dict(
        torch.load(root / "flow.pt", map_location="cpu", weights_only=True), strict=True
    )
    flow.to(device).eval()
    return flow


class NativeToken2Wav:
    def __init__(self, root, *, device="cpu"):
        import onnxruntime as ort
        from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator

        self.root = Path(root)
        info = verify_assets(root)
        self.device = device
        self.codec = S3Tokenizer(self.root / ASSET_FILES[0], info["sha256"][ASSET_FILES[0]])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.speaker = ort.InferenceSession(
            str(self.root / "campplus.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.flow = load_native_flow(self.root, device=device)
        self.hift = HiFTGenerator()
        weights = torch.load(self.root / "hift.pt", map_location="cpu", weights_only=True)
        self.hift.load_state_dict(
            {k.removeprefix("generator."): v for k, v in weights.items()}, strict=True
        )
        self.hift.to(device).eval()

    @torch.no_grad()
    def decode(self, codes, reference_audio):
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

        if not codes or min(codes) < 0 or max(codes) >= 6561:
            raise OmniFactoryError(
                "E_CODEC_TOKENS", "decoder needs nonempty native S3 codes without EOS"
            )
        info = sf.info(reference_audio)
        if info.duration > 30 or info.duration < 0.1:
            raise OmniFactoryError("E_REFERENCE_AUDIO", "reference duration must be 0.1–30 seconds")
        wave, sr = sf.read(reference_audio, dtype="float32", always_2d=True)
        wave = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
        if not torch.isfinite(wave).all():
            raise OmniFactoryError("E_REFERENCE_AUDIO", "nonfinite reference waveform")
        wav16 = torchaudio.functional.resample(wave, sr, 16000).squeeze(0)
        tokens = self.codec.encode(wav16.numpy()).unsqueeze(0).to(self.device, dtype=torch.int32)
        feats = kaldi.fbank(wav16.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        feats -= feats.mean(dim=0, keepdim=True)
        speaker = self.speaker.run(
            None, {self.speaker.get_inputs()[0].name: feats.unsqueeze(0).numpy()}
        )[0]
        speaker = torch.from_numpy(speaker).to(self.device)
        wav24 = torchaudio.functional.resample(wave, sr, 24000)
        mels = mel_spectrogram(wav24).transpose(1, 2).to(self.device)
        length = torch.tensor([mels.shape[1]], device=self.device, dtype=torch.int32)
        mels = F.pad(
            mels, (0, 0, 0, tokens.shape[1] * self.flow.up_rate - mels.shape[1]), mode="replicate"
        )
        result = self.flow.inference(
            torch.tensor([codes], device=self.device, dtype=torch.int32),
            torch.tensor([len(codes)], device=self.device, dtype=torch.int32),
            tokens,
            torch.tensor([tokens.shape[1]], device=self.device, dtype=torch.int32),
            mels,
            length,
            speaker,
            10,
        )
        waveform, _ = self.hift(speech_feat=result)
        waveform = waveform.squeeze().cpu().numpy()
        if not len(waveform) or not np.isfinite(waveform).all():
            raise OmniFactoryError("E_WAVEFORM", "decoder produced invalid waveform")
        return waveform, 24000
