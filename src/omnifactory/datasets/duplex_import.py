"""Import synchronized recordings with explicit, auditable token timing.

Word projection is opt-in. It estimates subword positions inside supplied word
intervals, then quantizes to native 80 ms frames. It is not forced alignment.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from omnifactory.artifacts import sha256
from omnifactory.data.schema import CanonicalSample, Record
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import new_directory, write_json

Time = Annotated[int, Field(ge=0, strict=True)]
Text = Annotated[str, Field(min_length=1)]
FRAME_US = 80000


class Word(Record):
    text: Text
    start_us: Time
    end_us: Time

    @model_validator(mode="after")
    def interval(self):
        if self.end_us <= self.start_us or self.text != self.text.strip():
            raise ValueError("word needs a positive interval and unpadded text")
        return self


class Recording(Record):
    id: Text
    dataset: Text
    group_id: Text
    split: Literal["train", "validation", "test"]
    license: Text
    source_url: Text
    # Declare all participating speakers so a different session ID cannot hide leakage.
    speaker_ids: Annotated[tuple[Text, ...], Field(min_length=2)]
    user_audio: Path | None = None
    agent_audio: Path | None = None
    stereo_audio: Path | None = None
    user_channel: Annotated[int, Field(ge=0, strict=True)] | None = None
    agent_channel: Annotated[int, Field(ge=0, strict=True)] | None = None
    start_us: Time = 0
    end_us: Annotated[int, Field(gt=0, strict=True)]
    transcript: Text
    persona: str | None = None
    alignment: Literal["tokens", "word_uniform"] = "tokens"
    words: tuple[Word, ...] = ()
    token_ids: tuple[Annotated[int, Field(ge=0, strict=True)], ...] = ()
    token_times_us: tuple[Time, ...] = ()

    @model_validator(mode="after")
    def sources(self):
        if self.end_us <= self.start_us:
            raise ValueError("end_us must exceed start_us")
        if len(set(self.speaker_ids)) != len(self.speaker_ids):
            raise ValueError("speaker IDs must be unique")
        if self.stereo_audio is not None:
            if self.user_audio is not None or self.agent_audio is not None:
                raise ValueError("select stereo OR two mono sources")
            if self.user_channel is None or self.agent_channel is None:
                raise ValueError("stereo sources need explicit channel indices")
            if self.user_channel == self.agent_channel:
                raise ValueError("user and agent channels must differ")
        elif (
            self.user_audio is None
            or self.agent_audio is None
            or self.user_channel is not None
            or self.agent_channel is not None
        ):
            raise ValueError("two mono sources required without stereo channel indices")
        if self.alignment == "tokens":
            if self.words or not self.token_ids or len(self.token_ids) != len(self.token_times_us):
                raise ValueError("tokens mode requires one timestamp per token and no words")
        elif self.token_ids or self.token_times_us or not self.words:
            raise ValueError("word_uniform requires word intervals and no token annotations")
        return self


def project_words(transcript, words, tokenizer, *, start_us, end_us, slack_us=0):
    """Find feasible increasing frames without dropping tokens or stretching audio."""
    if slack_us < 0 or slack_us > 320000:
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "word projection slack must be 0..320000 us")
    if transcript != " ".join(w.text for w in words):
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "transcript must equal words joined by spaces")
    if any(w.start_us < start_us or w.end_us > end_us for w in words):
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "word crosses crop boundary")
    if any(a.end_us > b.start_us for a, b in zip(words, words[1:])):
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "agent word intervals overlap or are unordered")
    proto = tokenizer.encode(transcript, out_type="immutable_proto")
    spans, position = [], 0
    for w in words:
        spans.append((position, position + len(w.text)))
        position += len(w.text) + 1
    owners = []
    for piece in proto.pieces:
        # SentencePiece includes preceding whitespace in a piece's surface.
        begin, end = piece.begin, piece.end
        while begin < end and transcript[begin].isspace():
            begin += 1
        matches = [i for i, (a, b) in enumerate(spans) if a <= begin < end <= b]
        if begin == end and (not piece.surface or piece.surface.isspace()):
            # Standalone/dummy SentencePiece whitespace belongs to the next word.
            matches = [i for i, (a, _) in enumerate(spans) if a == end]
        if len(matches) != 1:
            raise OmniFactoryError("E_TEXT_ALIGNMENT", "token crosses word spans or has no surface")
        owners.append(matches[0])
    if not owners or set(owners) != set(range(len(words))):
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "every word needs a tokenizer surface")
    bounds, desired = [], []
    total_frames = (end_us - start_us + FRAME_US - 1) // FRAME_US
    for i, word in enumerate(words):
        indices = [j for j, owner in enumerate(owners) if owner == i]
        for order, _ in enumerate(indices):
            midpoint = word.start_us + (word.end_us - word.start_us) * (order + 0.5) / len(indices)
            desired.append(int((midpoint - start_us) // FRAME_US))
            bounds.append(
                (
                    max(0, (word.start_us - start_us - slack_us) // FRAME_US),
                    min(total_frames - 1, (word.end_us - start_us + slack_us - 1) // FRAME_US),
                )
            )
    latest = [0] * len(bounds)
    following = total_frames
    for i in reversed(range(len(bounds))):
        latest[i] = min(bounds[i][1], following - 1)
        if latest[i] < bounds[i][0]:
            raise OmniFactoryError(
                "E_TEXT_ALIGNMENT", f"word/token density exceeds 80ms capacity near token {i}"
            )
        following = latest[i]
    frames, previous = [], -1
    for i, (lower, _) in enumerate(bounds):
        frame = min(max(desired[i], lower, previous + 1), latest[i])
        frames.append(frame)
        previous = frame
    return (
        tuple(p.id for p in proto.pieces),
        tuple(frames),
        dict(
            method="word_uniform",
            word_count=len(words),
            slack_us=slack_us,
            max_frame_shift=max(abs(a - b) for a, b in zip(desired, frames)),
            token_timing_is_estimated=True,
        ),
    )


def token_alignment(row, tokenizer, *, slack_us):
    if row.alignment == "word_uniform":
        return project_words(
            row.transcript,
            row.words,
            tokenizer,
            start_us=row.start_us,
            end_us=row.end_us,
            slack_us=slack_us,
        )
    if tuple(tokenizer.encode(row.transcript)) != row.token_ids:
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "token IDs differ from complete transcript")
    if any(t < row.start_us or t >= row.end_us for t in row.token_times_us):
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "token timestamp outside crop")
    frames = tuple((t - row.start_us) // FRAME_US for t in row.token_times_us)
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise OmniFactoryError(
            "E_TEXT_ALIGNMENT", "token times collide on 80ms grid or are unordered"
        )
    return row.token_ids, frames, dict(method="tokens", token_timing_is_estimated=False)


def read_tracks(row, source_root, *, max_seconds):
    import numpy as np
    import soundfile as sf

    if row.end_us - row.start_us > max_seconds * 1000000:
        raise OmniFactoryError("E_DUPLEX_TRACK", "crop exceeds configured duration limit")
    sources = {}
    if row.stereo_audio is not None:
        sources = {
            "user": (row.stereo_audio, row.user_channel),
            "agent": (row.stereo_audio, row.agent_channel),
        }
    else:
        sources = {"user": (row.user_audio, 0), "agent": (row.agent_audio, 0)}
    identities, tracks, infos = {}, {}, []
    for role, (relative, channel) in sources.items():
        path = (source_root / relative.expanduser()).resolve()
        before = path.stat()
        info = sf.info(path)
        if row.stereo_audio is None and info.channels != 1:
            raise OmniFactoryError("E_DUPLEX_TRACK", "two-file input requires mono recordings")
        if row.stereo_audio is not None and info.channels != 2:
            raise OmniFactoryError("E_DUPLEX_TRACK", "stereo input requires exactly two channels")
        if channel >= info.channels:
            raise OmniFactoryError("E_DUPLEX_TRACK", "channel index exceeds source channels")
        # Reject sub-sample boundaries rather than silently shifting one channel.
        if (row.start_us * info.samplerate) % 1000000 or (row.end_us * info.samplerate) % 1000000:
            raise OmniFactoryError("E_DUPLEX_TRACK", "crop boundaries must lie on source samples")
        start, end = (
            row.start_us * info.samplerate // 1000000,
            row.end_us * info.samplerate // 1000000,
        )
        if end > info.frames:
            raise OmniFactoryError("E_DUPLEX_TRACK", "crop exceeds recording length")
        audio, rate = sf.read(path, start=start, stop=end, dtype="float32", always_2d=True)
        track = audio[:, channel]
        if not np.isfinite(track).all():
            raise OmniFactoryError("E_DUPLEX_TRACK", "nonfinite audio samples")
        tracks[role] = track
        infos.append((info.samplerate, info.frames))
        identities[role] = dict(
            path=str(path),
            sha256=sha256(path),
            channel=channel,
            sample_rate=rate,
            source_frames=info.frames,
            start_sample=start,
            end_sample=end,
        )
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OmniFactoryError("E_SOURCE_CHANGED", f"recording changed while reading: {path}")
    if infos[0] != infos[1]:
        raise OmniFactoryError(
            "E_DUPLEX_TRACK", "tracks must have identical source rate and length"
        )
    if np.array_equal(tracks["user"], tracks["agent"]):
        raise OmniFactoryError(
            "E_DUPLEX_TRACK", "identical user/agent crops; check source channels"
        )
    return tracks, infos[0][0], identities


def import_duplex(source, output, tokenizer, *, word_slack_us=0, max_seconds=300):
    import sentencepiece
    import soundfile as sf

    source, output, tokenizer = Path(source), Path(output), Path(tokenizer)
    if max_seconds <= 0:
        raise OmniFactoryError("E_DUPLEX_TRACK", "max_seconds must be positive")
    sp = sentencepiece.SentencePieceProcessor(str(tokenizer))
    tokenizer_hash = sha256(tokenizer)
    source_bytes = source.read_bytes()
    rows = [
        Recording.model_validate_json(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({r.id for r in rows}) != len(rows):
        raise OmniFactoryError("E_IMPORT", "nonempty records with unique IDs required")
    groups, speakers, source_splits = {}, {}, {}
    alignments, details, counts = [], [], {}
    with new_directory(output) as root:
        (root / "media").mkdir()
        for index, row in enumerate(rows):
            for key, bucket in [
                ((row.dataset, row.group_id), groups),
                *((speaker, speakers) for speaker in row.speaker_ids),
            ]:
                if key in bucket and bucket[key] != row.split:
                    raise OmniFactoryError("E_SPLIT_LEAK", f"group/speaker {key} spans splits")
                bucket[key] = row.split
            try:
                ids, frames, timing = token_alignment(row, sp, slack_us=word_slack_us)
                tracks, rate, identities = read_tracks(row, source.parent, max_seconds=max_seconds)
            except OmniFactoryError as error:
                raise OmniFactoryError(error.code, f"sample {row.id}: {error}") from error
            assets = {}
            for role in ("user", "agent"):
                # A physical source may not occur in different splits even under aliases.
                digest = identities[role]["sha256"]
                if digest in source_splits and source_splits[digest] != row.split:
                    raise OmniFactoryError(
                        "E_SPLIT_LEAK", "recording bytes occur in different splits"
                    )
                source_splits[digest] = row.split
                path = root / "media" / f"{index:06d}-{role}.wav"
                sf.write(path, tracks[role], rate, subtype="FLOAT")
                assets[role] = dict(
                    uri=str(path.relative_to(root)),
                    modality="audio",
                    sha256=sha256(path),
                    sample_rate_hz=rate,
                )
            messages = []
            if row.persona:
                messages.append(
                    dict(
                        id="system",
                        role="system",
                        content=[dict(id="persona", type="text", text=row.persona)],
                    )
                )
            for role, canonical_role in (("user", "user"), ("agent", "assistant")):
                content = [
                    dict(
                        id=role + "_audio",
                        type="audio",
                        media=dict(asset_id=role),
                        placement=dict(timeline_id="session", offset_us=0),
                    )
                ]
                if role == "agent":
                    content.append(dict(id="transcript", type="text", text=row.transcript))
                messages.append(dict(id=canonical_role, role=canonical_role, content=content))
            sample = CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=row.id,
                    provenance=dict(
                        dataset=row.dataset,
                        source_id=row.id,
                        group_id=row.group_id,
                        split=row.split,
                    ),
                    assets=assets,
                    messages=messages,
                    predictions=[
                        dict(
                            id="p0",
                            target_message_id="assistant",
                            target_block_ids=["agent_audio", "transcript"],
                        )
                    ],
                    speech_pairs=[dict(text_block_id="transcript", audio_block_id="agent_audio")],
                )
            )
            with (root / f"{row.split}.jsonl").open("a") as stream:
                stream.write(sample.model_dump_json() + "\n")
            alignments.append(
                dict(
                    sample_id=row.id,
                    prediction_id="p0",
                    token_ids=ids,
                    frames=frames,
                    tokenizer_sha256=tokenizer_hash,
                    audio_sha256=assets["agent"]["sha256"],
                    text_sha256=hashlib.sha256(row.transcript.encode()).hexdigest(),
                )
            )
            counts[row.split] = counts.get(row.split, 0) + 1
            details.append(
                dict(
                    id=row.id,
                    license=row.license,
                    source_url=row.source_url,
                    speaker_ids=row.speaker_ids,
                    sources=identities,
                    start_us=row.start_us,
                    end_us=row.end_us,
                    timing=timing,
                )
            )
        write_json(root / "text-alignment.json", alignments)
        write_json(root / "source-records.json", [r.model_dump(mode="json") for r in rows])
        report = dict(
            format="omnifactory-duplex-import-v1",
            counts=counts,
            source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
            tokenizer_sha256=tokenizer_hash,
            records=details,
            audio_transformation="common crop, float32 mono WAV; no time warping or resampling",
            quality_verified=False,
        )
        write_json(root / "import-report.json", report)
    return dict(
        output=str(output.resolve()),
        counts=counts,
        alignment=str((output / "text-alignment.json").resolve()),
    )
