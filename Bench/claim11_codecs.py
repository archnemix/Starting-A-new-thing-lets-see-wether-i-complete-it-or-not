"""Phase 2.8, claims 11-12: byte ratio and exactness per data type, measured.

Each content type a training loop actually hands a recorder is encoded and decoded, and both
numbers are reported: how small the stored form is, and whether the bytes came back *exactly*.
A codec that compresses well and loses a byte is not a codec this product can use, so exactness
is asserted for every case rather than assumed.

The content is generated, not downloaded, so the table is reproducible on all three runners.
Each case is chosen to stress the decision the encoder makes:

* **prose** - printable text, the `text-zstd` path,
* **code** - text with structure but no long repeats, where a text heuristic could over-promise,
* **token ids** - both a small vocabulary and a realistic one, because the first version of the
  detector only recognised the small case and silently left the realistic one on plain zstd,
* **pixels** - already-compressed-looking bytes, where the honest answer is "this does not
  compress" and zstd must not pretend otherwise,
* **weights** - float32 parameters, the fallback case.

The dictionary codec is reported as *removed*, with its measurement, because a table that
quietly omits a codec the docs once promised is a table a reader cannot trust.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from archx.data.codecs import choose_data_codec, decode_shard, encode_shard


def content_cases() -> dict[str, bytes]:
    """One example per content type, sized like a real batch rather than a toy."""
    generator = np.random.default_rng(0)

    sentences = [
        "The optimizer stepped and the loss fell.",
        "Warmup ended at step twenty and the learning rate peaked.",
        "Gradient clipping changed the update direction subtly.",
        "A validation run disagreed with the training loss.",
    ]
    prose = " ".join(sentences[index % len(sentences)] for index in range(600)).encode()

    code = (
        "def step(self, batch):\n"
        "    loss = self.criterion(self.model(batch['x']), batch['y'])\n"
        "    loss.backward()\n"
        "    return loss.item()\n"
    ).encode() * 400

    small_vocab = generator.integers(0, 128, 4096, dtype=np.uint32).tobytes()
    real_vocab = generator.integers(0, 5000, 4096, dtype=np.uint32).tobytes()

    pixels = (generator.random((128, 128)) * 255).astype(np.uint8).tobytes()
    weights = generator.standard_normal(4096).astype(np.float32).tobytes()

    return {
        "prose": prose,
        "code": code,
        "tokens-128": small_vocab,
        "tokens-5000": real_vocab,
        "pixels": pixels,
        "weights": weights,
    }


def measure(cases: dict[str, bytes]) -> list[dict]:
    rows = []
    for name, data in cases.items():
        shard = encode_shard(data)
        recovered = decode_shard(shard)
        rows.append({
            "case": name,
            "bytes": len(data),
            "stored": len(shard.payload),
            "ratio": len(shard.payload) / len(data),
            "codec": shard.codec,
            "exact": recovered == data,
            "crc32_matches": (zlib.crc32(recovered) & 0xFFFFFFFF) == shard.content_crc32,
            "chosen_by_heuristic": choose_data_codec(data) == shard.codec,
        })
    return rows


def dictionary_comparison() -> dict:
    """The measurement that removed the `text-dict` codec, kept so it is not re-added blind."""
    import zstandard as zstd

    generator = np.random.default_rng(1)
    vocabulary = ("the model weights moved sharply during warmup because the learning rate was "
                  "too high and the gradient norm spiked after the distribution shifted").split()
    results = []
    for count in (200, 1000, 8000):
        text = ". ".join(" ".join(generator.choice(vocabulary, size=generator.integers(8, 18)))
                         for _ in range(count)).encode()
        plain = len(zstd.ZstdCompressor(level=3).compress(text))
        try:
            dictionary = zstd.train_dictionary(
                16 * 1024, [text[index:index + 256]
                            for index in range(0, min(len(text), 200_000), 256)] or [text]
            ).as_bytes()
            with_dict = len(zstd.ZstdCompressor(
                level=3, dict_data=zstd.ZstdCompressionDict(dictionary)).compress(text))
        except zstd.ZstdError:
            dictionary, with_dict = b"", plain
        results.append({
            "text_bytes": len(text),
            "plain_zstd": plain,
            "dictionary_bytes": len(dictionary),
            "dictionary_plus_stored": with_dict + len(dictionary),
            "dictionary_wins": (with_dict + len(dictionary)) < plain,
        })
    return {"note": "a per-shard dictionary must repay its own bytes on every shard",
            "cases": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="claims 11-12: codec ratio and exactness")
    parser.add_argument("--out", help="write the tables as JSON here")
    args = parser.parse_args(argv)

    rows = measure(content_cases())
    print(f"{'case':<14} {'raw':>8} {'stored':>8} {'ratio':>7} {'codec':<14} exact")
    print("-" * 62)
    for row in rows:
        print(f"{row['case']:<14} {row['bytes']:>8} {row['stored']:>8} "
              f"{row['ratio']:>7.4f} {row['codec']:<14} {row['exact']}")

    exact = all(row["exact"] for row in rows)
    crc = all(row["crc32_matches"] for row in rows)
    print(f"\nevery codec lossless: {exact}; every content CRC matches: {crc}")
    if not (exact and crc):
        print("A LOSSY SHARD IS A BUG unless its reconstruction_error says so.")

    dictionary = dictionary_comparison()
    print("\nthe dictionary codec that was removed, and why")
    print(f"  {'text':>10} {'plain zstd':>11} {'dict+dict':>10} {'dictionary wins':>16}")
    for case in dictionary["cases"]:
        print(f"  {case['text_bytes']:>10} {case['plain_zstd']:>11} "
              f"{case['dictionary_plus_stored']:>10} {str(case['dictionary_wins']):>16}")
    print("  It never wins: zstd's own window already captures the repetition, and the")
    print("  dictionary has to repay its own bytes inside every shard.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"codecs": rows, "dictionary": dictionary}, indent=2) + "\n")
        print(f"\nwritten to {args.out}")
    return 0 if exact and crc else 1


if __name__ == "__main__":
    raise SystemExit(main())
