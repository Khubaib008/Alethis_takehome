"""Build a small test folder of damaged / unusual inputs from one real FineVideo video.

    python make_broken_data.py --source finevideo/data/train-00000-of-01357.parquet --out testdata

Writes these Parquet files into --out (each case is short, so even CPU inference is quick):

  cases.parquet  rows:
    0 good          3-second clip, stream-copied from the source video          -> normal lines
    1 empty         b"" bytes                                                    -> error line
    2 null          missing value                                                -> error line
    3 garbage       random bytes                                                 -> error line
    4 truncated     first 60% of the good clip's bytes                           -> lines and/or error
    5 corrupted     good clip with random bytes stamped over its middle         -> lines (maybe glitchy), maybe error
    6 rotated       good clip tagged with a 90-degree rotation (if the muxer honours it)
    7 audio_only    the source's audio stream only, no video                     -> error line
  corrupt_file.parquet   random bytes, not Parquet at all                        -> one error line
  no_mp4_column.parquet  valid Parquet but only a `json` column (2 rows)         -> error line per row
  zero_rows.parquet      valid Parquet, 0 rows                                   -> nothing

expected.txt documents the same table next to the files.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def remux(src: bytes, seconds: float, kind: str = "video", rotate: int | None = None) -> bytes:
    """Stream-copy the first `seconds` of the video (or audio) stream into a new mp4."""
    inp = av.open(io.BytesIO(src))
    s_in = inp.streams.video[0] if kind == "video" else inp.streams.audio[0]
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    s_out = out.add_stream_from_template(s_in)
    if rotate is not None:
        s_out.metadata["rotate"] = str(rotate)
    for p in inp.demux(s_in):
        if p.dts is None:
            continue
        if p.dts * s_in.time_base >= seconds:
            break
        p.stream = s_out
        out.mux(p)
    out.close()
    inp.close()
    return buf.getvalue()


def set_tkhd_rotation(mp4: bytes, degrees: int = 90) -> bytes:
    """Write a rotation display matrix into the video track's tkhd box (what phones do)."""
    import struct

    data = bytearray(mp4)
    i = data.find(b"tkhd")
    if i < 0:
        raise ValueError("no tkhd box")
    version = data[i + 4]
    # tkhd body: version/flags(4) + times/ids (v0: 20 bytes, v1: 32 bytes) + reserved(8)
    # + layer(2) + alt group(2) + volume(2) + reserved(2), then the 3x3 matrix (9 x int32)
    m_off = i + 4 + 4 + (32 if version == 1 else 20) + 8 + 8
    one, zero = 0x10000, 0
    mats = {90: (zero, one, 0, -one, zero, 0, 0, 0, 0x40000000),
            180: (-one, zero, 0, zero, -one, 0, 0, 0, 0x40000000),
            270: (zero, -one, 0, one, zero, 0, 0, 0, 0x40000000)}
    struct.pack_into(">9i", data, m_off, *mats[degrees])
    return bytes(data)


def rotation_of(mp4: bytes) -> int:
    with av.open(io.BytesIO(mp4)) as c:
        fr = next(c.decode(video=0))
        return int(getattr(fr, "rotation", 0) or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="finevideo/data/train-00000-of-01357.parquet")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="testdata")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = pq.read_table(args.source, columns=["mp4"]).column("mp4")[args.row].as_py()

    good = remux(src, args.seconds)
    truncated = good[: int(len(good) * 0.6)]
    corrupted = bytearray(good)
    mid = len(corrupted) // 2
    corrupted[mid: mid + 4096] = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
    rotated = set_tkhd_rotation(good, 90)
    rot = rotation_of(rotated)
    try:
        audio_only = remux(src, args.seconds, kind="audio")
    except Exception:
        audio_only = b"\x00" * 1000

    cases = [
        ("good", good), ("empty", b""), ("null", None),
        ("garbage", rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes()),
        ("truncated", truncated), ("corrupted", bytes(corrupted)),
        ("rotated", rotated), ("audio_only", audio_only),
    ]
    table = pa.table({
        "mp4": pa.array([b for _, b in cases], type=pa.binary()),
        "json": pa.array([f'{{"case": "{n}"}}' for n, _ in cases], type=pa.string()),
    })
    pq.write_table(table, out / "cases.parquet")

    (out / "corrupt_file.parquet").write_bytes(rng.integers(0, 256, 50_000, dtype=np.uint8).tobytes())
    pq.write_table(pa.table({"json": pa.array(['{"a": 1}', '{"a": 2}'])}), out / "no_mp4_column.parquet")
    pq.write_table(pa.table({"mp4": pa.array([], type=pa.binary()), "json": pa.array([], type=pa.string())}),
                   out / "zero_rows.parquet")

    lines = [f"cases.parquet:{i}  {n:<11} {len(b) if b is not None else 'None':>9} bytes" for i, (n, b) in enumerate(cases)]
    lines.append(f"rotated clip reports rotation = {rot} (0 means the muxer ignored the tag)")
    lines += ["corrupt_file.parquet  random bytes -> one error line",
              "no_mp4_column.parquet  2 rows, no mp4 column -> 2 error lines",
              "zero_rows.parquet  0 rows -> no lines"]
    (out / "expected.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote test data to {out.resolve()}")


if __name__ == "__main__":
    main()
