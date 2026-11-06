"""Compatibility shims for container images with older decoders.

Some Modal images ship a ``brotli`` package whose ``Decompressor.process`` does
not accept ``output_buffer_limit``; newer ``httpx2`` always passes that keyword,
so any gzip/brotli-compressed Hugging Face response fails before the model is
loaded. Patch the decoder to call without the keyword.
"""

from __future__ import annotations


def patch_httpx_decoders() -> None:
    try:
        from httpx2 import _decoders as decoders

        brotli = decoders.brotli
    except Exception:
        return
    if brotli is None or getattr(decoders.BrotliDecoder, "_qsa_patched", False):
        return

    def decode(self, data: bytes):
        if not data:
            return
        self.seen_data = True
        try:
            decompressed = self._decompress(data)
            while decompressed:
                yield decompressed
                decompressed = self._decompress(b"")
        except brotli.error as exc:
            raise decoders.DecodingError(str(exc)) from exc

    decoders.BrotliDecoder.decode = decode
    decoders.BrotliDecoder._qsa_patched = True
