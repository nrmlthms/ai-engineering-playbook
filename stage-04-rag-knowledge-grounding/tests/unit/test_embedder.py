"""
Embedder tests — no network, no model downloads (DeterministicEmbedder only).
"""

import math

from embedder import DeterministicEmbedder, Embedder


def test_deterministic_satisfies_protocol() -> None:
    e = DeterministicEmbedder()
    assert isinstance(e, Embedder)


def test_deterministic_dimension_property() -> None:
    e = DeterministicEmbedder(dimension=64)
    assert e.dimension == 64


def test_deterministic_embed_one_length() -> None:
    e = DeterministicEmbedder(dimension=64)
    vec = e.embed_one("hello world")
    assert len(vec) == 64


def test_deterministic_unit_vector() -> None:
    e = DeterministicEmbedder(dimension=64)
    vec = e.embed_one("test text")
    mag = math.sqrt(sum(x * x for x in vec))
    assert abs(mag - 1.0) < 1e-5


def test_deterministic_consistent_same_text() -> None:
    e = DeterministicEmbedder()
    v1 = e.embed_one("same text")
    v2 = e.embed_one("same text")
    assert v1 == v2


def test_deterministic_different_texts_differ() -> None:
    e = DeterministicEmbedder()
    v1 = e.embed_one("apple")
    v2 = e.embed_one("banana")
    assert v1 != v2


def test_deterministic_embed_batch_length() -> None:
    e = DeterministicEmbedder(dimension=32)
    vecs = e.embed(["text a", "text b", "text c"])
    assert len(vecs) == 3


def test_deterministic_embed_batch_each_correct_dim() -> None:
    e = DeterministicEmbedder(dimension=32)
    vecs = e.embed(["a", "b", "c"])
    assert all(len(v) == 32 for v in vecs)


def test_deterministic_embed_batch_each_unit() -> None:
    e = DeterministicEmbedder(dimension=32)
    for vec in e.embed(["x", "y", "z"]):
        mag = math.sqrt(sum(x * x for x in vec))
        assert abs(mag - 1.0) < 1e-5


def test_deterministic_different_dimensions_give_different_sizes() -> None:
    e64 = DeterministicEmbedder(dimension=64)
    e128 = DeterministicEmbedder(dimension=128)
    assert len(e64.embed_one("text")) == 64
    assert len(e128.embed_one("text")) == 128
