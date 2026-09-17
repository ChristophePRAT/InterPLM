"""
Tests for ESM hidden-state intervention via nnsight.

Regression test for a nnsight API-version mismatch: `Envoy.input` and
`.save()` changed behavior between nnsight releases, and
`get_esm_output_with_intervention` needs to match the installed version.
"""

import pytest
import torch
from transformers import EsmForMaskedLM, EsmTokenizer
from nnsight import NNsight

from interplm.sae.intervention import get_esm_output_with_intervention

# Requires downloading a small ESM model
pytestmark = pytest.mark.slow

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
HIDDEN_DIM = 320


@pytest.fixture(scope="module")
def esm_model_and_tokenizer():
    try:
        tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
        model = EsmForMaskedLM.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        pytest.skip(f"Could not load {MODEL_NAME}: {e}")
    return model, tokenizer


@pytest.fixture(scope="module")
def batch_tokens_and_mask(esm_model_and_tokenizer):
    _, tokenizer = esm_model_and_tokenizer
    sequences = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MSEQVQVKVKALPDAQFEVVHSL"]
    encoded = tokenizer(sequences, return_tensors="pt", padding=True)
    return encoded["input_ids"], encoded["attention_mask"]


def test_intervention_without_override_returns_original_hidden_state(
    esm_model_and_tokenizer, batch_tokens_and_mask
):
    model, _ = esm_model_and_tokenizer
    batch_tokens, batch_attn_mask = batch_tokens_and_mask
    nnsight_model = NNsight(model)

    logits, hidden = get_esm_output_with_intervention(
        model, nnsight_model, batch_tokens, batch_attn_mask, hidden_layer_idx=3
    )

    assert hidden.shape == (batch_tokens.shape[0], batch_tokens.shape[1], HIDDEN_DIM)
    assert logits.shape[:2] == batch_tokens.shape


def test_zero_ablation_override_changes_output(
    esm_model_and_tokenizer, batch_tokens_and_mask
):
    """The hidden-state override must actually reach the model: patching with
    zeros should produce different logits than the unmodified forward pass.
    """
    model, _ = esm_model_and_tokenizer
    batch_tokens, batch_attn_mask = batch_tokens_and_mask
    nnsight_model = NNsight(model)
    layer_idx = 3

    orig_logits, orig_hidden = get_esm_output_with_intervention(
        model, nnsight_model, batch_tokens, batch_attn_mask, layer_idx
    )
    zero_logits, _ = get_esm_output_with_intervention(
        model,
        nnsight_model,
        batch_tokens,
        batch_attn_mask,
        layer_idx,
        torch.zeros_like(orig_hidden),
    )

    assert zero_logits.shape == orig_logits.shape
    assert not torch.allclose(orig_logits, zero_logits)
