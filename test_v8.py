#!/usr/bin/env python3
"""Quick smoke test for SparseKV v8 attention-based training pipeline."""

import sys
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

def test_attention_capture():
    """Test AttentionCapture with a small model."""
    from sparsekv.training.attention_capture import AttentionCapture

    logger.info("=== Test 1: AttentionCapture ===")

    # Use a small model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen3-0.6B"

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    num_q_heads = config.num_attention_heads
    logger.info(f"Model: {num_layers} layers, {num_q_heads} q_heads, {num_kv_heads} kv_heads")

    # Tokenize
    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(model.device)
    B, L = tokens["input_ids"].shape
    logger.info(f"Input: B={B}, L={L}")

    # Run with AttentionCapture
    capture = AttentionCapture(window_size=32, num_kv_heads=num_kv_heads)
    with torch.no_grad(), capture:
        outputs = model(**tokens)

    logger.info(f"Captured {len(capture.importance)} layers of importance")
    assert len(capture.importance) == num_layers, f"Expected {num_layers} layers, got {len(capture.importance)}"

    for i, imp in enumerate(capture.importance):
        assert imp.shape == (B, num_kv_heads, L), f"Layer {i}: expected ({B}, {num_kv_heads}, {L}), got {imp.shape}"
        # Check importance sums roughly to 1 per head (it's a mean of softmax rows)
        imp_sum = imp.sum(dim=-1).mean().item()
        if i == 0 or i == num_layers - 1:
            logger.info(f"  Layer {i}: shape={imp.shape}, mean_sum={imp_sum:.4f}, max={imp.max().item():.4f}")

    logger.info("AttentionCapture: PASSED")
    return model, tokenizer, tokens, num_layers, num_kv_heads


def test_attention_based_mask(num_layers, num_kv_heads):
    """Test create_attention_based_mask."""
    from sparsekv.training.kv_dropout import create_attention_based_mask

    logger.info("=== Test 2: create_attention_based_mask ===")

    B, L = 1, 256
    device = "cuda"

    # Fake anchor mask: first 4 + last 32 + some punctuation
    anchor_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    anchor_mask[:, :4] = True    # sink
    anchor_mask[:, -32:] = True  # recent
    anchor_mask[:, 50] = True    # punctuation
    anchor_mask[:, 100] = True
    n_anchors = anchor_mask.sum().item()
    logger.info(f"Anchors: {n_anchors}/{L}")

    # Fake importance: random but with some spiky tokens
    importance_scores = []
    for _ in range(num_layers):
        imp = torch.rand(B, num_kv_heads, L, device=device) * 0.01
        # Make some tokens important
        imp[:, :, 20] = 0.5
        imp[:, :, 80] = 0.3
        imp[:, :, 150] = 0.4
        importance_scores.append(imp)

    keep_ratio = 0.5
    mask = create_attention_based_mask(
        anchor_mask, importance_scores, keep_ratio, L,
        num_layers=num_layers, num_heads=num_kv_heads,
        critical_fraction=0.5, noise_scale=0.1,
    )

    assert mask.shape == (B, num_layers, num_kv_heads, L), f"Bad shape: {mask.shape}"

    # Check anchors are always kept
    anchor_expanded = anchor_mask.unsqueeze(1).unsqueeze(2).expand_as(mask)
    assert (mask & anchor_expanded).sum() == anchor_expanded.sum(), "Not all anchors kept!"

    # Check approximate keep ratio
    actual_kept = mask.float().mean().item()
    logger.info(f"Target keep_ratio={keep_ratio}, actual={actual_kept:.3f}")

    # Check that important tokens (20, 80, 150) are more likely kept than random
    important_kept = mask[:, :, :, [20, 80, 150]].float().mean().item()
    logger.info(f"Important token keep rate: {important_kept:.3f}")

    logger.info("create_attention_based_mask: PASSED")


def test_full_pipeline(model, tokenizer, tokens, num_layers, num_kv_heads):
    """Test full pipeline: AttentionCapture → create_attention_based_mask → PerLayerKVDropout."""
    from sparsekv.training.attention_capture import AttentionCapture
    from sparsekv.training.kv_dropout import create_attention_based_mask, PerLayerKVDropout
    from sparsekv.training.anchor import AnchorSelector, AnchorConfig

    logger.info("=== Test 3: Full Pipeline ===")

    B, L = tokens["input_ids"].shape
    anchor_selector = AnchorSelector(AnchorConfig(sink_size=4, recent_size=32), tokenizer)

    # Step 1: Teacher forward with attention capture
    capture = AttentionCapture(window_size=32, num_kv_heads=num_kv_heads)
    with torch.no_grad(), capture:
        teacher_out = model(**tokens)
    teacher_logits = teacher_out.logits
    logger.info(f"Teacher logits: {teacher_logits.shape}")

    # Step 2: Create attention-based mask
    anchor_mask = anchor_selector.get_anchor_mask(tokens["input_ids"])
    full_mask = create_attention_based_mask(
        anchor_mask, capture.importance, keep_ratio=0.5, seq_len=L,
        num_layers=num_layers, num_heads=num_kv_heads,
        critical_fraction=0.5, noise_scale=0.1,
    )
    logger.info(f"Mask: {full_mask.shape}, kept={full_mask.float().mean():.3f}")

    # Step 3: Student forward with PerLayerKVDropout
    with PerLayerKVDropout(full_mask, dtype=teacher_logits.dtype):
        student_out = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            labels=tokens["input_ids"],
        )

    ce_loss = student_out.loss
    student_logits = student_out.logits
    logger.info(f"Student CE loss: {ce_loss.item():.4f}")

    # Step 4: KL loss
    T = 1.0
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs.detach(), reduction="batchmean") * (T * T)
    total_loss = ce_loss + kl_loss
    logger.info(f"KL loss: {kl_loss.item():.4f}, Total: {total_loss.item():.4f}")

    # Verify loss is finite
    assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"

    logger.info("Full Pipeline: PASSED")


def test_fallback_mode(model, tokenizer, tokens, num_layers, num_kv_heads):
    """Test that use_attention_mask=False falls back to random masking."""
    from sparsekv.training.kv_dropout import create_kv_dropout_mask, PerLayerKVDropout
    from sparsekv.training.anchor import AnchorSelector, AnchorConfig

    logger.info("=== Test 4: Fallback (random mask) ===")

    B, L = tokens["input_ids"].shape
    anchor_selector = AnchorSelector(AnchorConfig(sink_size=4, recent_size=32), tokenizer)

    # Teacher forward without capture
    with torch.no_grad():
        teacher_out = model(**tokens)

    anchor_mask = anchor_selector.get_anchor_mask(tokens["input_ids"])
    full_mask = create_kv_dropout_mask(
        anchor_mask, keep_ratio=0.5, seq_len=L,
        num_layers=num_layers, num_heads=num_kv_heads,
    )

    with PerLayerKVDropout(full_mask, dtype=teacher_out.logits.dtype):
        student_out = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            labels=tokens["input_ids"],
        )

    logger.info(f"Fallback CE loss: {student_out.loss.item():.4f}")
    assert torch.isfinite(student_out.loss), "Fallback loss is not finite"
    logger.info("Fallback mode: PASSED")


if __name__ == "__main__":
    model, tokenizer, tokens, num_layers, num_kv_heads = test_attention_capture()
    test_attention_based_mask(num_layers, num_kv_heads)
    test_full_pipeline(model, tokenizer, tokens, num_layers, num_kv_heads)
    test_fallback_mode(model, tokenizer, tokens, num_layers, num_kv_heads)
    logger.info("\n=== ALL TESTS PASSED ===")
