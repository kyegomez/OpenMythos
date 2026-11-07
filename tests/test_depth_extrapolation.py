import torch

from examples import depth_extrapolation as dx


def test_make_addition_instance_is_correct_and_reversed():
    rng = torch.Generator().manual_seed(0)
    instance = dx.make_addition_instance(rng, max_digits=4)
    parsed = dx.parse_addition(instance.input_text)

    assert instance.input_text.count("+") == 1
    assert instance.input_text.endswith("=")
    assert parsed is not None
    a, b = parsed
    assert instance.answer_text == str(a + b)[::-1]
    assert 1 <= a <= 9999 and 1 <= b <= 9999


def test_make_parity_instance_counts_ones():
    rng = torch.Generator().manual_seed(1)
    for _ in range(20):
        instance = dx.make_parity_instance(rng, max_bits=32)
        bits = instance.input_text[: -1]  # strip trailing "="
        assert instance.input_text.endswith("=")
        assert set(bits) <= {"0", "1"}
        assert instance.answer_text == str(bits.count("1") % 2)


def test_encode_instance_masks_prompt_targets():
    instance = dx.TaskInstance(
        input_text="12+3=", answer_text="51", bucket="4-digit"
    )
    seq_len = 12
    x, y, mask = dx.encode_instance(instance, seq_len, dx.EOS_ID)

    prompt_len = len("12+3=")
    answer_len = len("51") + 1  # digits + terminating EOS
    assert mask[: prompt_len - 1].sum() == 0  # prompt-internal targets masked
    assert mask[prompt_len - 1 :].sum() == answer_len  # answer digits + one EOS only
    assert mask.sum() == answer_len  # pad targets must stay masked (mask-dilution bug)
    assert y[prompt_len - 1].item() == dx.char_id("5")  # first answer target
    assert x[0].item() == dx.char_id("1")


def test_decode_answer_matches_expected():
    eos = dx.EOS_ID
    ids = [dx.char_id(c) for c in "0801"] + [eos, eos]
    assert dx.decode_answer(ids, eos) == "0801"


def test_check_prediction_scores_exact_match():
    assert dx.check_prediction("0801", "0801") is True
    assert dx.check_prediction("080", "0801") is False
    assert dx.check_prediction("801", "0801") is False  # no silent length tolerance


def test_eval_sweep_runs_on_untrained_tiny_model():
    from open_mythos import MythosConfig, OpenMythos

    cfg = MythosConfig(
        vocab_size=dx.VOCAB_SIZE,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=4,
        dropout=0.0,
    )
    model = OpenMythos(cfg)

    rows = dx.eval_sweep(
        model,
        task="add",
        loops_list=[2, 4],
        n_eval=6,
        max_size=2,
        max_new_tokens=8,
        seed=5,
    )

    assert [row["loops"] for row in rows] == [2, 4]
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)


def test_build_batch_shapes_and_dtype():
    rng = torch.Generator().manual_seed(2)
    x, y, mask = dx.build_batch(
        rng, task="parity", batch_size=4, seq_len=48, max_size=32
    )

    assert x.shape == (4, 48) and y.shape == (4, 48) and mask.shape == (4, 48)
    assert x.dtype == torch.long and mask.dtype == torch.float32
    assert (mask.sum(dim=1) > 0).all()  # every sample has trainable targets
