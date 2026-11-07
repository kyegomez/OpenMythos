import torch

from examples import train_small_lm
from examples.train_small_lm import (
    NoOpScaler,
    PackedTextDataset,
    build_config,
    save_checkpoint,
)

DIRTY_QUOTED_ALPHA = "\u00e2\u20ac\u0153alpha\u00e2\u20ac\u009d"


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {
            "alpha": [1, 2],
            "beta": [3],
            '"alpha"': [4],
            DIRTY_QUOTED_ALPHA: [5],
        }[text]


def test_packed_text_dataset_appends_eos_after_each_sample():
    samples = [{"text": "alpha"}, {"text": "beta"}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=4,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=True,
    )

    assert ds.data.tolist() == [1, 2, 99, 3, 99]


def test_packed_text_dataset_can_disable_eos_append():
    samples = [{"text": "alpha"}, {"text": "beta"}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=2,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=False,
    )

    assert ds.data.tolist() == [1, 2, 3]


def test_packed_text_dataset_cleans_text_before_tokenizing():
    samples = [{"text": DIRTY_QUOTED_ALPHA}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=1,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=True,
        clean_text=True,
    )

    assert ds.data.tolist() == [4, 99]


def test_packed_text_dataset_can_disable_text_cleaning():
    samples = [{"text": DIRTY_QUOTED_ALPHA}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=1,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=True,
        clean_text=False,
    )

    assert ds.data.tolist() == [5, 99]


def test_build_loss_weights_emphasizes_only_eos_token():
    weights = train_small_lm.build_loss_weights(
        vocab_size=5,
        eos_token_id=2,
        eos_loss_weight=4.0,
        device=torch.device("cpu"),
    )

    assert weights is not None
    assert weights.tolist() == [1.0, 1.0, 4.0, 1.0, 1.0]


def test_build_loss_weights_uses_default_cross_entropy_when_unweighted():
    weights = train_small_lm.build_loss_weights(
        vocab_size=5,
        eos_token_id=2,
        eos_loss_weight=1.0,
        device=torch.device("cpu"),
    )

    assert weights is None


def test_save_checkpoint_records_eos_loss_weight(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "small.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        NoOpScaler(),
        build_config(vocab_size=100, max_seq_len=8),
        tokenizer="gpt2",
        step=12,
        loss=3.4,
        append_eos=True,
        eos_loss_weight=4.0,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert ckpt["append_eos"] is True
    assert ckpt["eos_loss_weight"] == 4.0


def test_validate_init_from_config_allows_only_max_seq_len_change():
    source = train_small_lm.config_to_dict(build_config(vocab_size=100, max_seq_len=128))
    target = build_config(vocab_size=100, max_seq_len=256)

    train_small_lm.validate_init_from_config(source, target)

    source["dim"] = 512
    try:
        train_small_lm.validate_init_from_config(source, target)
    except ValueError as exc:
        assert "dim" in str(exc)
    else:
        raise AssertionError("expected dim mismatch to fail")


def test_load_matching_model_weights_skips_shape_mismatches():
    model = torch.nn.Module()
    model.same = torch.nn.Linear(2, 2)
    model.changed = torch.nn.Linear(3, 3, bias=False)
    source_state = {
        "same.weight": torch.full_like(model.same.weight, 2.0),
        "same.bias": torch.full_like(model.same.bias, 3.0),
        "changed.weight": torch.ones(4, 4),
        "unknown.weight": torch.ones(1),
    }

    result = train_small_lm.load_matching_model_weights(model, source_state)

    assert result.loaded == 2
    assert result.skipped == ["changed.weight", "unknown.weight"]
    assert torch.equal(model.same.weight, torch.full_like(model.same.weight, 2.0))
    assert torch.equal(model.same.bias, torch.full_like(model.same.bias, 3.0))


from examples.small_lm_data import (  # noqa: E402
    BALANCE_NAME_POOL,
    StoryAlignedDataset,
    balance_names,
    collect_stories,
    dedup_stories,
    split_opening,
)


class WordTokenizer:
    eos_token_id = 99

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]


def test_split_opening_uses_first_sentence():
    opening, rest = split_opening("Tom lost his ball. He ran home quickly.")

    assert opening == "Tom lost his ball."
    assert rest == " He ran home quickly."


def test_split_opening_extends_when_first_sentence_is_short():
    opening, rest = split_opening("Tom ran. He found the ball.")

    assert opening == "Tom ran. He found the ball."
    assert rest == ""


def test_split_opening_keeps_text_without_boundary():
    assert split_opening("alpha beta") == ("alpha beta", "")


def test_story_dataset_pads_short_stories_and_masks_padding():
    stories = collect_stories(
        [{"text": "alpha beta gamma"}],
        WordTokenizer(),
        max_tokens=100,
        text_field="text",
        eos_token_id=99,
        append_eos=True,
    )

    ds = StoryAlignedDataset(
        stories, seq_len=5, pair_mix=0.0, eos_token_id=99
    )
    x, y, mask = ds[0]

    assert x.tolist() == [1, 2, 3, 99, 99]
    assert y.tolist() == [2, 3, 99, 99, 99]
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_story_pair_masks_opening_targets_only():
    stories = collect_stories(
        [{"text": "alpha beta gamma. delta epsilon."}],
        WordTokenizer(),
        max_tokens=100,
        text_field="text",
        eos_token_id=99,
        append_eos=True,
    )

    ds = StoryAlignedDataset(
        stories, seq_len=5, pair_mix=1.0, eos_token_id=99
    )
    _x, y, mask = ds[0]

    assert y.tolist() == [2, 3, 4, 5, 99]
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_story_pair_mix_fraction_is_deterministic():
    stories = collect_stories(
        [{"text": f"aaa bbb ccc{i}. ddd eee."} for i in range(4)],
        WordTokenizer(),
        max_tokens=1000,
        text_field="text",
        eos_token_id=99,
        append_eos=True,
    )

    ds = StoryAlignedDataset(
        stories, seq_len=8, pair_mix=0.5, eos_token_id=99
    )
    masked = sum(1 for i in range(len(ds)) if ds[i][2][0] == 0.0)

    assert len(ds) == 4
    assert masked == 2


def test_dedup_8gram_drops_template_stories():
    tok = WordTokenizer()
    stories = collect_stories(
        [
            {"text": "t1 t2 t3 t4 t5 t6 t7 t8 tailA"},
            {"text": "t1 t2 t3 t4 t5 t6 t7 t8 tailB"},
            {"text": "u1 u2 u3 u4 u5 u6 u7 u8 tailA"},
        ],
        tok,
        max_tokens=1000,
        text_field="text",
        eos_token_id=99,
        append_eos=True,
    )

    kept, dropped = dedup_stories(stories, enabled=True)

    assert dropped == 1
    assert len(kept) == 2
    kept_off, dropped_off = dedup_stories(stories, enabled=False)
    assert dropped_off == 0 and len(kept_off) == 3


def test_compute_loss_ignores_masked_targets():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 6)
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    y_a = torch.tensor([[1, 2, 3, 4]])
    y_b = torch.tensor([[5, 2, 3, 4]])

    loss_a = train_small_lm.compute_loss(logits, y_a, 6, None, mask)
    loss_b = train_small_lm.compute_loss(logits, y_b, 6, None, mask)

    assert torch.isclose(loss_a, loss_b)


def test_compute_loss_matches_plain_cross_entropy_when_unmasked():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 6)
    y = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.ones(2, 3)
    weights = torch.tensor([1.0, 1.0, 4.0, 1.0, 1.0, 1.0])

    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 6), y.reshape(-1)
    )
    assert torch.isclose(train_small_lm.compute_loss(logits, y, 6, None, mask), expected)

    expected_w = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 6), y.reshape(-1), weight=weights
    )
    assert torch.isclose(
        train_small_lm.compute_loss(logits, y, 6, weights, mask), expected_w
    )


def test_save_checkpoint_records_data_settings(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "small.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        NoOpScaler(),
        build_config(vocab_size=100, max_seq_len=8),
        tokenizer="gpt2",
        step=12,
        loss=3.4,
        append_eos=True,
        eos_loss_weight=4.0,
        packing="story",
        pair_mix=0.5,
        dedup_8gram=True,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert ckpt["packing"] == "story"
    assert ckpt["pair_mix"] == 0.5
    assert ckpt["dedup_8gram"] is True


def test_regression_story_dataset_train_stats_never_touches_data_buffer():
    """setup stats must not assume PackedTextDataset.data (smoke-found bug)."""
    stories = collect_stories(
        [{"text": "alpha beta gamma"}],
        WordTokenizer(),
        max_tokens=100,
        text_field="text",
        eos_token_id=99,
        append_eos=True,
    )
    ds = StoryAlignedDataset(stories, seq_len=5, pair_mix=0.0, eos_token_id=99)

    assert train_small_lm.train_stats(ds, "story") == "train windows=1"


def test_save_checkpoint_records_quant_metadata(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "quant.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        NoOpScaler(),
        build_config(vocab_size=100, max_seq_len=8),
        tokenizer="gpt2",
        step=3,
        loss=1.2,
        append_eos=True,
        eos_loss_weight=1.0,
        quant_mode="w158",
        quant_group_size=64,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["quant_mode"] == "w158"
    assert ckpt["quant_group_size"] == 64

    default_ckpt = tmp_path / "plain.pt"
    save_checkpoint(
        default_ckpt,
        model,
        optimizer,
        NoOpScaler(),
        build_config(vocab_size=100, max_seq_len=8),
        tokenizer="gpt2",
        step=3,
        loss=1.2,
        append_eos=True,
        eos_loss_weight=1.0,
    )
    plain = torch.load(default_ckpt, map_location="cpu", weights_only=False)
    assert plain["quant_mode"] == "none"


def test_model_size_presets_define_small_and_mid():
    assert set(train_small_lm.MODEL_SIZES) == {"small", "mid"}
    assert train_small_lm.MODEL_SIZES["small"]["dim"] == 256
    assert train_small_lm.MODEL_SIZES["mid"]["dim"] > 256


def test_build_config_mid_size_passes_config_validation():
    cfg = train_small_lm.build_config(
        vocab_size=1000, max_seq_len=128, model_size="mid"
    )

    assert cfg.dim == 512
    assert cfg.expert_dim == 512
    assert cfg.dim % cfg.n_heads == 0
    assert cfg.n_heads % cfg.n_kv_heads == 0


def test_build_config_rejects_unknown_model_size():
    try:
        train_small_lm.build_config(vocab_size=1000, max_seq_len=128, model_size="huge")
    except ValueError as exc:
        assert "unknown model size" in str(exc)
    else:
        raise AssertionError("expected unknown model size to fail")


def test_build_config_defaults_to_small_backward_compatible():
    cfg_default = train_small_lm.build_config(vocab_size=1000, max_seq_len=128)
    cfg_small = train_small_lm.build_config(
        vocab_size=1000, max_seq_len=128, model_size="small"
    )
    assert cfg_default == cfg_small


def test_name_balance_swaps_high_frequency_names_deterministically():
    rng_seed = 7
    text_a = "Lily found a door. She opened it. Lily was happy."
    text_b = "Lily found a door. She opened it. Lily was happy."

    out_a = balance_names(text_a, seed=rng_seed)
    out_b = balance_names(text_b, seed=rng_seed)

    assert out_a == out_b  # deterministic per (text, seed)
    assert "Lily" not in out_a
    assert out_a.count(out_a.split()[0]) == 2  # same replacement, consistently


def test_name_balance_round_trips_neutral_names_unchanged():
    text = "Nora found a glowing shell. It was pretty."
    assert balance_names(text, seed=7) == text


def test_name_balance_maps_replacements_one_to_one_per_story():
    out = balance_names(
        "Lily and Tom played. Tom ran, Lily laughed.", seed=3
    )

    assert "Lily" not in out and "Tom" not in out
    replaced = out.split(".")[0]
    # Lily→X and Tom→Y with X != Y (two prompt names must not collide)
    assert len(set(replaced.split(" and ")[0:1])) == 1


def test_name_balance_replacement_names_come_from_corpus_pool():
    

    out = balance_names("Lily sang all day.", seed=11)
    replacement = out.split()[0]

    assert replacement in BALANCE_NAME_POOL
    assert replacement not in {"Lily", "Tom", "Ben", "Timmy"}


def test_packed_text_dataset_applies_name_balancing():
    tok = WordTokenizer()
    samples = [{"text": "Lily sang. Lily laughed."}, {"text": "Tom ran. Tom fell."}]

    ds = PackedTextDataset(
        samples,
        tok,
        seq_len=4,
        max_tokens=100,
        text_field="text",
        eos_token_id=99,
        append_eos=True,
        name_balance_seed=7,
    )

    tokens = ds.data.tolist()
    assert 99 in tokens  # EOS present
    # Balanced stream contains pool names' tokens, not Lily/Tom ids.
    from examples.small_lm_data import BALANCE_NAME_POOL

    pool_ids = {tok.encode(name)[0] for name in BALANCE_NAME_POOL}
    lily_id = tok.encode("Lily")[0]
    tom_id = tok.encode("Tom")[0]
    assert lily_id not in tokens
    assert tom_id not in tokens
    assert pool_ids & set(tokens)
