import torch

from examples.measure_act_halt import halt_steps_from_probabilities


def test_halt_steps_record_first_threshold_crossing():
    p1 = torch.tensor([[0.6, 0.2]])
    p2 = torch.tensor([[0.1, 0.9]])
    p3 = torch.tensor([[0.5, 0.05]])

    steps = halt_steps_from_probabilities([p1, p2, p3], threshold=0.99)

    # Position 0 crosses at loop 2 (0.6 + 0.1... no: 0.6 < 0.99, then 0.7 < 0.99,
    # then 1.2 >= 0.99 → step 3). Position 1 crosses at loop 2 (0.2 + 0.9 >= 0.99).
    assert steps.tolist() == [[3, 2]]


def test_halt_steps_report_never_halted_as_loop_count():
    p1 = torch.tensor([[0.1]])
    p2 = torch.tensor([[0.1]])

    steps = halt_steps_from_probabilities([p1, p2], threshold=0.99)

    assert steps.tolist() == [[2]]
