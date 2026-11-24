"""Replay-buffer check: the det trainer's buffer vs the original's Reverb buffer.

Feeds the same trajectories (episodes of varying length, each ending with the
driver's boundary trajectory LAST -> FIRST) to
- Reverb, set up as in calypso/scripts/rl/softActorCritic.py: a Uniform/Fifo table
  written by ReverbAddTrajectoryObserver(sequence_length=2, stride_length=1), plus a
  mirror table with the same capacity and FIFO remover that is drained in order to
  list exactly the items the uniform table holds;
- the det trainer's buffer (softActorCritic_det: PairObserver, seededDataset).

Checks:
1. items: the det buffer holds the same items, in the same order, as Reverb, with
   and without eviction (capacity), including each episode's last item;
2. no item, stored or sampled, spans two episodes; for contrast, how many 2-step
   windows over the raw trajectory stream do (the rewrite's num_steps=2 sampling);
3. sample shapes and dtypes match;
4. how often each stored item is sampled: chi-square vs uniform for each buffer and
   a chi-square homogeneity test between them;
5. det sampling: same seed -> same rows; batch k uses seed [seed, k];
6. freshness under the trainer's MirroredStrategy (distribute_datasets_from_function
   with the det trainer's EXPERIENCE_DATASET_OPTIONS): add one item, draw one batch (eagerly and inside a
   tf.function, as the Learner does), and check the batch was drawn from every item
   added so far; for contrast, the Reverb dataset with .prefetch's table_size lag.

Run from the repo root: python scratch/sac_replay_buffer_check.py
Writes results/sac_training/replay_buffer_check.json.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import reverb
import tensorflow as tf
from scipy.stats import chi2_contingency, chisquare
from tf_agents.replay_buffers import (
    reverb_replay_buffer,
    reverb_utils,
    tf_uniform_replay_buffer,
)
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "calypso"))
from scripts.rl.softActorCritic_det import (  # noqa: E402
    EXPERIENCE_DATASET_OPTIONS,
    PairObserver,
    pairSpec,
    seededDataset,
)

OUT = REPO / "results" / "sac_training" / "replay_buffer_check.json"
SEED = 7
BATCH = 64

SPEC = trajectory.Trajectory(
    step_type=tf.TensorSpec((), tf.int32, "step_type"),
    observation=tf.TensorSpec((3,), tf.float64, "observation"),
    action=tensor_spec.BoundedTensorSpec((1,), tf.float64, 0.0, 1.0, "action"),
    policy_info=(),
    next_step_type=tf.TensorSpec((), tf.int32, "step_type"),
    reward=tf.TensorSpec((), tf.float64, "reward"),
    discount=tf.TensorSpec((), tf.float64, "discount"),
)


def episode_trajectories(episode, n_transitions):
    """What the py driver passes its observers for one episode: the transitions
    s_t -> s_{t+1} (t < T), then the boundary trajectory s_T (LAST) -> reset (FIRST).
    observation = [episode, t, 0], so every item identifies its states."""
    trajs = []
    for t in range(n_transitions + 1):
        boundary = t == n_transitions
        step_type = (
            ts.StepType.LAST if boundary
            else ts.StepType.FIRST if t == 0 else ts.StepType.MID
        )
        next_step_type = (
            ts.StepType.FIRST if boundary
            else ts.StepType.LAST if t == n_transitions - 1 else ts.StepType.MID
        )
        trajs.append(trajectory.Trajectory(
            step_type=np.asarray(step_type, np.int32),
            observation=np.array([episode, t, 0.0]),
            action=np.array([0.5]),
            policy_info=(),
            next_step_type=np.asarray(next_step_type, np.int32),
            reward=np.asarray(0.0 if boundary else 1000.0 * episode + t + 1),
            discount=np.asarray(1.0 if boundary else 0.0 if t == n_transitions - 1 else 0.99),
        ))
    return trajs


def trajectory_stream(lengths):
    return [traj for e, n in enumerate(lengths) for traj in episode_trajectories(e, n)]


def item_keys(observations):
    """(episode, t) of both states of each item; observations [n, 2, 3]."""
    return [tuple(map(int, o[:, :2].ravel())) for o in np.asarray(observations)]


def spans_two_episodes(keys):
    return [k for k in keys if k[0] != k[2] or k[3] != k[1] + 1]


def make_reverb(capacity):
    tables = [
        reverb.Table(
            name="uniform", sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(), max_size=capacity,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        ),
        reverb.Table(
            name="mirror", sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(), max_size=capacity, max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        ),
    ]
    server = reverb.Server(tables)
    buffer = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec=SPEC, table_name="uniform", sequence_length=2, local_server=server
    )
    observer = reverb_utils.ReverbAddTrajectoryObserver(
        py_client=buffer.py_client, table_name=["uniform", "mirror"],
        sequence_length=2, stride_length=1,
    )
    return server, buffer, observer


def table_size(server, name):
    return server.localhost_client().server_info()[name].current_size


def drain_mirror(server):
    """The mirror table's items, oldest first (Fifo sampler, each sampled once)."""
    n = table_size(server, "mirror")
    samples = server.localhost_client().sample(
        "mirror", num_samples=n, emit_timesteps=False
    )
    return [
        item_keys([tf.nest.pack_sequence_as(SPEC, list(s.data)).observation])[0]
        for s in samples
    ]


def make_det(capacity):
    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=pairSpec(SPEC), batch_size=1, max_length=capacity
    )
    return buffer, PairObserver(buffer)


def det_items(buffer):
    return item_keys(buffer.gather_all().observation.numpy()[0])


def check_items(lengths, capacity):
    trajs = trajectory_stream(lengths)
    server, _, observer = make_reverb(capacity)
    det_buffer, det_observer = make_det(capacity)
    for traj in trajs:
        observer(traj)
        det_observer(traj)
    observer.flush()
    reverb_keys = drain_mirror(server)
    det_keys = det_items(det_buffer)
    observer.close()
    server.stop()
    n_written = sum(lengths)  # one item per transition
    expected = [
        (e, t, e, t + 1) for e, n in enumerate(lengths) for t in range(n)
    ][-capacity:]
    raw_windows = [
        tuple(map(int, np.concatenate([a.observation[:2], b.observation[:2]])))
        for a, b in zip(trajs[:-1], trajs[1:])
    ]
    return {
        "capacity": capacity,
        "items_written": n_written,
        "reverb_items": len(reverb_keys),
        "det_items": len(det_keys),
        "identical_items_in_order": reverb_keys == det_keys,
        "match_expected": det_keys == expected,
        "last_item_of_each_episode_present": all(
            (e, n - 1, e, n) in set(det_keys)
            for e, n in enumerate(lengths)
            if sum(lengths[e + 1:]) + n <= capacity
        ),
        "det_items_spanning_episodes": len(spans_two_episodes(det_keys)),
        "reverb_items_spanning_episodes": len(spans_two_episodes(reverb_keys)),
        "raw_2step_windows": len(raw_windows),
        "raw_2step_windows_spanning_episodes": len(spans_two_episodes(raw_windows)),
    }


def nest_shapes(nest):
    return [(tuple(t.shape), t.dtype.name) for t in tf.nest.flatten(nest)]


def check_sampling(lengths, capacity, n_batches):
    trajs = trajectory_stream(lengths)
    server, reverb_buffer, observer = make_reverb(capacity)
    det_buffer, det_observer = make_det(capacity)
    for traj in trajs:
        observer(traj)
        det_observer(traj)
    observer.flush()
    keys = det_items(det_buffer)
    index = {k: i for i, k in enumerate(keys)}

    reverb_dataset = reverb_buffer.as_dataset(sample_batch_size=BATCH, num_steps=2)
    det_dataset = seededDataset(det_buffer, BATCH, SEED)
    counts = {"reverb": np.zeros(len(keys), int), "det": np.zeros(len(keys), int)}
    spanning = {"reverb": 0, "det": 0}
    shapes = {}
    for name, dataset in (("reverb", reverb_dataset), ("det", det_dataset)):
        for data, _ in dataset.take(n_batches):
            shapes[name] = nest_shapes(data)
            sampled = item_keys(data.observation.numpy())
            spanning[name] += len(spans_two_episodes(sampled))
            for k in sampled:
                counts[name][index[k]] += 1
    observer.close()
    server.stop()

    rows_a = [info[0].numpy() for _, info in seededDataset(det_buffer, BATCH, SEED).take(3)]
    rows_b = [info[0].numpy() for _, info in seededDataset(det_buffer, BATCH, SEED).take(3)]
    rows_c = [info[0].numpy() for _, info in seededDataset(det_buffer, BATCH, SEED + 1).take(3)]
    expected_rows = [
        tf.random.stateless_uniform(
            [BATCH], seed=[SEED, k], minval=0, maxval=len(keys), dtype=tf.int64
        ).numpy()
        for k in range(3)
    ]
    return {
        "items": len(keys),
        "samples_per_buffer": n_batches * BATCH,
        "shapes_match": shapes["reverb"] == shapes["det"],
        "shapes": shapes["det"],
        "sampled_items_spanning_episodes": spanning,
        "chi2_uniform_p": {
            name: float(chisquare(c).pvalue) for name, c in counts.items()
        },
        "chi2_homogeneity_p": float(
            chi2_contingency(np.stack([counts["reverb"], counts["det"]]))[1]
        ),
        "count_range": {
            name: [int(c.min()), int(c.max())] for name, c in counts.items()
        },
        "same_seed_same_rows": all(np.array_equal(a, b) for a, b in zip(rows_a, rows_b)),
        "different_seed_different_rows": not all(
            np.array_equal(a, c) for a, c in zip(rows_a, rows_c)
        ),
        "batch_k_uses_seed_[seed,k]": all(
            np.array_equal(a, e) for a, e in zip(rows_a, expected_rows)
        ),
    }


def check_freshness(n_steps):
    """One item added per learner step, as in the trainer's collect loop."""
    strategy = tf.distribute.MirroredStrategy()
    det_buffer, det_observer = make_det(1000)
    stream = iter(trajectory_stream([10_000]))
    with strategy.scope():
        iterator = iter(strategy.distribute_datasets_from_function(
            lambda _: seededDataset(det_buffer, BATCH, SEED),
            options=EXPERIENCE_DATASET_OPTIONS,
        ))

    @tf.function
    def next_in_function(it):
        return next(it)

    det_observer(next(stream))  # first trajectory of the episode: no item yet
    lags, wrong_rows = [], 0
    for k in range(n_steps):
        det_observer(next(stream))
        n_added = k + 1
        _, (rows, n_items) = next_in_function(iterator) if k % 2 else next(iterator)
        lags.append(n_added - int(n_items))
        expected = tf.random.stateless_uniform(
            [BATCH], seed=[SEED, k], minval=0, maxval=n_added, dtype=tf.int64
        )
        wrong_rows += int(not np.array_equal(rows.numpy(), expected.numpy()))

    # The original: Reverb dataset + .prefetch(50), as in softActorCritic.py.
    server, reverb_buffer, observer = make_reverb(1000)
    stream = iter(trajectory_stream([10_000]))
    for _ in range(3):  # a few items first (MinSize(1) would block on an empty table)
        observer(next(stream))
    observer.flush()
    with strategy.scope():
        reverb_iterator = iter(strategy.distribute_datasets_from_function(
            lambda _: reverb_buffer.as_dataset(
                sample_batch_size=BATCH, num_steps=2
            ).prefetch(50)
        ))
    reverb_lags = []
    for _ in range(n_steps):
        observer(next(stream))
        observer.flush()
        _, info = next(reverb_iterator)
        reverb_lags.append(table_size(server, "uniform") - int(info.table_size.numpy().ravel()[0]))
    observer.close()
    server.stop()
    return {
        "strategy": type(strategy).__name__,
        "det_steps": n_steps,
        "det_max_lag_items": max(lags),
        "det_batches_not_seeded_by_step_index": wrong_rows,
        "reverb_prefetch_lag_items": {
            "min": min(reverb_lags), "median": float(np.median(reverb_lags)),
            "max": max(reverb_lags),
        },
    }


def main():
    rng = np.random.default_rng(0)
    lengths = [int(n) for n in rng.integers(1, 9, size=12)] + [1, 1, 5]
    results = {
        "episode_lengths": lengths,
        "items_no_eviction": check_items(lengths, capacity=1000),
        "items_with_eviction": check_items(lengths, capacity=17),
        "sampling": check_sampling(lengths, capacity=1000, n_batches=1000),
        "sampling_after_eviction": check_sampling(lengths, capacity=17, n_batches=500),
        "freshness": check_freshness(n_steps=200),
    }
    print(json.dumps(results, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    i, e, s, f = (
        results["items_no_eviction"], results["items_with_eviction"],
        results["sampling"], results["freshness"],
    )
    failures = [
        name for name, ok in [
            ("items identical (no eviction)", i["identical_items_in_order"] and i["match_expected"]),
            ("items identical (eviction)", e["identical_items_in_order"] and e["match_expected"]),
            ("episode last items", i["last_item_of_each_episode_present"]),
            ("no stored item spans episodes", i["det_items_spanning_episodes"] == 0),
            ("shapes", s["shapes_match"]),
            ("no sampled item spans episodes", s["sampled_items_spanning_episodes"]["det"] == 0),
            ("seeded rows", s["same_seed_same_rows"] and s["batch_k_uses_seed_[seed,k]"]),
            ("no prefetch (det)", f["det_max_lag_items"] == 0),
            ("batch k seeded with k", f["det_batches_not_seeded_by_step_index"] == 0),
        ] if not ok
    ]
    print("FAILED: " + ", ".join(failures) if failures else "All checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
