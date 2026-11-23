import importlib.util
from pathlib import Path

# (file relative to the tf_agents package, original text, replacement)
PATCHES = [
    # numpy >= 1.24 removed np.bool.
    (
        "typing/types.py",
        "Bool = Union[bool, np.bool, Tensor, Array]",
        "Bool = Union[bool, np.bool_, Tensor, Array]",
    ),
    # The SAC actor's tanh squashing takes its shift and scale from here; use the
    # action spec's dtype instead of always float32, so float64 specs work
    # (float32 specs are unchanged).
    (
        "utils/common.py",
        "  return np.array(\n"
        "      action_means, dtype=np.float32), np.array(\n"
        "          action_magnitudes, dtype=np.float32)\n",
        "  dtype = tf.as_dtype(tf.nest.flatten(action_spec)[0].dtype).as_numpy_dtype\n"
        "  return np.array(\n"
        "      action_means, dtype=dtype), np.array(\n"
        "          action_magnitudes, dtype=dtype)\n",
    ),
    # SAC's entropy temperature: same dtype as the action spec, not always float32.
    (
        "agents/sac/sac_agent.py",
        "        initial_value=initial_log_alpha,\n        dtype=tf.float32,\n",
        "        initial_value=initial_log_alpha,\n"
        "        dtype=tf.nest.flatten(action_spec)[0].dtype,\n",
    ),
    # The regularisation loss (a float32 zero for networks without regularisers)
    # takes the dtype of the loss it is added to.
    (
        "utils/common.py",
        "      total_loss += reg_loss\n",
        "      total_loss += tf.cast(reg_loss, total_loss.dtype)\n",
    ),
]


def patch_tf_agents():
    spec = importlib.util.find_spec("tf_agents")

    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("tf_agents is not installed")

    tf_agents_dir = Path(spec.submodule_search_locations[0])
    for relative_path, old, new in PATCHES:
        path = tf_agents_dir / relative_path
        text = path.read_text()
        if old in text:
            path.write_text(text.replace(old, new))
            print(f"Patched {path}")
        elif new in text:
            print(f"Already patched: {path}")
        else:
            raise RuntimeError(f"Could not find expected TF-Agents text in {path}")


if __name__ == "__main__":
    patch_tf_agents()
