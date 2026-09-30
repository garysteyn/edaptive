from pathlib import Path
import importlib.util


def patch_tf_agents():
    spec = importlib.util.find_spec("tf_agents")

    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("tf_agents is not installed")

    tf_agents_dir = Path(spec.submodule_search_locations[0])
    path = tf_agents_dir / "typing" / "types.py"

    text = path.read_text()

    old = "Bool = Union[bool, np.bool, Tensor, Array]"
    new = "Bool = Union[bool, np.bool_, Tensor, Array]"

    if old in text:
        path.write_text(text.replace(old, new))
        print(f"Patched {path}")
    elif new in text:
        print(f"Already patched: {path}")
    else:
        raise RuntimeError(
            f"Could not find expected TF-Agents line in {path}"
        )


if __name__ == "__main__":
    patch_tf_agents()