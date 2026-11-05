"""Atomic, no-overwrite publication of persistent sibling bundle directories."""

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import shlex
import shutil
import stat
import tempfile


PUBLICATION_LAYOUT = "relative-sibling-symlink-v1"


def logical_output(path):
    """Resolve the parent while preserving the final publication name."""
    path = Path(os.path.abspath(path))
    return path.parent.resolve() / path.name


def directory_identity(path):
    info = Path(path).lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Backing directory must be a real directory: {path}")
    return {"device": info.st_dev, "inode": info.st_ino}


def publication_matches(backing, output, identity):
    """An unrelated link or a replaced backing directory never grants ownership."""
    try:
        return (
            os.readlink(output) == backing.name
            and backing.parent == output.parent
            and directory_identity(backing) == identity
        )
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.EINVAL, errno.ENOTDIR}:
            return False
        raise


def publish_bundle(backing, output, identity):
    """Create one complete-bundle reference, refusing any different destination."""
    if backing.parent != output.parent or directory_identity(backing) != identity:
        raise ValueError("Publication requires the owned sibling backing directory.")
    try:
        os.symlink(backing.name, output, target_is_directory=True)
    except OSError as exc:
        # A remote filesystem may commit the operation before reporting an
        # error. Reconcile only a link to this exact owned directory.
        try:
            if publication_matches(backing, output, identity):
                return
        except OSError as inspection:
            exc.add_note(f"Could not reconcile publication: {inspection}")
        raise


def probe_publication_parent(parent):
    """Exercise the actual parent filesystem before copying deployment data."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".bundle-publication-probe-", dir=parent) as temporary:
            root = Path(temporary)
            backing, other, output = root / "backing", root / "other", root / "published"
            backing.mkdir()
            other.mkdir()
            (backing / "sentinel").write_text("verified")
            publish_bundle(backing, output, directory_identity(backing))
            if (output / "sentinel").read_text() != "verified":
                raise RuntimeError("Relative symlink traversal did not reach the probe file.")
            try:
                publish_bundle(other, output, directory_identity(other))
            except FileExistsError:
                pass
            else:
                raise RuntimeError("Publication probe did not preserve an existing destination.")
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"Bundle publication preflight failed in {parent}: relative symlink "
            f"creation/traversal with no overwrite is required. {exc}", str(parent),
        ) from exc


def preflight_output(path):
    output = logical_output(path)
    if os.path.lexists(output):
        raise FileExistsError(errno.EEXIST, "Deployment output already exists", str(output))
    probe_publication_parent(output.parent)
    return output


def retry_command(backing, output):
    return shlex.join([
        "python", "-m", "maths_ai.gnn_inference.scripts.publish_source_aware_bundle",
        "--backing", str(backing), "--output", str(output),
    ])


class RetainedBundleError(RuntimeError):
    """Validation completed; publication failed and its backing data is retained."""

    def __init__(self, backing, output, cause):
        self.backing, self.output = backing, output
        super().__init__(
            f"Bundle was validated but publication failed: {cause}\n"
            f"Verified data retained at: {backing}\n"
            f"Retry without copying artifacts again:\n{retry_command(backing, output)}"
        )


@dataclass
class BundlePublication:
    """The migration owns incomplete storage and transfers verified storage to publication."""

    output: Path
    backing: Path
    identity: dict
    validated: bool = False

    @classmethod
    def create(cls, output):
        backing = Path(tempfile.mkdtemp(prefix=f".{output.name}.bundle-", dir=output.parent))
        return cls(output, backing, directory_identity(backing))

    def metadata(self):
        return {
            "layout": PUBLICATION_LAYOUT, "output": str(self.output),
            "backing": self.backing.name, "identity": self.identity,
        }

    def publish(self):
        if not self.validated:
            raise RuntimeError("An incomplete bundle cannot be published.")
        try:
            publish_bundle(self.backing, self.output, self.identity)
        except Exception as exc:
            raise RetainedBundleError(self.backing, self.output, exc) from exc
        except BaseException as exc:
            exc.add_note(f"Verified bundle retained. Retry: {retry_command(self.backing, self.output)}")
            raise

    def cleanup(self):
        if self.validated:
            return
        try:
            if publication_matches(self.backing, self.output, self.identity):
                return
            if directory_identity(self.backing) != self.identity:
                return
        except (OSError, ValueError):
            # Uncertain ownership must not cause deletion of somebody else's
            # directory or a bundle already reachable through publication.
            return
        shutil.rmtree(self.backing)
