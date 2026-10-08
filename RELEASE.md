# KVLR Code Release - 2026-10-08

Status: **Released**.

This release restores the KVLR implementation, configuration files, training and
inference scripts, evaluation script, license, and supporting documentation.
The public TODO commit is retained as the parent of this release.

Implementation source: the previously backed-up code snapshot at
`1c68c9f6a1139cc45abeadae52d2b4ddb7bce049`. Runtime source is unchanged from that
snapshot. This is a renewed public release, not a claim of new model training
or new experimental results on this date.

Every distributed file is marked `released` with update date `2026-10-08` in
`release-manifest.json`, with its size and SHA-256 digest. The manifest itself
is excluded from its own file list to avoid a self-referential digest.

Validation: Python syntax compilation and shell syntax checks. GPU training,
inference, and benchmark experiments have not been rerun for this release.

Private backups, local manuscript drafts, and the separate generic evaluation
package are not included in this release. Existing asset licenses and usage
conditions continue to apply.

- [Project page](https://arlo0o.github.io/KVLR-project/)
- [Paper](https://arxiv.org/abs/2605.08712)
- [Demo](https://arlo0o.github.io/KVLR-project/#demo)
