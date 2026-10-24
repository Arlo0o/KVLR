# KASA Data Access

**Last updated: 2026-10-24.**

[Public action annotations and demo](https://drive.google.com/drive/folders/14vHj7IB-pA09QhrYT7D06Ei6A6md3z8k?usp=sharing)

## Current Availability

| Asset | Public access | Contents |
| --- | --- | --- |
| `annotations.zip` | Available | Full released action annotations: per-instance `memory_pool.pth` files grouped by action and clip |
| `KVLR_demo.mp4` | Available | The existing project-page presentation video with selected examples |
| `knotting.zip` | Temporarily restricted | Full knotting videos |
| `needleGrasping.zip` | Temporarily restricted | Full needle-grasping videos |
| `needlePuncture.zip` | Temporarily restricted | Full needle-puncture videos |

We are coordinating data ownership, attribution, and redistribution permissions
with contributing partners. While this process is ongoing, the complete
source-video archives are not publicly distributed. They are retained privately,
not deleted. We plan to release the complete videos once these matters are
resolved and the necessary permissions are confirmed; no release date has been set.

The original public folder URL remains valid for action annotations and the demo.
Previous direct links to full-video archives no longer provide public access.
Code and checkpoint availability are unchanged by this data-access update.

## Public Files

- [Full action-annotation archive](https://drive.google.com/file/d/1iky0xnf9_8v0tZp7BOoUkg63Ba0gqpNm/view?usp=drivesdk)
- [Selected-example demonstration](https://drive.google.com/file/d/1dVQ5uSbXgwdLh8ofHb4ulcBec8lELPZr/view?usp=drivesdk)
- [Watch the project-page demo](https://arlo0o.github.io/KVLR-project/#demo)

The demo is a presentation video, not a paired dataset subset or a replacement
for source videos. The public annotation archive and demo alone are not enough
to reproduce full training or benchmark evaluation. Video-dependent workflows
require an authorized copy of the corresponding source videos and reference frames.

Download and extract the public annotations:

```bash
# Run from <repo-root>, the directory containing README.md.
mkdir -p data/kasa/annotations
unzip /path/to/annotations.zip -d data/kasa/annotations
```

## Annotation Archive Inventory

The public archive was inspected on 2026-10-24. It contains 1,382
`memory_pool.pth` files in these directories:

| Directory | Per-instance annotation files |
| --- | ---: |
| `knotting_results/` | 86 |
| `needleGrasping_results/` | 676 |
| `needlePuncture_results/` | 620 |

These counts describe annotation files, not unique videos or frames. The archive
does not contain video/image files, captions, or split/task CSV files. References
to those additional inputs in training examples describe required workflow inputs,
not extra files bundled in this download. Keep the extracted `*_results/`
directories under `data/kasa/annotations/`; the action loader supports this
per-instance layout.

## Full-Video Layout (Authorized Copies or Future Release)

The layout and commands below are for authorized video copies or the planned
future release. They do not imply that the complete videos are public today.

```text
<repo-root>/data/kasa/
  annotations/
  videos/
    knotting/
    needleGrasping/
    needlePuncture/
```

Extraction when authorized full-video archives are available:

```bash
# Run from <repo-root>, the directory containing README.md.
mkdir -p data/kasa/annotations data/kasa/videos
unzip /path/to/annotations.zip -d data/kasa/annotations
unzip /path/to/knotting.zip -d data/kasa/videos
unzip /path/to/needleGrasping.zip -d data/kasa/videos
unzip /path/to/needlePuncture.zip -d data/kasa/videos
```

## Usage and Attribution

The repository's code license does not grant redistribution rights to underlying
third-party source videos. Please respect the applicable asset permissions and
attribution requirements. We will update this page with the full-video release
and its confirmed terms when the coordination is complete.
