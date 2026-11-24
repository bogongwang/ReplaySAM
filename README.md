# ReplaySAM

Interactive particle segmentation by replaying 3D volumes as video across orthogonal views.

**Paper:** [ReplaySAM: Zero-Shot 3D Particle-Pack Segmentation with SAM 2](https://arxiv.org/abs/2611.30016) (arXiv preprint, 2026).

> **Early preview:** this repository is an early research preview. APIs, configuration defaults, installation steps, and example data layout may change as the project develops.

ReplaySAM treats a tomogram as three orthogonal video sequences: XY, XZ, and YZ. It generates point prompts automatically from distance-transform peaks, runs SAM 2 on view-wise slices, and fuses the resulting masks with majority voting to produce a 3D particle instance mask. The goal is zero-shot 3D particle instance segmentation without task-specific model training, while still supporting interactive annotation workflows.

## Features

- Automatic prompt generation from local maxima in a distance transform.
- Triton blockwise-argmax prompt generation, with a CPU reference fallback.
- SAM 2 inference replayed across orthogonal tomographic views.
- Multi-axis mask fusion through majority voting.
- Multiple positive and negative 3D prompts for guided inference.
- Optional zoomed, multi-scale seed and propagation inference.
- Axis-pair consistency filtering for rejecting contradictory views.
- NumPy, PyTorch tensor, and Dask model data loaders.
- Zarr, NumPy, TIFF, NIfTI, and NetCDF volume loading utilities.
- Zarr output writer for instance segmentation volumes.
- Demo notebook for the example iron ore tomogram.

## Repository Layout

```text
replaysam/
  pipeline.py              # End-to-end SAM2 particle segmentation pipeline
  models/
    adapters.py            # Multi-axis SAM2 adapter and mask fusion
    sam2/                  # Vendored SAM 2 code
    sam2_backbone.py       # ReplaySAM-specific SAM2 video backbone wrapper
  utils/
    configs.py             # Pipeline, backbone, and prompt configs
    io.py                  # Volume readers and Zarr writer
    prompt_generator.py    # Distance-transform prompt generation
notebooks/
  example.ipynb            # Demo notebook
```

## Setup

ReplaySAM is developed for Python 3.12 and a CUDA-capable PyTorch environment. A GPU is strongly recommended for full SAM 2 inference. Some utility and notebook cells can run on CPU, but the complete pipeline expects SAM 2 checkpoints and a compatible GPU.

Clone the repository and install the project dependencies with `uv`:

```bash
git clone https://github.com/bogongwang/ReplaySAM.git
cd ReplaySAM
uv sync
```

Install the vendored SAM 2 package so `import sam2` works:

```bash
uv pip install -e replaysam/models/sam2
```

If the SAM 2 CUDA extension causes build problems, install without that extension:

```bash
SAM2_BUILD_CUDA=0 uv pip install -e replaysam/models/sam2
```

Install the notebook kernel:

```bash
uv run python -m ipykernel install --user --name replaysam --display-name "ReplaySAM"
```

## SAM 2 Checkpoints

ReplaySAM expects SAM 2.1 checkpoints under:

```text
replaysam/models/sam2/checkpoints/
```

For the default tiny model used in the example notebook:

```bash
mkdir -p replaysam/models/sam2/checkpoints
curl -L \
  -o replaysam/models/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

Other supported model sizes are `small`, `base_plus`, and `large`; their checkpoint filenames are `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt`, and `sam2.1_hiera_large.pt`.

## Example Data

Example data is hosted on Hugging Face:

<https://huggingface.co/datasets/bogongwang/synthetic-particle-pack>

Download it with Git LFS and arrange it into the layout used by the demo:

```bash
git lfs install
git clone https://huggingface.co/datasets/bogongwang/synthetic-particle-pack /tmp/synthetic-particle-pack

mkdir -p data/iron_ore_GOH6_500_1/sim_res_0
cp -a /tmp/synthetic-particle-pack/iron_ore_GOH6_500_1/*.zarr \
  data/iron_ore_GOH6_500_1/sim_res_0/
```

The current demo notebook expects this input:

```text
data/iron_ore_GOH6_500_1/sim_res_0/tomo_noisy.zarr
```

The Hugging Face dataset contains the iron ore volume pack, including `tomo.zarr`, `tomo_noisy.zarr`, and `mask.zarr`.

## Demo Notebook

Open:

```text
notebooks/example.ipynb
```

The notebook loads the iron ore `tomo_noisy.zarr`, previews a slice, generates candidate point prompts, and builds a `SAM2PipelineConfig`. The full pipeline cell is disabled by default:

```python
RUN_FULL_PIPELINE = False
```

Set it to `True` after the SAM 2 package, checkpoints, CUDA/PyTorch environment, and example data are ready.

## Minimal Pipeline Example

```python
from pathlib import Path

from replaysam.pipeline import Pipeline
from replaysam.utils.configs import (
    PromptGeneratorConfig,
    SAM2BackboneConfig,
    SAM2PipelineConfig,
)

config = SAM2PipelineConfig(
    volume_path=Path("data/iron_ore_GOH6_500_1/sim_res_0/tomo_noisy.zarr"),
    output_parent_dir=Path("outputs"),
    backbone_config=SAM2BackboneConfig(
        model_size="tiny",
        compute_device="cuda:0",
        storage_device="cpu",
        compile=False,
    ),
    prompt_generator_config=PromptGeneratorConfig(
        method="peak_local_max",  # or "blockwise_argmax" on CUDA
        binarisation_threshold=None,
        dist_val_thresh=5.0,
        max_filter_size=7,
        crop_size=(256, 256, 256),
        crop_overlap=(32, 32, 32),
    ),
    max_prompts=10,
    inference_axes=(0, 1, 2),
    majority_voting_threshold=2,
)

pipeline = Pipeline(config)
pipeline.run()
print(pipeline.output_path)
```

For guided segmentation, provide any number of positive points and optional
negative points in `(z, y, x)` order:

```python
pipeline.run_with_prompt_points(
    positive_point_prompts=[(42, 120, 180), (43, 122, 181)],
    negative_point_prompts=[(42, 90, 160)],
)
```

The model adapter also accepts a `dask.array.Array`, `torch.Tensor`, or
`numpy.ndarray` directly. Set `volume_loader="dask"` in
`SAM2PipelineConfig` to keep a file-backed volume lazy; set
`backbone_config.zoom_enabled=True` to enable the configured multi-scale zoom
path, and adjust `consistency_score` to control axis-agreement filtering.

Outputs are written to a timestamped directory under `output_parent_dir`, with the segmentation stored as `segmentation.zarr`.

## Notes

- The default prompt generator may use CuPy when available and falls back to CPU if CuPy/CUDA is unavailable.
- Full-volume prompt generation can be expensive for large tomograms; tune `crop_size`, `crop_overlap`, thresholds, and `max_prompts` for quick experiments.
- SAM 2 checkpoint loading is controlled by `SAM2BackboneConfig.model_size`; short aliases `t`, `s`, `b+`, and `l` map to `tiny`, `small`, `base_plus`, and `large`.

## Citation

If you use ReplaySAM in your research, please cite:

```bibtex
@misc{wang2026replaysam,
  title         = {{ReplaySAM}: Zero-Shot {3D} Particle-Pack Segmentation with {SAM 2}},
  author        = {Wang, Bogong and Zhang, Yulai and Lösel, Philipp D. and Kingston, Andrew and Knackstedt, Mark},
  year          = {2026},
  eprint        = {2611.30016},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.30016}
}
```
