# CEE-Split

Research code, Unity implementation, experiments, and paper artifacts for CEE-Split.

## Repository layout

- `render-tests/` — evaluation, training, calibration, and experiment outputs.
- `unity-split-render/` — Unity split-rendering project sources and assets.
- `qoe-proxy-unity/` — Unity QoE calibration and data-collection project, with derived CSV datasets.
- `paper/` — manuscript sources, figures, results, scripts, and deliverables.

## Unity project

Open `unity-split-render/` as a Unity project. Generated directories such as `Library/`, `Logs/`, `obj/`, and local builds are intentionally excluded and will be recreated by Unity.

The QoE data-generation project uses the same policy for generated Unity directories. Its raw `DatasetOutputs` PNG frames are archived separately because of their size; the compact metadata and final training CSV files are included here.

## Notes

Large binary assets and document deliverables are stored with Git LFS.
