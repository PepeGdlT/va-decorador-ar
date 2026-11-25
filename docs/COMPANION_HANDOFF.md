# Expo status and remaining work

## What is implemented

- A photo-based interior decorator: SegFormer wall/floor masks, ZInD-referenced room planes, perspective placement of pictures/rugs, and an interactive editor.
- Procedural low-poly furniture (including sofa, chair, and plant), with 3D projection, local lighting, and contact shadows.
- Monocular depth-based proposals for placing accessories on detected supports.
- ORB/RANSAC registration on two views with known camera rotation, plus a threshold-sensitivity CSV.
- A reproducible notebook run with PNG/CSV backups. All 16 unit tests pass; all 10 executable notebook cells completed without errors.

## Still to do before Tuesday

1. On the presentation computer, manually verify editor preview, add/remove, scale/rotation, lighting, support detection, and PNG export.
2. Create and rehearse a concise six-slide deck using `docs/GUIA_EXPO.md`.
3. Copy the generated notebook, selected PNGs, and CSV from `results/` to the group share; `results/` is Git-ignored. Make sure required datasets and model weights are cached before the demo.
4. Explain the scope change: the current demonstrable MVP is photo-based, not live camera-motion AR.

## Claims and limitations

- ZInD provides annotated geometry; placement is not fully automatic on arbitrary photos.
- Depth Anything proposes support regions; metric accuracy against RGB-D has not been evaluated.
- The own-photo translation trial has no independent ground truth. The rotation experiment does not validate translation.
- Real-scene depth occlusion and imported OBJ assets are not integrated into the demo.
- The proposed BT coverage is provisional; confirm it against the official assignment and instructor.

Do not spend expo preparation time on final-project extensions such as video tracking, SIFT, larger models, or third-party assets.
