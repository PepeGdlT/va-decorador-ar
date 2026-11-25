# Quick update for Tuesday's expo

Hey everyone! The photo-based MVP is in place and the main notebook now runs from start to finish. It can segment walls and floors, place pictures and rugs in perspective, and add procedural 3D furniture (sofa, chair, plant) with lighting and shadows. It also suggests places for objects on real supports using estimated depth, and includes an ORB/RANSAC experiment for views with known camera rotation.

All 16 unit tests pass, and the notebook completed without errors. It generated backup images and a RANSAC CSV under `results/`; that folder is ignored by Git, so the files still need to be copied to the group share.

## What remains before the presentation

- Try the editor on the presentation laptop: preview, add/remove, scale/rotation, lighting, support detection, and PNG export.
- Make a short six-slide deck using `docs/GUIA_EXPO.md`, include the saved results, and rehearse the demo.
- Cache the datasets and model weights beforehand so nothing has to download during the presentation.
- Explain that the current demo is photo-based; it is not live AR with a moving camera.

## A few things to be clear about

The room geometry comes from ZInD annotations, so placement is not fully automatic on any arbitrary photo. Depth Anything only proposes support regions; we have not measured metric depth accuracy against RGB-D. The translation test has no independent ground truth, real-object depth occlusion is not part of the demo, and OBJ loading has not been validated. The BT coverage percentage is still provisional until checked against the official assignment and the instructor.

For Tuesday, let's focus on testing the demo and preparing the presentation rather than adding final-project features such as video tracking, SIFT, bigger models, or third-party assets.
