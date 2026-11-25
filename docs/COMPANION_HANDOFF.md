# Before Tuesday's presentation

The photo-based MVP is implemented and the notebook has been run successfully. Please focus only on presentation readiness, not final-project extensions.

1. **Run the tests** from the repository root:
   ```powershell
   $env:PYTHONPATH = 'src'
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```
2. **Prepare the demo laptop.** Run `notebooks/Decorador_AR.ipynb` once with the project environment. Ensure the ADE20K ZIP, ZInD sample, and model weights are available before the talk; do not rely on downloads during the live demo.
3. **Test the editor manually:** preview, add/remove, scale/rotation, lighting, one 3D model, kitchen support detection, and PNG export. Open the exported PNG to confirm it is valid.
4. **Make and rehearse a six-slide deck** using `docs/GUIA_EXPO.md` and the generated PNG/CSV evidence. Copy the deck and selected backups from `results/` to the team's shared folder; `results/` is not tracked by Git.

Keep the presentation claims within this scope:

- This is a photo editor, not live camera-motion AR. Explain the change from the original proposal.
- ZInD supplies annotated room geometry; placement is not fully automatic on arbitrary photos.
- Furniture is procedural low-poly. Depth Anything proposes support regions; metric depth accuracy has not been evaluated.
- The RANSAC evidence is for views with known rotation. The own-photo translation trial has no independent ground truth.
- Do not claim real-object depth occlusion, validated OBJ imports, or certified BT coverage. Confirm the assignment requirements with the official brief/instructor.

Do not spend presentation time on SIFT, larger segmentation models, OBJ assets, or video tracking; those are outside this expo checklist. The private photos in `data/own/` are intentionally ignored by Git and must not be shared without the owner's approval.
