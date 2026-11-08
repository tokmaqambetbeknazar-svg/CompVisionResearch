# Data sources and limitations

## Original dataset

**REHAB24-6: A multi-modal dataset of physical rehabilitation exercises**, v1 (2024).
Authors: Andrej Černek, Jan Sedmidubsky, Petra Budikova, Miriama Jánošová, Lukáš Katzer, Michal Procházka.

- Record and use conditions: https://zenodo.org/records/13305826
- DOI: https://doi.org/10.5281/zenodo.13305826
- Paper citation requested by authors: Černek, A., Sedmidubsky, J., Budikova, P. (2024). *REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods.* SISAP, Springer.

The dataset page permits academic/non-profit noncommercial research and requests citation of the paper. Do not describe these data as unrestricted commercial data or CC0: a derivative repository calls it CC0, but the original authors' explicit use conditions are the source used here. Our project is a noncommercial course experiment. These conditions also apply to the derived feature table.

The original dataset contains videos and ground-truth motion-capture skeletons. This experiment uses only the skeleton data, not video or textual prompts. Ex3 is **push-ups with hands on a table**, not necessarily floor push-ups. Ex6 is squats. The dataset correctness labels are per repetition and were collected under physiotherapist guidance.

## Coordinate-table derivative used for convenient access

Repository: https://github.com/YattaHab/REHAB24-6-Standardized-Training-Data

Pinned commit: `71b31a8853f9d9b5568d88484b4e3263ddc5da03`.

Files: `standardized_data/rehab246_pushup_positions_standardized.csv` and `standardized_data/rehab246_squat_positions_standardized.csv`.

The derivative maps the original 26-joint skeleton to 14 named columns, recentres coordinates on the pelvis, and cuts recordings into repetitions. It supplies joint coordinates, **not MediaPipe predictions from webcam footage**. The two hip columns both contain the same central pelvis joint. No claim is made that they are separate hip measurements. We compute 3D angles and normalized distances and aggregate over complete repetitions.

We independently downloaded the **original** `Segmentation.csv` from Zenodo. It is identical as a parsed table to the derivative's metadata. Every repetition is checked against the original exercise label, correctness label and frame count. The derivative field called `subject_id` is actually a **video ID**; actual participant identity comes from `person_id` in the original annotations. This prevents different recordings of the same person from leaking across evaluation splits.

`download_data.py` lists exact URLs and SHA-256 checksums. It downloads about 25 MB and refuses mismatched files. No upstream Python code is executed. The raw CSVs are not included in Git; the compact, derived `data/features.csv` is sufficient to reproduce the model comparison without downloading videos.

## Method references

- Random forest and parameter definitions: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Leave-one-group-out validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
- Context on fitness pose classification: https://chuoling.github.io/mediapipe/solutions/pose_classification.html

The feature extraction, training, parameter comparison, plots and course report in this folder were prepared for this project with AI assistance. The original dataset and derivative conversion belong to their respective authors.

## Scope

The experiment evaluates recognition of dataset labels from complete motion-capture repetitions. It does not train MediaPipe, YOLO or an LLM; it trains a new random-forest classifier. It does not prove real-time webcam accuracy, injury prevention, or clinical safety. Correctness combines different kinds of mistakes into one label. Only two held-out people are used for the final test; uncertainty and generalization to other people remain substantial.
