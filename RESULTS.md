# Retinal OCT analysis: results and preprocessing sensitivity

Completed 11 October 2026. Research feasibility study, not a clinically validated diagnostic model.

## Cohort and QC
Source: OCTDL v2, https://zenodo.org/records/10839556. Archive MD5 verified against the published value. Original target cohort: 1,563 scans, 531 patient identifiers. Exact decoded cross-patient duplicates affected 40 patients and 132 scans; these patients were excluded conservatively. Eight further repeated scans within retained patients were removed. Analysed cohort: 1423 scans, 491 patients (385 AMD; 106 normal).

## Patient-level five-fold out-of-fold results
| Model | AUROC | Balanced accuracy |
|---|---:|---:|
| 32x32 grayscale pixel features + logistic regression | 0.793 | 0.697 |
| Frozen ImageNet ResNet18, central crop + logistic regression | 0.991 | 0.950 |
| Frozen ImageNet ResNet18, full image with padding + logistic regression | 0.945 | 0.868 |
| Image size/brightness nuisance features + logistic regression | 0.729 | 0.702 |
| Prior-only classifier | — | 0.500 |

ResNet18 sensitivity: 0.966; specificity: 0.934 at fixed threshold 0.5. Approximate AUROC bootstrap interval: 0.983–0.996. Predictions are out-of-fold at patient level, with feature scaling fit separately on training folds. Hyperparameters were fixed in advance; no tuning was performed. The bootstrap conditions on fitted OOF predictions and omits retraining uncertainty.

## Interpretation
The pretrained baseline strongly discriminates the labels in this single dataset. This is not evidence of clinical generalisation. The nuisance model's AUROC of 0.729 suggests that acquisition or quality characteristics associate with diagnosis. Remaining nonidentical near-duplicates, acquisition confounding and uncertain patient metadata need investigation. Excluding duplicate-affected patients changes the sample and may introduce bias. Normal versus AMD is a simplified task and does not establish AMD subtype prediction, progression forecasting or treatment-response prediction.

ResNet18 is pretrained on natural images, not retinal OCT. RETFound has not yet been implemented. Averaging scans per patient ignores eye and visit structure. There is no external cohort or prospective test set. Global PCA is exploratory and not an input to the predictive validation.

## Next research steps
1. Review cross-patient duplicate groups, near-duplicates and identifier provenance.
2. Inspect false positives/negatives and central-crop effects; assess age, device and acquisition metadata where available.
3. Compare OCT-specific RETFound features with a protocol that prevents selecting models on the final evaluation cohort.
4. Explore image-space clustering with stability analysis and subtype metadata; do not equate clusters with clinical phenotypes without supporting evidence.
5. Start a separate longitudinal analysis only with verified repeated-visit metadata.

## Validation completed
Python compilation, two synthetic evaluation checks, full checksum-verified pixel and pretrained-image-feature executions, saved per-patient OOF predictions and environment versions. Results and PCA figures were inspected. No foundation model, longitudinal model or deployment claim is made.

## Preprocessing comparison reproduced in WSL
The user's WSL run reproduced the original baseline closely. The padding experiment used the same retained patients, five-fold assignments, frozen ImageNet encoder and logistic regression settings. It resized the full image to fit 224x224 while preserving aspect ratio and added black padding. The default crop resized the short edge to 256 and centre-cropped to 224x224. This is a composite preprocessing comparison, not a controlled isolation of cropping alone: field of view, effective anatomical resolution, resampling and padding also differ.

| Measure | Central crop | Full scan with padding |
|---|---:|---:|
| AUROC | 0.990591 | 0.944940 |
| Balanced accuracy | 0.950098 | 0.867520 |
| Sensitivity | 0.966234 | 0.914286 |
| Specificity | 0.933962 | 0.820755 |
| Misclassified patients | 20 | 52 |

Both methods correctly classified 431 patients and both missed 12. Padding corrected 8 original errors and introduced 40 new errors. Thus padding produced 32 additional errors overall. Review figures show that the crop removes lateral coverage, but the prominent elevation in the illustrated scan of patient a selected AMD case remains in the model input; coverage loss alone does not explain that case.

## Exploratory AMD stage analysis
Sensitivity at threshold 0.5, derived from out-of-fold predictions. These are subgroup summaries of AMD-versus-normal predictions, not a trained disease-stage classifier.

| Metadata stage | Patients | Crop detected / total | Crop sensitivity | Padding detected / total | Padding sensitivity |
|---|---:|---:|---:|---:|---:|
| Early | 113 | 103 / 113 | 91.2% | 84 / 113 | 74.3% |
| Intermediate | 68 | 67 / 68 | 98.5% | 66 / 68 | 97.1% |
| Late | 202 | 200 / 202 | 99.0% | 201 / 202 | 99.5% |
| Early; intermediate | 1 | 1 / 1 | 100% | 0 / 1 | 0% |
| Intermediate; late | 1 | 1 / 1 | 100% | 1 / 1 | 100% |

Ten of the 13 AMD patients missed by cropping were labelled early AMD. All seven AMD patients missed by both methods were labelled early AMD with drusen. The other five shared errors were labelled normal. No label corrections are justified from this analysis or the montages. Clinical review would be needed to assess the image-level label correspondence.

The poorer padding result is concentrated in early disease. Reduced anatomical resolution when fitting a wide image inside a square is one plausible explanation, not an established cause. The nuisance-only AUROC of 0.729 indicates label-associated acquisition/quality characteristics; it does not quantify how much ResNet18 relies on those characteristics.

## Metadata and evaluation limits clarified
- The two mixed-stage patients have scan metadata with more than one stage. These labels do not establish longitudinal progression because reliable visit chronology has not been analysed.
- At least one shared-error patient (a selected normal case) has both OD and OS scans. Current patient-mean feature aggregation combines eyes. The code 0 in the eye field has not been interpreted as a laterality value.
- The preprocessing alternative and stage analysis were introduced after reviewing baseline errors. They are exploratory; selecting the better method and reporting it as independently confirmed would be optimistic. All existing patients have contributed to evaluation; splitting this already-inspected cohort now would not create a genuinely untouched test set.
- Bootstrap intervals condition on OOF predictions, omit refitting uncertainty and do not establish external validity.
- No RETFound, temporal modelling, survival analysis, formal clinical phenotype discovery or external validation has been completed.

## Package verification
The uploaded metrics were recomputed from all saved patient predictions. The crop/padding patient identifiers, labels and fold assignments agree. Two existing synthetic evaluation checks passed. The uploaded archive lacks source images, so image-level duplicate verification and feature extraction were not rerun during this review. The baseline and padding extractions were previously executed by the user in WSL.

## Suggested portfolio description
Reproducible retinal OCT analysis using frozen pretrained image features, patient-level cross-validation, duplicate auditing, preprocessing sensitivity analysis and exploratory assessment by AMD stage. Developed with AI assistance and reproduced in WSL; methods should be understood before presentation in an interview.
