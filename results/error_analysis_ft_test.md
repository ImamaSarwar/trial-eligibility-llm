# Error analysis: ft_test

58 label errors / 206 records.

| category | count |
|---|---|
| missing-information confusion (insufficient info vs decision) | 49 |
| applicability confusion (not applicable vs decision) | 9 |

Correct label but different evidence IDs: 29


## Summary of failure modes (test split, n=206, 58 label errors)

The printed expert-to-model pair counts were: not enough information -> not excluded 28; included -> not enough information 10;
not applicable -> not excluded 8; not included -> not enough information 7; not applicable -> not enough information 3;
not excluded -> not applicable 1 (these six account for 57 of the 58 errors).

1. **Missing information treated as negative (exclusion criteria), 28 errors.** Most serious: the model never predicted 'not enough information' for an exclusion criterion (recall 0 of 28). Probable cause: label prior (75% of exclusion training labels are 'not excluded'). Errors 3.
2. **Over-abstention on inclusion criteria, 17 errors** (10 included and 7 not included predicted as 'not enough information'). The model is cautious when the evidence is indirect or requires inferring a negative (Errors 1, 5, 11).
3. **Applicability boundary, about 12 errors** (not applicable vs not excluded / not enough information). Partly label ambiguity (Errors 2, 4, 6); partly the model ignoring the patient's age or sex (Errors 8, 10, 12). All five sampled not-applicable -> not-excluded errors (4, 6, 8, 10, 12) come from the two paediatric patients (a 10-year-old and a 5-year-old). Low practical impact because both labels are non-blocking in the aggregation rule.
4. **Over-confident positives are rare** (Error 9), linked to an incomplete criterion text.
5. **29 further records have the correct label but different evidence IDs**, so label accuracy overstates evidence quality.

Impact differs by error type: category 1 can silently make a patient look eligible (the app adds a caution when a 'not excluded' has no cited evidence); category 3 does not change the aggregated outcome.

## Representative errors

### Error 1 - missing-information confusion (insufficient info vs decision)
- annotation 161 (patient sigir-201410), inclusion criterion
- criterion: Patients undergoing elective carotid endarterectomy
- expert: **not included**, evidence []: (none)
- model: **not enough information**, evidence []: (none)
- analysis: **Over-abstention on 'not included'.** Patient sigir-201410 is described as having cardiac catheterisation (see Error 5), not a carotid operation, so the criterion is clearly unmet. The expert concluded 'not included' without citing a sentence. The model abstained because no sentence literally says the patient is not having this surgery. Reaching 'not included' means inferring a negative from a different clinical context. The class is rare in training (13 of 603 annotations) and the model never predicted it on the test set (0 of 7).

### Error 2 - applicability confusion (not applicable vs decision)
- annotation 322 (patient sigir-201417), exclusion criterion
- criterion: Pregnant women or women of childbearing age who do not have adequate contraception.
- expert: **not excluded**, evidence [0]: [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
- model: **not applicable**, evidence []: (none)
- analysis: **Label-boundary ambiguity (not applicable vs not excluded); the model is arguably reasonable.** The patient is a 48-year-old male (sentence 0), so a pregnancy/contraception criterion cannot apply. The expert still chose 'not excluded' and cited sentence 0; the model chose 'not applicable' and cited nothing. In Errors 8, 10 and 12 the roles are reversed (expert 'not applicable', model 'not excluded'), so the convention for 'cannot apply to this patient' is not consistent. Part of this error group reflects annotation ambiguity, not pure model failure.

### Error 3 - missing-information confusion (insufficient info vs decision)
- annotation 169 (patient sigir-201410), exclusion criterion
- criterion: Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- expert: **not enough information**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- analysis: **Missing information treated as negative (the most important failure).** A pre-operative GFR value is a laboratory result; the expert says the note does not provide it (not enough information). The model answered 'not excluded' and cited nothing, i.e. it treated the absence of a mention as proof that the exclusion does not apply. This is the largest error group (28 of 58 test errors) and has a plausible cause: 292 of 388 exclusion annotations in training are 'not excluded' (75%), and the model never predicted 'not enough information' for an exclusion criterion (0 of 28 on test). In the app this is dangerous because it can make a patient look eligible.

### Error 4 - applicability confusion (not applicable vs decision)
- annotation 683 (patient sigir-20158), exclusion criterion
- criterion: Previous tonsillectomy, including partial tonsillectomy
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- analysis: **Label-convention / applicability error (not applicable vs not excluded).** Patient sigir-20158 is a 10-year-old boy with snoring, breathing pauses and daytime sleepiness; the note never mentions his tonsils. The expert marked 'previous tonsillectomy' as 'not applicable' rather than 'not enough information', so the expert label cannot be read directly from the note. The model took the majority exclusion label 'not excluded' and cited nothing. This looks more like ambiguity in the annotation convention than a clear reasoning failure.

### Error 5 - missing-information confusion (insufficient info vs decision)
- annotation 176 (patient sigir-201410), inclusion criterion
- criterion: No medical history of central vena catheterization.
- expert: **included**, evidence [0]: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- model: **not enough information**, evidence []: (none)
- analysis: **Over-abstention on a negated inclusion criterion with indirect evidence.** The criterion is 'No medical history of central vena catheterization', so 'included' means the patient satisfies it by NOT having had one. The expert cited sentence 0: the only catheterisation mentioned is cardiac catheterisation via the femoral ARTERY, which is not a central venous line. Recognising that needs domain knowledge (arterial vs central venous) plus the negation. The model found no direct statement and abstained.

### Error 6 - applicability confusion (not applicable vs decision)
- annotation 693 (patient sigir-20158), exclusion criterion
- criterion: Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome)
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- analysis: **Label-convention ambiguity (not applicable vs not excluded).** Same patient (10-year-old boy). Sentence 1 says 'No history of headache or night terrors'; night terrors are a sleep disorder, so 'not excluded' citing sentence 1 would also be a defensible answer. The expert chose 'not applicable' with no evidence and the model 'not excluded' with no evidence. The two answers are close in meaning, and the difference mainly reflects an inconsistent gold convention.

### Error 7 - missing-information confusion (insufficient info vs decision)
- annotation 422 (patient sigir-201422), inclusion criterion
- criterion: If female, using birth control
- expert: **not applicable**, evidence [3]: [3] She has had no sexual partners and her menses are regular.
- model: **not enough information**, evidence [3]: [3] She has had no sexual partners and her menses are regular.
- analysis: **Correct evidence, wrong label (boundary between not applicable and not enough information).** The model cited exactly the expert's sentence 3 ('no sexual partners and her menses are regular'). That sentence implies birth control is not relevant, so the expert chose 'not applicable'; the model read it as insufficient information. Evidence localisation worked, the final judgement did not; 'not applicable' has only 15 test examples and recall 0.27.

### Error 8 - applicability confusion (not applicable vs decision)
- annotation 751 (patient sigir-201513), exclusion criterion
- criterion: breast-feeding
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- analysis: **Applicability not recognised.** Patient sigir-201513 is a 5-year-old boy (sentence 0, see Error 9), so 'breast-feeding' cannot apply. The information was in the note but the model did not use it; it defaulted to 'not excluded' with no citation. Consequence for the app is small: both 'not applicable' and 'not excluded' are non-blocking in the aggregation rule.

### Error 9 - missing-information confusion (insufficient info vs decision)
- annotation 762 (patient sigir-201513), inclusion criterion
- criterion: A sore and scratchy throat and/or pain on swallowing (odynophagia) together with at least 2 of the following clinical signs:
- expert: **not enough information**, evidence [0]: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
- model: **included**, evidence [0, 1, 2]: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes. [1] He is toxic-appearing, and leans forward while sitting on his mother's lap. [2] He is drooling and speaks with a muffled "hot potato" voice.
- analysis: **Over-confident positive (the model is more decisive than the expert).** The criterion text is cut off: it ends with a colon after 'together with at least 2 of the following clinical signs:' and the list of signs is missing, so it cannot be fully verified. The expert answered 'not enough information' citing sentence 0 only. The model answered 'included' and cited sentences 0-2. It treated dysphagia, drooling and a muffled voice as equivalent to a sore throat or pain on swallowing plus two signs. This is a false positive caused by semantic similarity and an incomplete criterion.

### Error 10 - applicability confusion (not applicable vs decision)
- annotation 754 (patient sigir-201513), exclusion criterion
- criterion: alcoholism (drunk at least once a week)
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- analysis: **Applicability not recognised.** Same patient as Error 8 (a 5-year-old boy). 'Alcoholism (drunk at least once a week)' cannot apply to a child. The model chose 'not excluded' without citing the age. As with Error 8 the label is wrong but the practical outcome (not blocking) is the same.

### Error 11 - missing-information confusion (insufficient info vs decision)
- annotation 162 (patient sigir-201410), inclusion criterion
- criterion: Patients undergoing open abdominal aortic aneurysm repair
- expert: **not included**, evidence []: (none)
- model: **not enough information**, evidence []: (none)
- analysis: **Over-abstention on 'not included'.** Same patient and reasoning as Error 1: the patient had a cardiac catheterisation, not an open aortic aneurysm repair, so the criterion is unmet. The expert gave 'not included' with no cited sentence; the model abstained. Two of the twelve sampled errors (1 and 11) are this identical failure.

### Error 12 - applicability confusion (not applicable vs decision)
- annotation 758 (patient sigir-201513), exclusion criterion
- criterion: army recruit
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- analysis: **Applicability not recognised.** Same patient as Errors 8 and 10 (a 5-year-old boy): 'army recruit' cannot apply. The model returned the majority label 'not excluded' with no evidence. Together Errors 8, 10 and 12 show the model not using the age statement in sentence 0 to decide applicability.
