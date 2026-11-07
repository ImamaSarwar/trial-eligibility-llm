# Error analysis: ft_test

58 label errors / 206 records.

| category | count |
|---|---|
| missing-information confusion (insufficient info vs decision) | 49 |
| applicability confusion (not applicable vs decision) | 9 |

Correct label but different evidence IDs: 29

## Representative errors

### Error 1 - missing-information confusion (insufficient info vs decision)
- annotation 161 (patient sigir-201410), inclusion criterion
- criterion: Patients undergoing elective carotid endarterectomy
- expert: **not included**, evidence []: (none)
- model: **not enough information**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 2 - applicability confusion (not applicable vs decision)
- annotation 322 (patient sigir-201417), exclusion criterion
- criterion: Pregnant women or women of childbearing age who do not have adequate contraception.
- expert: **not excluded**, evidence [0]: [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
- model: **not applicable**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 3 - missing-information confusion (insufficient info vs decision)
- annotation 169 (patient sigir-201410), exclusion criterion
- criterion: Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- expert: **not enough information**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 4 - applicability confusion (not applicable vs decision)
- annotation 683 (patient sigir-20158), exclusion criterion
- criterion: Previous tonsillectomy, including partial tonsillectomy
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 5 - missing-information confusion (insufficient info vs decision)
- annotation 176 (patient sigir-201410), inclusion criterion
- criterion: No medical history of central vena catheterization.
- expert: **included**, evidence [0]: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- model: **not enough information**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 6 - applicability confusion (not applicable vs decision)
- annotation 693 (patient sigir-20158), exclusion criterion
- criterion: Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome)
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 7 - missing-information confusion (insufficient info vs decision)
- annotation 422 (patient sigir-201422), inclusion criterion
- criterion: If female, using birth control
- expert: **not applicable**, evidence [3]: [3] She has had no sexual partners and her menses are regular.
- model: **not enough information**, evidence [3]: [3] She has had no sexual partners and her menses are regular.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 8 - applicability confusion (not applicable vs decision)
- annotation 751 (patient sigir-201513), exclusion criterion
- criterion: breast-feeding
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 9 - missing-information confusion (insufficient info vs decision)
- annotation 762 (patient sigir-201513), inclusion criterion
- criterion: A sore and scratchy throat and/or pain on swallowing (odynophagia) together with at least 2 of the following clinical signs:
- expert: **not enough information**, evidence [0]: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
- model: **included**, evidence [0, 1, 2]: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes. [1] He is toxic-appearing, and leans forward while sitting on his mother's lap. [2] He is drooling and speaks with a muffled "hot potato" voice.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 10 - applicability confusion (not applicable vs decision)
- annotation 754 (patient sigir-201513), exclusion criterion
- criterion: alcoholism (drunk at least once a week)
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 11 - missing-information confusion (insufficient info vs decision)
- annotation 162 (patient sigir-201410), inclusion criterion
- criterion: Patients undergoing open abdominal aortic aneurysm repair
- expert: **not included**, evidence []: (none)
- model: **not enough information**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 12 - applicability confusion (not applicable vs decision)
- annotation 758 (patient sigir-201513), exclusion criterion
- criterion: army recruit
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)
