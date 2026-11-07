# Error analysis: ft_val

54 label errors / 206 records.

| category | count |
|---|---|
| missing-information confusion (insufficient info vs decision) | 44 |
| applicability confusion (not applicable vs decision) | 8 |
| polarity flip (direct contradiction of expert) | 2 |

Correct label but different evidence IDs: 42

## Representative errors

### Error 1 - missing-information confusion (insufficient info vs decision)
- annotation 8 (patient sigir-20141), exclusion criterion
- criterion: Unstable angina
- expert: **not enough information**, evidence [0, 1, 2]: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. [2] The latest episode of pain ended half an hour prior to her arrival.
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 2 - applicability confusion (not applicable vs decision)
- annotation 24 (patient sigir-20141), exclusion criterion
- criterion: Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 3 - polarity flip (direct contradiction of expert)
- annotation 263 (patient sigir-201415), exclusion criterion
- criterion: 13. Evidence of uterine pathology other than leiomyoma
- expert: **excluded**, evidence [5]: [5] The differential diagnosis includes vesicular mole vs fibroid degeneration.
- model: **not excluded**, evidence [4]: [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 4 - missing-information confusion (insufficient info vs decision)
- annotation 15 (patient sigir-20141), exclusion criterion
- criterion: Pregnant and nursing women
- expert: **not enough information**, evidence []: (none)
- model: **not applicable**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 5 - applicability confusion (not applicable vs decision)
- annotation 272 (patient sigir-201415), exclusion criterion
- criterion: 22. Women who are breast feeding.
- expert: **not excluded**, evidence []: (none)
- model: **not applicable**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 6 - polarity flip (direct contradiction of expert)
- annotation 935 (patient sigir-201522), inclusion criterion
- criterion: Thought likely to be Mycobacterium culture positive (including NTM infected for the NIH CC) by enrolling physician
- expert: **not included**, evidence [3]: [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
- model: **included**, evidence [3]: [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 7 - missing-information confusion (insufficient info vs decision)
- annotation 21 (patient sigir-20141), inclusion criterion
- criterion: Clinical diagnosis of hypertension
- expert: **included**, evidence [3]: [3] She is known to have hypertension and obesity.
- model: **not enough information**, evidence [3]: [3] She is known to have hypertension and obesity.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 8 - applicability confusion (not applicable vs decision)
- annotation 558 (patient sigir-201430), exclusion criterion
- criterion: Inability to provide informed consent.
- expert: **not applicable**, evidence [8]: [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- model: **not excluded**, evidence [8]: [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 9 - missing-information confusion (insufficient info vs decision)
- annotation 31 (patient sigir-20142), inclusion criterion
- criterion: patients have fever more than one week
- expert: **not included**, evidence [0]: [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
- model: **not enough information**, evidence [0]: [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 10 - applicability confusion (not applicable vs decision)
- annotation 801 (patient sigir-201515), exclusion criterion
- criterion: Definite small-vessel etiology by history or imaging
- expert: **not applicable**, evidence []: (none)
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 11 - missing-information confusion (insufficient info vs decision)
- annotation 33 (patient sigir-20142), inclusion criterion
- criterion: full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis
- expert: **not applicable**, evidence []: (none)
- model: **not enough information**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)

### Error 12 - applicability confusion (not applicable vs decision)
- annotation 817 (patient sigir-201515), exclusion criterion
- criterion: Subjects who have a non-vascular origin of the neurological symptoms after brain imaging (CT scan or MRI)
- expert: **not applicable**, evidence [0]: [0] Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.
- model: **not excluded**, evidence []: (none)
- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)
