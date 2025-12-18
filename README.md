# NanoHELDI-Pain
Code for SweatPain-Decoding

NanoHELDI-Pain/
│
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── preprocessing/
│   ├── maldiquant_pipeline.R
│   ├── peak_alignment.py
│   └── normalization.py
│
├── analysis/
│   ├── statistics/
│   │   ├── volcano_plots.py
│   │   ├── time_course_analysis.py
│   │   └── similarity_analysis.py
│   │
│   ├── ml/
│   │   ├── classification/
│   │   │   ├── train_mlp.py
│   │   │   ├── baseline_models.py
│   │   │   └── evaluation.py
│   │   │
│   │   ├── regression/
│   │   │   ├── pain_score_regression.py
│   │   │   └── metrics.py
│   │   │
│   │   └── interpretability/
│   │       ├── shap_analysis.py
│   │       └── integrated_gradients.py
│
├── msms/
│   ├── metfrag_annotation/
│   ├── standard_matching/
│   └── cosine_similarity.py
│
├── pathway/
│   ├── metabolite_mapping.py
│   └── metaboanalyst_inputs/
│
├── figures/
│   ├── fig2_nanostructure/
│   ├── fig3_metabolic_fingerprints/
│   ├── fig4_ml_and_pathway/
│   └── fig5_menstrual_pain/
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── menstrual_cohort_analysis.ipynb
│
└── scripts/
    ├── run_full_pipeline.sh
    └── reproduce_figures.sh
