"""
day4_inject.py
--------------
Injects Day 4 content at the END of the notebook (after Exercise 2):
  - Fairness and Responsible AI section
  - Model serialisation with joblib
  - predict_segment() production function
"""

import json

NB_PATH = r"notebooks\01_Clustering_Analysis.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


day4_cells = [

    md_cell(
        "---\n\n"
        "## Fairness and Responsible AI Considerations\n\n"
        "Insurance pricing based on behavioural clustering carries real ethical and\n"
        "regulatory risk. This section documents the key considerations that must\n"
        "be addressed before deploying this model in a production pricing environment.\n\n"
        "IBM's Responsible AI principles and the EU AI Act (2024) classify automated\n"
        "pricing decisions in insurance as high-risk AI applications. The following\n"
        "framework applies.\n\n"
        "### 1. Proxy Feature Risk\n\n"
        "While the five features in this dataset capture driving behaviour, they may\n"
        "act as proxies for demographic characteristics:\n\n"
        "- **Trip distance and time of day** may correlate with urban vs. rural residency,\n"
        "  which in turn correlates with socioeconomic status.\n"
        "- **Night driving frequency** may inadvertently reflect occupational patterns\n"
        "  (e.g., shift workers, delivery drivers) that skew along socioeconomic or\n"
        "  ethnic dimensions.\n\n"
        "**Recommendation:** Conduct a disparity impact analysis across protected\n"
        "characteristics (age, gender, postcode as a socioeconomic proxy) before\n"
        "any segment-based premium is applied.\n\n"
        "### 2. Pricing Discrimination Risk\n\n"
        "Cluster-based premium differentiation must comply with the UK Equality Act 2010\n"
        "and the FCA's Consumer Duty (2023), which prohibits pricing practices that\n"
        "systematically disadvantage vulnerable customer groups. Cluster membership\n"
        "alone is not a legally permissible basis for differential pricing without\n"
        "actuarial justification and documented audit trails.\n\n"
        "### 3. Model Drift and Re-evaluation\n\n"
        "Driver behaviour evolves: vehicle adoption rates, road infrastructure changes,\n"
        "and macroeconomic shifts all affect the feature distributions underlying the\n"
        "clusters. The model should be re-evaluated on a scheduled cadence (at minimum,\n"
        "quarterly) with cluster stability metrics tracked over time.\n\n"
        "### 4. Data Governance\n\n"
        "Telematics data — the likely source of these engineered features — is personal\n"
        "data under UK GDPR. The following governance requirements apply:\n\n"
        "- Explicit informed consent from the policyholder for telematics data collection.\n"
        "- Data minimisation: only the features necessary for the pricing model should\n"
        "  be retained.\n"
        "- Right to explanation: policyholders must be able to request a plain-language\n"
        "  explanation of how their cluster assignment affected their premium.\n"
        "- Retention limits: raw telematics logs should not be retained beyond the\n"
        "  period necessary to compute the engineered features."
    ),

    # Model serialisation
    md_cell(
        "---\n\n"
        "## Model Serialisation and Production Pipeline\n\n"
        "The final K-Means model is serialised to disk using `joblib`. This enables\n"
        "the model to score new drivers without retraining, supporting integration\n"
        "into a batch pricing pipeline or a real-time API."
    ),

    code_cell(
        "import joblib\n"
        "import os\n"
        "import sys\n\n"
        "# Ensure src is importable\n"
        "sys.path.insert(0, os.path.abspath('../src'))\n"
        "from models import predict_segment\n\n"
        "# Serialise the final fitted K-Means model\n"
        "model_output_path = cfg['model']['output_path']\n"
        "os.makedirs(os.path.dirname(model_output_path), exist_ok=True)\n\n"
        "joblib.dump(kmeans_final, model_output_path)\n"
        "print(f'Model saved to: {model_output_path}')\n\n"
        "# Verify the saved model loads and produces identical predictions\n"
        "loaded_model = joblib.load(model_output_path)\n"
        "verification_labels = loaded_model.predict(X)\n"
        "assert (verification_labels == final_labels).all(), 'Model serialisation verification failed.'\n"
        "print('Serialisation verified: loaded model produces identical predictions.')"
    ),

    code_cell(
        "# Demonstrate the production prediction function\n"
        "# predict_segment() is defined in src/models.py\n"
        "\n"
        "# Simulate 3 new drivers arriving for pricing\n"
        "sample_new_drivers = X.sample(3, random_state=cfg['clustering']['random_state']).copy()\n"
        "sample_new_drivers.index = ['New Driver A', 'New Driver B', 'New Driver C']\n\n"
        "predictions = predict_segment(sample_new_drivers, model_output_path)\n\n"
        "sample_new_drivers['predicted_cluster'] = predictions\n"
        "sample_new_drivers['segment_label'] = sample_new_drivers['predicted_cluster'].map(\n"
        "    {k: v['label'] for k, v in personas.items()}\n"
        ")\n"
        "sample_new_drivers['risk_level'] = sample_new_drivers['predicted_cluster'].map(\n"
        "    {k: v['risk_level'] for k, v in personas.items()}\n"
        ")\n"
        "sample_new_drivers['pricing_action'] = sample_new_drivers['predicted_cluster'].map(\n"
        "    {k: v['pricing_action'] for k, v in personas.items()}\n"
        ")\n\n"
        "print('New driver segment predictions:')\n"
        "sample_new_drivers"
    ),

]

# Append Day 4 cells at the very end
for cell in day4_cells:
    nb["cells"].append(cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Injected {len(day4_cells)} Day 4 cells at end of notebook.")
