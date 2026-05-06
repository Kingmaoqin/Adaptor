from __future__ import annotations

from typing import List

from .schemas import FeatureMetadata


ROLE_FEATURE_LIBRARY = {
    "confounding": [
        ("age_at_icu_admission", "Patient age captured before intervention assignment."),
        ("baseline_sofa_score", "Baseline organ failure burden before intervention assignment."),
        ("pre_index_mean_arterial_pressure", "Mean arterial pressure measured during the pre-index window."),
        ("baseline_serum_lactate", "Lactate level indicating shock severity before intervention assignment."),
        ("baseline_creatinine", "Kidney function marker available before intervention assignment."),
        ("charlson_comorbidity_index", "Comorbidity burden summarizing pre-existing disease severity."),
        ("icu_admission_apache_component", "Acute physiology component measured before intervention assignment."),
        ("pre_index_heart_rate", "Heart rate trend observed before intervention assignment."),
        ("baseline_white_blood_cell_count", "Inflammatory burden marker measured before intervention assignment."),
        ("pre_index_respiratory_rate", "Respiratory distress measure observed before intervention assignment."),
    ],
    "intervention": [
        ("pressor_assignment_propensity_score", "Latent signal associated with whether the intervention will be assigned."),
        ("clinician_escalation_signal", "Pre-index severity cue that influences intervention assignment logic."),
        ("suspected_refractory_hypotension_marker", "Pre-index indicator that intervention escalation is likely."),
        ("vasopressor_readiness_flag", "Operational readiness feature related to intervention initiation."),
        ("pre_index_fluid_nonresponse_indicator", "Feature suggesting fluids may be insufficient before intervention."),
        ("shock_team_activation_signal", "Workflow signal correlated with intervention assignment."),
    ],
    "outcome": [
        ("baseline_mortality_risk_marker", "Pre-index pattern associated with subsequent mortality risk."),
        ("end_organ_hypoperfusion_index", "Severity marker associated with downstream adverse outcomes."),
        ("baseline_oxygenation_impairment", "Pre-index respiratory compromise associated with outcome risk."),
        ("multi_organ_dysfunction_pattern", "Composite pre-index marker of adverse outcome risk."),
        ("baseline_coagulation_instability", "Coagulation abnormality associated with downstream outcome risk."),
        ("pre_index_metabolic_acidosis_pattern", "Metabolic instability observed before intervention assignment."),
    ],
    "proxy": [
        ("latent_acuity_surrogate", "Observed proxy correlated with unmeasured latent acuity."),
        ("nursing_intensity_proxy", "Operational intensity feature indirectly reflecting latent acuity."),
        ("pre_index_monitoring_density", "Monitoring frequency proxy for clinician concern and latent acuity."),
        ("support_device_burden_proxy", "Proxy marker for hidden severity not fully captured by confounders."),
        ("pre_index_laboratory_density_proxy", "Lab ordering density as a proxy for latent acuity."),
        ("early_resuscitation_intensity_proxy", "Observed proxy correlated with hidden physiological instability."),
    ],
    "post_intervention": [
        ("post_index_mean_arterial_pressure_response", "Hemodynamic response measured after intervention assignment."),
        ("post_index_lactate_clearance", "Post-index biomarker response occurring after intervention assignment."),
        ("post_index_urine_output_response", "Kidney output response measured after intervention assignment."),
        ("post_index_pressor_dose_adjustment", "Dose titration signal available only after intervention assignment."),
        ("post_index_perfusion_recovery_marker", "Perfusion recovery feature generated after intervention assignment."),
        ("post_index_hemodynamic_stabilization_flag", "Stability signal recorded after intervention assignment."),
    ],
}


def build_synthetic_feature_metadata(role_labels: List[str]) -> List[FeatureMetadata]:
    metadata: List[FeatureMetadata] = []
    role_to_keyword = {
        "confounding": 0.0,
        "intervention": 0.0,
        "outcome": 0.0,
        "proxy": 0.0,
        "post_intervention": 1.0,
    }
    role_to_time = {
        "confounding": -12.0,
        "intervention": -6.0,
        "outcome": -4.0,
        "proxy": -8.0,
        "post_intervention": 4.0,
    }
    role_to_window = {
        "confounding": "baseline",
        "intervention": "ambiguous",
        "outcome": "baseline",
        "proxy": "ambiguous",
        "post_intervention": "post_index",
    }
    role_to_missingness = {
        "confounding": 0.05,
        "intervention": 0.10,
        "outcome": 0.10,
        "proxy": 0.15,
        "post_intervention": 0.85,
    }
    for feature_index, role_name in enumerate(role_labels):
        feature_templates = ROLE_FEATURE_LIBRARY[role_name]
        template_name, template_description = feature_templates[feature_index % len(feature_templates)]
        feature_name = f"{template_name}_{feature_index:03d}"
        metadata.append(
            FeatureMetadata(
                feature_index=feature_index,
                feature_name=feature_name,
                description=template_description,
                oracle_role=role_name,
                relative_time=role_to_time[role_name],
                measurement_window=role_to_window[role_name],
                post_intervention_keyword=role_to_keyword[role_name],
                missingness_pre_index=role_to_missingness[role_name],
                always_missing_pre_index=1.0 if role_name == "post_intervention" else 0.0,
            )
        )
    return metadata
