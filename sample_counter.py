import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sample Counter", layout="wide")


# -----------------------------
# Helper functions
# -----------------------------

def default_concentration_names(n: int) -> list[str]:
    if n == 1:
        return ["High"]
    if n == 2:
        return ["Low", "High"]
    return [f"C{i+1}" for i in range(n)]


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def build_experiment_two_stage(
    experiment_name: str,
    toxins: list[str],
    concentrations_tissue: list[str],
    concentrations_fruit: list[str],
    # plants
    tissue_plants_per_toxin_conc: int,
    fruit_plants_per_toxin: int,
    neg_ctrl_tissue_plants_total: int,
    neg_ctrl_fruit_plants_total: int,
    # sampling rules (per plant)
    samples_per_tissue_plant: int,  # root + leaf + end-solution (default 3)
    samples_per_fruit_plant: int,   # fruit + end-solution (default 2)
    # initial exposure QC samples
    qc_initial_per_toxin_per_conc_tissue: int,  # default 3
    qc_initial_per_toxin_per_conc_fruit: int,   # default 3
    # positive control vessels sampled at END
    pos_ctrl_vessels_per_toxin_per_conc_tissue: int,  # default 3
    pos_ctrl_vessels_per_toxin_per_conc_fruit: int,   # default 3
    pos_ctrl_end_samples_per_vessel: int,             # default 1
) -> tuple[pd.DataFrame, dict]:

    rows: list[dict] = []

    def add_row(
        stage: str,
        group: str,
        toxin: str,
        conc: str,
        tissue_plants: int,
        fruit_plants: int,
        initial_qc_samples: int,
        plant_samples: int,
        pos_ctrl_vessels: int,
        pos_ctrl_end_samples: int,
    ):
        tissue_plants = int(tissue_plants)
        fruit_plants = int(fruit_plants)
        initial_qc_samples = int(initial_qc_samples)
        plant_samples = int(plant_samples)
        pos_ctrl_vessels = int(pos_ctrl_vessels)
        pos_ctrl_end_samples = int(pos_ctrl_end_samples)

        total_plants = tissue_plants + fruit_plants
        plant_vessels = total_plants  # 1 vessel per plant (for that stage)

        total_samples = plant_samples + initial_qc_samples + pos_ctrl_end_samples

        rows.append({
            "Experiment": experiment_name,
            "Stage": stage,  # TISSUE_EXPOSURE / FRUIT_EXPOSURE / CONTROLS
            "Group": group,  # TREATED / NEGATIVE_CONTROL
            "Toxin": toxin,
            "Concentration": conc,
            "TissuePlants": tissue_plants,
            "FruitPlants": fruit_plants,
            "TotalPlants": total_plants,
            "PlantVessels": plant_vessels,
            "PlantSamples": plant_samples,
            "InitialQCSamples": initial_qc_samples,
            "PosCtrlVessels": pos_ctrl_vessels,
            "PosCtrlEndSamples": pos_ctrl_end_samples,
            "TotalSamples": total_samples,
        })

    # -----------------------------
    # Stage A: Tissue exposure (all selected tissue concentrations)
    # -----------------------------
    for toxin in toxins:
        for conc in concentrations_tissue:
            tissue_plants = tissue_plants_per_toxin_conc
            fruit_plants = 0

            plant_samples = tissue_plants * samples_per_tissue_plant

            # QC initial samples for this toxin × this conc
            initial_qc = qc_initial_per_toxin_per_conc_tissue

            # Positive control vessels (solution-only) for this toxin × this conc
            pos_vessels = pos_ctrl_vessels_per_toxin_per_conc_tissue
            pos_end_samples = pos_vessels * pos_ctrl_end_samples_per_vessel

            add_row(
                stage="TISSUE_EXPOSURE",
                group="TREATED",
                toxin=toxin,
                conc=conc,
                tissue_plants=tissue_plants,
                fruit_plants=fruit_plants,
                initial_qc_samples=initial_qc,
                plant_samples=plant_samples,
                pos_ctrl_vessels=pos_vessels,
                pos_ctrl_end_samples=pos_end_samples,
            )

    # -----------------------------
    # Stage B: Fruit exposure (only selected fruit concentrations)
    # fruit plants counted per toxin per concentration here.
    # If concentrations_fruit has 1 item (e.g., High), this matches your setup.
    # -----------------------------
    if fruit_plants_per_toxin > 0 and len(concentrations_fruit) > 0:
        for toxin in toxins:
            for conc in concentrations_fruit:
                tissue_plants = 0
                fruit_plants = fruit_plants_per_toxin

                plant_samples = fruit_plants * samples_per_fruit_plant

                # QC initial samples for this toxin × this fruit conc
                initial_qc = qc_initial_per_toxin_per_conc_fruit

                # Positive control vessels (solution-only) for this toxin × this fruit conc
                pos_vessels = pos_ctrl_vessels_per_toxin_per_conc_fruit
                pos_end_samples = pos_vessels * pos_ctrl_end_samples_per_vessel

                add_row(
                    stage="FRUIT_EXPOSURE",
                    group="TREATED",
                    toxin=toxin,
                    conc=conc,
                    tissue_plants=tissue_plants,
                    fruit_plants=fruit_plants,
                    initial_qc_samples=initial_qc,
                    plant_samples=plant_samples,
                    pos_ctrl_vessels=pos_vessels,
                    pos_ctrl_end_samples=pos_end_samples,
                )

    # -----------------------------
    # Negative controls (single groups; no QC and no positive-control vessels)
    # -----------------------------
    if neg_ctrl_tissue_plants_total > 0:
        plant_samples = neg_ctrl_tissue_plants_total * samples_per_tissue_plant
        add_row(
            stage="CONTROLS",
            group="NEGATIVE_CONTROL",
            toxin="None",
            conc="ALL",
            tissue_plants=neg_ctrl_tissue_plants_total,
            fruit_plants=0,
            initial_qc_samples=0,
            plant_samples=plant_samples,
            pos_ctrl_vessels=0,
            pos_ctrl_end_samples=0,
        )

    if neg_ctrl_fruit_plants_total > 0:
        plant_samples = neg_ctrl_fruit_plants_total * samples_per_fruit_plant
        add_row(
            stage="CONTROLS",
            group="NEGATIVE_CONTROL",
            toxin="None",
            conc="ALL",
            tissue_plants=0,
            fruit_plants=neg_ctrl_fruit_plants_total,
            initial_qc_samples=0,
            plant_samples=plant_samples,
            pos_ctrl_vessels=0,
            pos_ctrl_end_samples=0,
        )

    df = pd.DataFrame(rows)

    totals = {
        "TissuePlants": int(df["TissuePlants"].sum()) if not df.empty else 0,
        "FruitPlants": int(df["FruitPlants"].sum()) if not df.empty else 0,
        "TotalPlants": int(df["TotalPlants"].sum()) if not df.empty else 0,
        "PlantVessels": int(df["PlantVessels"].sum()) if not df.empty else 0,
        "PlantSamples": int(df["PlantSamples"].sum()) if not df.empty else 0,
        "InitialQCSamples": int(df["InitialQCSamples"].sum()) if not df.empty else 0,
        "PosCtrlVessels": int(df["PosCtrlVessels"].sum()) if not df.empty else 0,
        "PosCtrlEndSamples": int(df["PosCtrlEndSamples"].sum()) if not df.empty else 0,
        "TotalSamples": int(df["TotalSamples"].sum()) if not df.empty else 0,
    }

    return df, totals


# -----------------------------
# UI
# -----------------------------

st.title("Sample Counter")
st.caption(
    "Counts plant-derived samples + initial exposure QC samples + end-solution samples from positive-control vessels "
    "(solution + toxin, no plant). Separate Tissue and Fruit exposure stages."
)

with st.sidebar:
    st.header("Experimental Design")

    toxins_input = st.text_input("Toxins (comma-separated)", value="Brodifacoum, Bromadiolone")
    toxins = parse_csv_list(toxins_input)

    n_conc = st.number_input("Number of concentrations", min_value=1, max_value=6, value=2, step=1)
    default_concs = default_concentration_names(int(n_conc))
    conc_input = st.text_input("Concentration names (comma-separated)", value=", ".join(default_concs))
    concentrations = parse_csv_list(conc_input)

    st.divider()
    st.header("MAIN — Tissue exposure (per toxin × concentration)")
    tissue_plants_per_toxin_conc = st.number_input(
        "Tissue plants per toxin × concentration",
        min_value=0,
        value=5,
        step=1
    )

    st.divider()
    st.header("MAIN — Fruit exposure (per toxin, usually High only)")
    fruit_plants_per_toxin = st.number_input(
        "Fruit plants per toxin (fruit exposure stage)",
        min_value=0,
        value=3,
        step=1
    )
    fruit_concs = st.multiselect(
        "Fruit exposure concentrations (usually High only)",
        options=concentrations,
        default=[concentrations[-1]] if concentrations else []
    )

    st.divider()
    st.header("Negative Controls (totals)")
    neg_ctrl_tissue_plants_total = st.number_input("Negative control tissue plants (total)", min_value=0, value=5, step=1)
    neg_ctrl_fruit_plants_total = st.number_input("Negative control fruit plants (total)", min_value=0, value=5, step=1)

    st.divider()
    st.header("Sampling Rules (per plant)")
    samples_per_tissue_plant = st.number_input(
        "Samples per tissue plant (root + leaf + end-solution)",
        min_value=0,
        value=3,
        step=1
    )
    samples_per_fruit_plant = st.number_input(
        "Samples per fruit plant (fruit + end-solution)",
        min_value=0,
        value=2,
        step=1
    )

    st.divider()
    st.header("Initial Exposure QC (start; from prepared exposure solution)")
    qc_initial_per_toxin_per_conc_tissue = st.number_input(
        "QC samples per toxin × concentration (tissue exposure start)",
        min_value=0,
        value=3,
        step=1
    )
    qc_initial_per_toxin_per_conc_fruit = st.number_input(
        "QC samples per toxin × concentration (fruit exposure start)",
        min_value=0,
        value=3,
        step=1
    )

    st.divider()
    st.header("Positive-Control Vessels (solution + toxin, no plant; sampled at end)")
    pos_ctrl_vessels_per_toxin_per_conc_tissue = st.number_input(
        "Pos-control vessels per toxin × concentration (tissue stage)",
        min_value=0,
        value=3,
        step=1
    )
    pos_ctrl_vessels_per_toxin_per_conc_fruit = st.number_input(
        "Pos-control vessels per toxin × concentration (fruit stage)",
        min_value=0,
        value=3,
        step=1
    )
    pos_ctrl_end_samples_per_vessel = st.number_input(
        "End-solution samples per pos-control vessel",
        min_value=0,
        value=1,
        step=1
    )

    st.divider()
    enable_prelim = st.checkbox("Include preliminary experiment", value=False)

# Validation
if not toxins:
    st.error("Please enter at least one toxin.")
    st.stop()

if not concentrations:
    st.error("Please enter at least one concentration name.")
    st.stop()

if fruit_plants_per_toxin > 0 and not fruit_concs:
    st.warning("Fruit plants per toxin > 0, but no fruit concentrations selected. Fruit exposure stage will be skipped.")


# -----------------------------
# Main experiment (two-stage)
# -----------------------------
main_df, main_totals = build_experiment_two_stage(
    experiment_name="Main",
    toxins=toxins,
    concentrations_tissue=concentrations,
    concentrations_fruit=fruit_concs,
    tissue_plants_per_toxin_conc=int(tissue_plants_per_toxin_conc),
    fruit_plants_per_toxin=int(fruit_plants_per_toxin),
    neg_ctrl_tissue_plants_total=int(neg_ctrl_tissue_plants_total),
    neg_ctrl_fruit_plants_total=int(neg_ctrl_fruit_plants_total),
    samples_per_tissue_plant=int(samples_per_tissue_plant),
    samples_per_fruit_plant=int(samples_per_fruit_plant),
    qc_initial_per_toxin_per_conc_tissue=int(qc_initial_per_toxin_per_conc_tissue),
    qc_initial_per_toxin_per_conc_fruit=int(qc_initial_per_toxin_per_conc_fruit),
    pos_ctrl_vessels_per_toxin_per_conc_tissue=int(pos_ctrl_vessels_per_toxin_per_conc_tissue),
    pos_ctrl_vessels_per_toxin_per_conc_fruit=int(pos_ctrl_vessels_per_toxin_per_conc_fruit),
    pos_ctrl_end_samples_per_vessel=int(pos_ctrl_end_samples_per_vessel),
)

st.subheader("Main Experiment — Summary")
m1, m2, m3, m4, m5, m6, m7, m8, m9 = st.columns(9)
m1.metric("Tissue plants", main_totals["TissuePlants"])
m2.metric("Fruit plants", main_totals["FruitPlants"])
m3.metric("Total plants", main_totals["TotalPlants"])
m4.metric("Plant vessels", main_totals["PlantVessels"])
m5.metric("Plant samples", main_totals["PlantSamples"])
m6.metric("Initial QC samples", main_totals["InitialQCSamples"])
m7.metric("Pos-control vessels", main_totals["PosCtrlVessels"])
m8.metric("Pos-control end samples", main_totals["PosCtrlEndSamples"])
m9.metric("Total samples", main_totals["TotalSamples"])

with st.expander("Detailed table (Main)"):
    st.dataframe(main_df, use_container_width=True, hide_index=True)


# -----------------------------
# Preliminary experiment (optional)
# -----------------------------
pre_df = pd.DataFrame()
pre_totals = {k: 0 for k in main_totals}

if enable_prelim:
    st.divider()
    st.subheader("Preliminary Experiment")

    pre_concentrations = st.multiselect(
        "Concentrations to include (preliminary tissue exposure)",
        options=concentrations,
        default=[concentrations[-1]] if concentrations else []
    )

    if not pre_concentrations:
        st.error("Please select at least one concentration for the preliminary experiment.")
        st.stop()

    colA, colB, colC = st.columns(3)
    with colA:
        pre_tissue_plants_per_toxin_conc = st.number_input(
            "Tissue plants per toxin × concentration (preliminary)",
            min_value=0, value=3, step=1
        )
        pre_fruit_plants_per_toxin = st.number_input(
            "Fruit plants per toxin (preliminary) — usually 0",
            min_value=0, value=0, step=1
        )
    with colB:
        pre_qc_tissue = st.number_input(
            "QC samples per toxin × concentration (preliminary tissue start)",
            min_value=0, value=3, step=1
        )
        pre_qc_fruit = st.number_input(
            "QC samples per toxin × concentration (preliminary fruit start)",
            min_value=0, value=3, step=1
        )
    with colC:
        pre_neg_tissue = st.number_input("Negative control tissue plants (preliminary)", min_value=0, value=3, step=1)
        pre_neg_fruit = st.number_input("Negative control fruit plants (preliminary)", min_value=0, value=0, step=1)

    st.divider()
    st.caption("Preliminary positive-control vessels (solution + toxin, no plant; sampled at end).")
    colD, colE = st.columns(2)
    with colD:
        pre_pos_ctrl_tissue = st.number_input(
            "Pos-control vessels per toxin × concentration (preliminary tissue stage)",
            min_value=0, value=3, step=1
        )
    with colE:
        pre_pos_ctrl_fruit = st.number_input(
            "Pos-control vessels per toxin × concentration (preliminary fruit stage)",
            min_value=0, value=0, step=1
        )

    pre_df, pre_totals = build_experiment_two_stage(
        experiment_name="Preliminary",
        toxins=toxins,
        concentrations_tissue=pre_concentrations,
        concentrations_fruit=fruit_concs,  # usually empty for preliminary
        tissue_plants_per_toxin_conc=int(pre_tissue_plants_per_toxin_conc),
        fruit_plants_per_toxin=int(pre_fruit_plants_per_toxin),
        neg_ctrl_tissue_plants_total=int(pre_neg_tissue),
        neg_ctrl_fruit_plants_total=int(pre_neg_fruit),
        samples_per_tissue_plant=int(samples_per_tissue_plant),
        samples_per_fruit_plant=int(samples_per_fruit_plant),
        qc_initial_per_toxin_per_conc_tissue=int(pre_qc_tissue),
        qc_initial_per_toxin_per_conc_fruit=int(pre_qc_fruit),
        pos_ctrl_vessels_per_toxin_per_conc_tissue=int(pre_pos_ctrl_tissue),
        pos_ctrl_vessels_per_toxin_per_conc_fruit=int(pre_pos_ctrl_fruit),
        pos_ctrl_end_samples_per_vessel=int(pos_ctrl_end_samples_per_vessel),
    )

    st.subheader("Preliminary Experiment — Summary")
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = st.columns(9)
    p1.metric("Tissue plants", pre_totals["TissuePlants"])
    p2.metric("Fruit plants", pre_totals["FruitPlants"])
    p3.metric("Total plants", pre_totals["TotalPlants"])
    p4.metric("Plant vessels", pre_totals["PlantVessels"])
    p5.metric("Plant samples", pre_totals["PlantSamples"])
    p6.metric("Initial QC samples", pre_totals["InitialQCSamples"])
    p7.metric("Pos-control vessels", pre_totals["PosCtrlVessels"])
    p8.metric("Pos-control end samples", pre_totals["PosCtrlEndSamples"])
    p9.metric("Total samples", pre_totals["TotalSamples"])

    with st.expander("Detailed table (Preliminary)"):
        st.dataframe(pre_df, use_container_width=True, hide_index=True)


# -----------------------------
# Combined summary + export
# -----------------------------
st.divider()
st.subheader("Combined Summary (Main + Preliminary)")

combined_df = pd.concat([df for df in [main_df, pre_df] if not df.empty], ignore_index=True)

# Ensure both dicts share keys (they do) and sum safely
combined_totals = {k: int(main_totals.get(k, 0)) + int(pre_totals.get(k, 0)) for k in main_totals}

c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
c1.metric("Tissue plants", combined_totals["TissuePlants"])
c2.metric("Fruit plants", combined_totals["FruitPlants"])
c3.metric("Total plants", combined_totals["TotalPlants"])
c4.metric("Plant vessels", combined_totals["PlantVessels"])
c5.metric("Plant samples", combined_totals["PlantSamples"])
c6.metric("Initial QC samples", combined_totals["InitialQCSamples"])
c7.metric("Pos-control vessels", combined_totals["PosCtrlVessels"])
c8.metric("Pos-control end samples", combined_totals["PosCtrlEndSamples"])
c9.metric("Total samples", combined_totals["TotalSamples"])

st.dataframe(combined_df, use_container_width=True, hide_index=True)

st.divider()
csv = combined_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download CSV",
    data=csv,
    file_name="sample_counter.csv",
    mime="text/csv",
)
