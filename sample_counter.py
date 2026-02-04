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


def build_experiment(
    experiment_name: str,
    toxins: list[str],
    concentrations: list[str],
    fruit_concs: list[str],
    tissue_plants_per_treatment: int,
    fruit_plants_per_treatment: int,
    pos_ctrl_vessels_per_toxin_tissue: int,
    pos_ctrl_vessels_per_toxin_fruit: int,
    neg_ctrl_tissue_plants_total: int,
    neg_ctrl_fruit_plants_total: int,
    solution_samples_per_vessel: int,
    tissue_samples_per_tissueplant: int,
    fruit_samples_per_fruitplant: int,
    include_pos_ctrl_fruit: bool,
) -> tuple[pd.DataFrame, dict]:

    rows: list[dict] = []

    def add_row(
        group: str,
        toxin: str,
        conc: str,
        tissue_plants: int,
        fruit_plants: int,
        control_vessels: int,
    ):
        tissue_plants = int(tissue_plants)
        fruit_plants = int(fruit_plants)
        control_vessels = int(control_vessels)

        total_plants = tissue_plants + fruit_plants
        plant_vessels = total_plants  # 1 vessel per plant
        total_vessels = plant_vessels + control_vessels

        solution_samples = total_vessels * int(solution_samples_per_vessel)
        tissue_samples = tissue_plants * int(tissue_samples_per_tissueplant)
        fruit_samples = fruit_plants * int(fruit_samples_per_fruitplant)

        rows.append(
            {
                "Experiment": experiment_name,
                "Group": group,  # TREATED / NEGATIVE_CONTROL / POS_CONTROL_TISSUE / POS_CONTROL_FRUIT
                "Toxin": toxin,
                "Concentration": conc,
                "TissuePlants": tissue_plants,
                "FruitPlants": fruit_plants,
                "ControlVessels": control_vessels,
                "PlantVessels": plant_vessels,
                "TotalPlants": total_plants,
                "TotalVessels": total_vessels,
                "SolutionSamples": solution_samples,
                "TissueSamples": tissue_samples,
                "FruitSamples": fruit_samples,
                "TotalSamples": solution_samples + tissue_samples + fruit_samples,
            }
        )

    # Treated rows (no controls embedded here)
    for toxin in toxins:
        for conc in concentrations:
            fruit_plants_here = fruit_plants_per_treatment if conc in fruit_concs else 0
            add_row(
                group="TREATED",
                toxin=toxin,
                conc=conc,
                tissue_plants=tissue_plants_per_treatment,
                fruit_plants=fruit_plants_here,
                control_vessels=0,
            )

    # Negative control (single group)
    if (neg_ctrl_tissue_plants_total + neg_ctrl_fruit_plants_total) > 0:
        add_row(
            group="NEGATIVE_CONTROL",
            toxin="None",
            conc="ALL",
            tissue_plants=neg_ctrl_tissue_plants_total,
            fruit_plants=neg_ctrl_fruit_plants_total,
            control_vessels=0,
        )

    # Positive controls per toxin (separate for tissue vs fruit)
    for toxin in toxins:
        if pos_ctrl_vessels_per_toxin_tissue > 0:
            add_row(
                group="POS_CONTROL_TISSUE",
                toxin=toxin,
                conc="ALL",
                tissue_plants=0,
                fruit_plants=0,
                control_vessels=pos_ctrl_vessels_per_toxin_tissue,
            )

        if include_pos_ctrl_fruit and pos_ctrl_vessels_per_toxin_fruit > 0:
            add_row(
                group="POS_CONTROL_FRUIT",
                toxin=toxin,
                conc="ALL",
                tissue_plants=0,
                fruit_plants=0,
                control_vessels=pos_ctrl_vessels_per_toxin_fruit,
            )

    df = pd.DataFrame(rows)

    totals = {
        "TissuePlants": int(df["TissuePlants"].sum()) if not df.empty else 0,
        "FruitPlants": int(df["FruitPlants"].sum()) if not df.empty else 0,
        "TotalPlants": int(df["TotalPlants"].sum()) if not df.empty else 0,
        "ControlVessels": int(df["ControlVessels"].sum()) if not df.empty else 0,
        "PlantVessels": int(df["PlantVessels"].sum()) if not df.empty else 0,
        "TotalVessels": int(df["TotalVessels"].sum()) if not df.empty else 0,
        "SolutionSamples": int(df["SolutionSamples"].sum()) if not df.empty else 0,
        "TissueSamples": int(df["TissueSamples"].sum()) if not df.empty else 0,
        "FruitSamples": int(df["FruitSamples"].sum()) if not df.empty else 0,
        "TotalSamples": int(df["TotalSamples"].sum()) if not df.empty else 0,
    }

    return df, totals


# -----------------------------
# UI
# -----------------------------

st.title("Sample Counter")

with st.sidebar:
    st.header("Experimental Design")

    toxins_input = st.text_input("Toxins (comma-separated)", value="Brodifacoum, Bromadiolone")
    toxins = parse_csv_list(toxins_input)

    n_conc = st.number_input("Number of concentrations", min_value=1, max_value=6, value=2, step=1)
    default_concs = default_concentration_names(int(n_conc))
    conc_input = st.text_input("Concentration names (comma-separated)", value=", ".join(default_concs))
    concentrations = parse_csv_list(conc_input)

    st.divider()
    st.header("Plants per Treatment (per toxin × concentration)")
    tissue_plants_per_treatment = st.number_input("Tissue plants per treatment (root + leaf)", min_value=0, value=5, step=1)
    fruit_plants_per_treatment = st.number_input("Fruit plants per treatment", min_value=0, value=5, step=1)

    fruit_concs = st.multiselect(
        "Concentrations that include fruit plants",
        options=concentrations,
        default=[concentrations[-1]] if concentrations else []
    )

    st.divider()
    st.header("Positive Controls (per toxin)")
    pos_ctrl_vessels_per_toxin_tissue = st.number_input(
        "Positive control vessels per toxin (TISSUE)",
        min_value=0,
        value=3,
        step=1
    )
    pos_ctrl_vessels_per_toxin_fruit = st.number_input(
        "Positive control vessels per toxin (FRUIT)",
        min_value=0,
        value=3,
        step=1
    )

    st.divider()
    st.header("Negative Controls (single groups)")
    neg_ctrl_tissue_plants_total = st.number_input("Negative control tissue plants (total)", min_value=0, value=5, step=1)
    neg_ctrl_fruit_plants_total = st.number_input("Negative control fruit plants (total)", min_value=0, value=5, step=1)

    st.divider()
    st.header("Sampling Rules")
    solution_samples_per_vessel = st.number_input("Solution samples per vessel", min_value=0, value=2, step=1)
    tissue_samples_per_tissueplant = st.number_input("Tissue samples per tissue plant", min_value=0, value=2, step=1)
    fruit_samples_per_fruitplant = st.number_input("Fruit samples per fruit plant", min_value=0, value=1, step=1)

    st.divider()
    enable_prelim = st.checkbox("Include preliminary experiment", value=False)

# Validation
if not toxins:
    st.error("Please enter at least one toxin.")
    st.stop()

if not concentrations:
    st.error("Please enter at least one concentration name.")
    st.stop()

if not fruit_concs and fruit_plants_per_treatment > 0:
    st.warning("You entered fruit plants per treatment, but selected no concentrations for fruit plants. Fruit plants will be counted as 0.")

has_any_fruit_in_main = (fruit_plants_per_treatment > 0 and len(fruit_concs) > 0) or (neg_ctrl_fruit_plants_total > 0)

# -----------------------------
# Main experiment
# -----------------------------
main_df, main_totals = build_experiment(
    experiment_name="Main",
    toxins=toxins,
    concentrations=concentrations,
    fruit_concs=fruit_concs,
    tissue_plants_per_treatment=int(tissue_plants_per_treatment),
    fruit_plants_per_treatment=int(fruit_plants_per_treatment),
    pos_ctrl_vessels_per_toxin_tissue=int(pos_ctrl_vessels_per_toxin_tissue),
    pos_ctrl_vessels_per_toxin_fruit=int(pos_ctrl_vessels_per_toxin_fruit),
    neg_ctrl_tissue_plants_total=int(neg_ctrl_tissue_plants_total),
    neg_ctrl_fruit_plants_total=int(neg_ctrl_fruit_plants_total),
    solution_samples_per_vessel=int(solution_samples_per_vessel),
    tissue_samples_per_tissueplant=int(tissue_samples_per_tissueplant),
    fruit_samples_per_fruitplant=int(fruit_samples_per_fruitplant),
    include_pos_ctrl_fruit=has_any_fruit_in_main,
)

st.subheader("Main Experiment — Summary")
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Tissue plants", main_totals["TissuePlants"])
m2.metric("Fruit plants", main_totals["FruitPlants"])
m3.metric("Total plants", main_totals["TotalPlants"])
m4.metric("Control vessels", main_totals["ControlVessels"])
m5.metric("Plant vessels", main_totals["PlantVessels"])
m6.metric("Solution samples", main_totals["SolutionSamples"])
m7.metric("Total samples", main_totals["TotalSamples"])

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
        "Concentrations to include (preliminary)",
        options=concentrations,
        default=[concentrations[-1]] if concentrations else []
    )

    if not pre_concentrations:
        st.error("Please select at least one concentration for the preliminary experiment.")
        st.stop()

    colA, colB, colC = st.columns(3)
    with colA:
        pre_tissue = st.number_input("Tissue plants per treatment (preliminary)", min_value=0, value=3, step=1)
        pre_fruit = st.number_input("Fruit plants per treatment (preliminary)", min_value=0, value=0, step=1)
    with colB:
        pre_pos_tissue = st.number_input("Pos ctrl vessels per toxin (preliminary, TISSUE)", min_value=0, value=3, step=1)
        pre_pos_fruit = st.number_input("Pos ctrl vessels per toxin (preliminary, FRUIT)", min_value=0, value=0, step=1)
    with colC:
        pre_neg_tissue = st.number_input("Negative control tissue plants (preliminary)", min_value=0, value=3, step=1)
        pre_neg_fruit = st.number_input("Negative control fruit plants (preliminary)", min_value=0, value=0, step=1)

    has_any_fruit_in_pre = (pre_fruit > 0 and len(fruit_concs) > 0) or (pre_neg_fruit > 0)

    pre_df, pre_totals = build_experiment(
        experiment_name="Preliminary",
        toxins=toxins,
        concentrations=pre_concentrations,   # FIXED: preliminary has its own concentrations
        fruit_concs=fruit_concs,             # fruit concentrations selection reused (can keep)
        tissue_plants_per_treatment=int(pre_tissue),
        fruit_plants_per_treatment=int(pre_fruit),
        pos_ctrl_vessels_per_toxin_tissue=int(pre_pos_tissue),
        pos_ctrl_vessels_per_toxin_fruit=int(pre_pos_fruit),
        neg_ctrl_tissue_plants_total=int(pre_neg_tissue),
        neg_ctrl_fruit_plants_total=int(pre_neg_fruit),
        solution_samples_per_vessel=int(solution_samples_per_vessel),
        tissue_samples_per_tissueplant=int(tissue_samples_per_tissueplant),
        fruit_samples_per_fruitplant=int(fruit_samples_per_fruitplant),
        include_pos_ctrl_fruit=has_any_fruit_in_pre,
    )

    st.subheader("Preliminary Experiment — Summary")
    p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
    p1.metric("Tissue plants", pre_totals["TissuePlants"])
    p2.metric("Fruit plants", pre_totals["FruitPlants"])
    p3.metric("Total plants", pre_totals["TotalPlants"])
    p4.metric("Control vessels", pre_totals["ControlVessels"])
    p5.metric("Plant vessels", pre_totals["PlantVessels"])
    p6.metric("Solution samples", pre_totals["SolutionSamples"])
    p7.metric("Total samples", pre_totals["TotalSamples"])

    with st.expander("Detailed table (Preliminary)"):
        st.dataframe(pre_df, use_container_width=True, hide_index=True)

# -----------------------------
# Combined summary + export
# -----------------------------
st.divider()
st.subheader("Combined Summary (Main + Preliminary)")

combined_df = pd.concat([df for df in [main_df, pre_df] if not df.empty], ignore_index=True)
combined_totals = {k: int(main_totals.get(k, 0)) + int(pre_totals.get(k, 0)) for k in main_totals}

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Tissue plants", combined_totals["TissuePlants"])
c2.metric("Fruit plants", combined_totals["FruitPlants"])
c3.metric("Total plants", combined_totals["TotalPlants"])
c4.metric("Control vessels", combined_totals["ControlVessels"])
c5.metric("Plant vessels", combined_totals["PlantVessels"])
c6.metric("Solution samples", combined_totals["SolutionSamples"])
c7.metric("Total samples", combined_totals["TotalSamples"])

st.dataframe(combined_df, use_container_width=True, hide_index=True)

st.divider()
csv = combined_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download CSV",
    data=csv,
    file_name="sample_counter.csv",
    mime="text/csv",
)
