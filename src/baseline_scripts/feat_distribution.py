import pandas as pd

DATA_PATH = "data/preprocessed/Binerized_features.csv"

BIN_FEATURES = [
    "Flow_Duration_bin",
    "Tot_Fwd_Pkts_bin",
    "Tot_Bwd_Pkts_bin",
    "Pkt_Len_Mean_bin",
    "Flow_Bytes_s_bin",
    "ACK_Flag_Cnt_bin",
    "Init_Fwd_Win_Byts_bin",
    "SYN_Flag_Cnt_bin",
    "syn_ratio_bin",
]

def malicious_ratio_by_bin(df, feature):
    """
    Returns a table where each row is bin_value and columns are:
        0 → P(benign | bin)
        1 → P(malicious | bin)
    """
    tmp = (
        df.groupby(feature)["label"]
          .value_counts(normalize=True)
          .unstack()
          .fillna(0)
    )
    return tmp

def analyze():
    df = pd.read_csv(DATA_PATH)

    print("\n=== CLASS DISTRIBUTION ===")
    print(df["label"].value_counts(normalize=True), "\n")

    for feat in BIN_FEATURES:
        print(f"\n\n========== {feat} ==========")
        ratio = malicious_ratio_by_bin(df, feat)
        print(ratio)

        # Highlight malicious-heavy bins
        print("\nMalicious-skewed bins (P(malicious) > 0.7):")
        for bin_val, row in ratio.iterrows():
            if row.get(1, 0) > 0.7:
                print(f"  {feat} == {bin_val} → P(malicious) = {row[1]:.2f}")


if __name__ == "__main__":
    analyze()
