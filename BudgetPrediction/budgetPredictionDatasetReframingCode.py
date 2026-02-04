
import pandas as pd

df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)


df = df.replace(r"^\s*$", pd.NA, regex=True)


final_df = (
    df.groupby(
        [
            "Program_Description",
            "Function_Description",
            "Fund_Description",
            "Location_Description" 
        ],
        dropna=False
    )
    .agg(
        total_budget=("Total", "sum"),
        total_fte=("FTE", "sum"),
        num_records=("Total", "count")
    )
    .reset_index()
)


for col in ["Program_Description", "Function_Description"]:
    top = final_df[col].value_counts().nlargest(15).index
    final_df[col] = final_df[col].where(final_df[col].isin(top), "Other")


final_df = (
    final_df.groupby(
        [
            "Program_Description",
            "Function_Description",
            "Fund_Description",
            "Location_Description"
        ],
        dropna=False
    )
    .agg(
        total_budget=("total_budget", "sum"),
        total_fte=("total_fte", "sum"),
        num_records=("num_records", "sum")
    )
    .reset_index()
)


final_df.to_csv("FINAL_budget_dataset.csv", index=False)

print("Final dataset saved as FINAL_budget_dataset.csv")
print("Rows:", final_df.shape[0])
print("Columns:", final_df.shape[1])
