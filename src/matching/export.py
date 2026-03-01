import pandas as pd


def export_to_excel(df: pd.DataFrame, path_xlsx: str, project_id_col: str) -> None:
    with pd.ExcelWriter(path_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="top_k_por_proyecto", index=False)
        top1 = df.sort_values([project_id_col, "rank"]).groupby(project_id_col, as_index=False).first()
        top1.to_excel(writer, sheet_name="top_1_por_proyecto", index=False)
