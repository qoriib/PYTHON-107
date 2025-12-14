import pandas as pd
import streamlit as st

from i18n import t
from utils import (
    coerce_numeric,
    correlation_block,
    correlation_ci,
    descriptive_stats,
    download_with_totals,
    load_csv,
    make_hist,
    make_scatter,
    strength_label,
)

DEFAULT_X = [
    "Saya sering melakukan pembelian melalui platform belanja online.",
    "Saya sering melihat-lihat produk di e-commerce meskipun tidak ada niat untuk membeli. ",
    "Saya mudah tergoda dengan promo, diskon, atau gratis ongkir saat berbelanja online. ",
    "Saya pernah membeli barang secara spontan tanpa perencanaan sebelumnya. ",
    "Saya menghabiskan cukup banyak waktu untuk menjelajahi produk di aplikasi belanja online. ",
    "Saya sering menambahkan barang ke keranjang toko online tersebut meskipun belum tentu membeli. ",
    "Saya mengikuti tren atau rekomendasi online saat membeli produk tertentu. ",
    "Saya merasa tertarik membeli produk hanya karena ulasan atau rating yang tinggi. ",
    "Saya lebih memilih berbelanja online daripada berbelanja langsung di toko fisik. ",
    "Saya sering membeli barang yang sebenarnya bukan prioritas hanya karena sedang ada potongan harga.",
]

DEFAULT_Y = [
    "Saya mampu menahan diri untuk tidak membeli barang yang tidak saya butuhkan.  ",
    "Saya berusaha menjaga pengeluaran agar tetap sesuai dengan anggaran pribadi. ",
    "Saya lebih mengutamakan kebutuhan dibandingkan keinginan saat berbelanja. ",
    "Saya mempertimbangkan kembali sebelum membeli barang secara impulsif. ",
    "Saya selalu mengevaluasi kondisi keuangan sebelum memutuskan untuk membeli sesuatu. ",
    "Saya berusaha disiplin dalam mengikuti batasan pengeluaran bulanan yang saya tetapkan. ",
    "Saya menghindari pembelian barang yang tidak masuk dalam rencana pengeluaran saya. ",
    "Saya menghindari membeli barang hanya karena tren. ",
    "Saya berusaha mengontrol diri ketika melihat promo atau diskon yang menarik. ",
    "Saya mempertimbangkan dampak jangka panjang dari setiap keputusan belanja yang saya buat. ",
]

DEFAULT_DEMOGRAPHICS = [
    "Nama Anda",
    "  Berapa usia Anda?  ",
    "Jenis Kelamin  ",
    "Pekerjaan Anda",
    "Apakah Anda pernah melakukan belanja online dalam 3 bulan terakhir? ",
    "Platform belanja online yang paling sering Anda gunakan.",
    "Frekuensi belanja online per bulan  ",
    "Rata-rata pengeluaran untuk belanja online per bulan  ",
]


st.set_page_config(page_title="Analisis Perilaku Belanja Online vs Kontrol Diri", layout="wide")


def main():
    if "df" not in st.session_state:
        st.session_state.df = None

    with st.sidebar:
        lang = st.selectbox(
            "Language / Bahasa", ["id", "en"], format_func=lambda x: "Bahasa Indonesia" if x == "id" else "English"
        )
    st.title(t("title", lang))

    with st.sidebar:
        st.header(t("navigation", lang))
        nav = st.radio(
            t("nav_prompt", lang),
            ["Import", "Demographics", "Descriptive Statistics", "Association Analysis", "Export"],
            format_func=lambda x: {
                "Import": t("nav_upload", lang),
                "Demographics": t("nav_demo", lang),
                "Descriptive Statistics": t("nav_desc", lang),
                "Association Analysis": t("nav_assoc", lang),
                "Export": t("nav_export", lang),
            }[x],
            index=0,
        )

        df = st.session_state.df
        if df is not None:
            x_default = [c for c in DEFAULT_X if c in df.columns]
            y_default = [c for c in DEFAULT_Y if c in df.columns]
            demo_default = [c for c in DEFAULT_DEMOGRAPHICS if c in df.columns]
            st.subheader(t("variable_section", lang))
            x_cols = st.multiselect(
                t("x_label", lang), df.columns.tolist(), default=x_default
            )
            y_cols = st.multiselect(
                t("y_label", lang), df.columns.tolist(), default=y_default
            )
            demographics = st.multiselect(t("demo_label", lang), df.columns.tolist(), default=demo_default)
            agg_method = st.radio(
                t("agg_label", lang),
                ["Mean", "Sum"],
                index=0,
                format_func=lambda x: t(x.lower(), lang),
            )
        else:
            nav = "Import"
            x_cols, y_cols, demographics, agg_method = [], [], [], "Mean"
            st.info(t("need_upload_menu", lang))

    if st.session_state.df is None:
        if nav == "Import":
            with st.container():
                st.subheader(t("description_title", lang))
                st.markdown(t("description", lang))
                st.markdown(t("description_steps", lang))
            st.subheader(t("upload_section", lang))
            uploaded = st.file_uploader(t("file_label", lang), type=["csv"])
            if uploaded is not None:
                st.session_state.df = load_csv(uploaded)

            if st.session_state.df is not None:
                st.success(
                    t("data_loaded_brief", lang).format(
                        rows=st.session_state.df.shape[0], cols=st.session_state.df.shape[1]
                    )
                )
                st.dataframe(st.session_state.df.head(10))
            else:
                st.info(t("upload_needed", lang))
        return

    df = st.session_state.df

    if nav == "Import":
        st.subheader(t("description_title", lang))
        st.markdown(t("description", lang))
        st.markdown(t("description_steps", lang))
        # When data already imported but user stays on Import menu
        st.subheader(t("upload_section", lang))
        st.success(
            t("data_loaded", lang).format(rows=df.shape[0], cols=df.shape[1])
        )
        st.dataframe(df.head(10))
        return

    all_numeric_cols = list(set(x_cols + y_cols))
    numeric_df = coerce_numeric(df, all_numeric_cols)
    x_total = None
    y_total = None
    if x_cols:
        x_total = numeric_df[x_cols].mean(axis=1) if agg_method == "Mean" else numeric_df[x_cols].sum(axis=1)
    if y_cols:
        y_total = numeric_df[y_cols].mean(axis=1) if agg_method == "Mean" else numeric_df[y_cols].sum(axis=1)

    if nav == "Demographics":
        st.subheader(t("demographics_header", lang))
        if not demographics:
            st.info(t("demo_prompt", lang))
        else:
            for col in demographics:
                st.markdown(f"{col}")
                freq = df[col].value_counts(dropna=False).reset_index()
                freq.columns = [col, "Frekuensi"]
                freq["Persentase (%)"] = (freq["Frekuensi"] / df.shape[0] * 100).round(2)
                st.dataframe(freq)

    elif nav == "Descriptive Statistics":
        st.subheader(t("desc_header", lang))
        if not (x_cols or y_cols):
            st.info(t("desc_prompt", lang))
        else:
            if x_cols:
                st.markdown(f"**{t('x_item', lang)}**")
                x_stats = {col: descriptive_stats(numeric_df[col]) for col in x_cols}
                st.dataframe(pd.DataFrame(x_stats).T)
            if y_cols:
                st.markdown(f"**{t('y_item', lang)}**")
                y_stats = {col: descriptive_stats(numeric_df[col]) for col in y_cols}
                st.dataframe(pd.DataFrame(y_stats).T)
            if x_total is not None:
                st.markdown(f"**{t('x_total', lang)}**")
                st.dataframe(pd.DataFrame([descriptive_stats(x_total)]))
            if y_total is not None:
                st.markdown(f"**{t('y_total', lang)}**")
                st.dataframe(pd.DataFrame([descriptive_stats(y_total)]))

        st.subheader(t("viz_header", lang))
        if x_total is not None or y_total is not None:
            col_left, col_right = st.columns(2)
            if x_total is not None:
                with col_left:
                    st.pyplot(make_hist(x_total, "Histogram X_total"))
            if y_total is not None:
                with col_right:
                    st.pyplot(make_hist(y_total, "Histogram Y_total"))
            if x_total is not None and y_total is not None:
                st.pyplot(make_scatter(x_total, y_total, "Scatter Plot X_total vs Y_total"))
        else:
            st.info(t("viz_prompt", lang))

    elif nav == "Association Analysis":
        st.subheader(t("assoc_header", lang))
        if x_total is None or y_total is None:
            st.info(t("assoc_prompt", lang))
        else:
            corr = correlation_block(x_total, y_total)
            dir_label = t("direction_pos", lang) if corr["Pearson r"] >= 0 else t("direction_neg", lang)
            st.write(t("pairs", lang).format(n=corr["N pairs"]))
            pearson_ci = correlation_ci(corr["Pearson r"], corr["N pairs"])
            summary_rows = []
            summary_rows.append(
                {
                    t("method", lang): "Pearson",
                    t("coef", lang): round(corr["Pearson r"], 3) if pd.notna(corr["Pearson r"]) else None,
                    t("p_value", lang): round(corr["Pearson p-value"], 4) if pd.notna(corr["Pearson p-value"]) else None,
                    t("strength_col", lang): strength_label(corr["Pearson r"]),
                    t("direction_col", lang): dir_label,
                    t("significant", lang): "Yes" if corr["Pearson p-value"] < 0.05 else "No",
                    t("ci", lang): (
                        f"[{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}]" if pd.notna(pearson_ci[0]) else "N/A"
                    ),
                }
            )
            dir_label_s = t("direction_pos", lang) if corr["Spearman rho"] >= 0 else t("direction_neg", lang)
            summary_rows.append(
                {
                    t("method", lang): "Spearman",
                    t("coef", lang): round(corr["Spearman rho"], 3) if pd.notna(corr["Spearman rho"]) else None,
                    t("p_value", lang): round(corr["Spearman p-value"], 4) if pd.notna(corr["Spearman p-value"]) else None,
                    t("strength_col", lang): strength_label(corr["Spearman rho"]),
                    t("direction_col", lang): dir_label_s,
                    t("significant", lang): "Yes" if corr["Spearman p-value"] < 0.05 else "No",
                    t("ci", lang): "N/A",
                }
            )
            st.markdown(f"**{t('assoc_table_title', lang)}**")
            st.dataframe(pd.DataFrame(summary_rows))
            st.markdown(
                t("pearson", lang).format(
                    r=corr["Pearson r"], p=corr["Pearson p-value"], strength=strength_label(corr["Pearson r"]), direction=dir_label
                )
                + "\n"
                + t("spearman", lang).format(
                    r=corr["Spearman rho"], p=corr["Spearman p-value"], strength=strength_label(corr["Spearman rho"])
                )
            )

            st.subheader(t("interpret_header", lang))
            st.markdown(t("interpret_lines", lang))

    elif nav == "Export":
        st.subheader(t("export_header", lang))
        x_export = x_total
        y_export = y_total
        if x_export is None or y_export is None:
            # Fallback: compute using default columns if available
            x_fallback = [c for c in DEFAULT_X if c in df.columns]
            y_fallback = [c for c in DEFAULT_Y if c in df.columns]
            cols_needed = list(set(x_fallback + y_fallback))
            if cols_needed:
                numeric_df_exp = coerce_numeric(df, cols_needed)
                if x_export is None and x_fallback:
                    x_export = (
                        numeric_df_exp[x_fallback].mean(axis=1)
                        if agg_method == "Mean"
                        else numeric_df_exp[x_fallback].sum(axis=1)
                    )
                if y_export is None and y_fallback:
                    y_export = (
                        numeric_df_exp[y_fallback].mean(axis=1)
                        if agg_method == "Mean"
                        else numeric_df_exp[y_fallback].sum(axis=1)
                    )

        if x_export is None and y_export is None:
            st.info(t("export_prompt", lang))
        else:
            csv_bytes = download_with_totals(df, x_export, y_export)
            st.download_button(
                label=t("download_label", lang),
                data=csv_bytes,
                file_name="data_dengan_skor.csv",
                mime="text/csv",
            )
            st.caption(t("download_caption", lang))


if __name__ == "__main__":
    main()
