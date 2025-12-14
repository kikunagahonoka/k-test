import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import re

from back import get_city_data, get_available_cities

# --- 設定 ---
st.set_page_config(layout="wide", page_title="不動産エリア分析ツール")

# =========================
# CSV 読み込み（文字化け対策）
# =========================
def read_csv_flexible(file_or_path, is_path: bool = False) -> pd.DataFrame:
    encodings = ["cp932", "utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            if is_path:
                return pd.read_csv(file_or_path, encoding=enc)
            else:
                try:
                    file_or_path.seek(0)
                except Exception:
                    pass
                return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

# =========================
# 取引CSV（国交省系）を攻略ガイド用に前処理
# =========================
def preprocess_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # ㎡単価（数値化）
    if "取引価格（㎡単価）" in d.columns:
        d["㎡単価"] = pd.to_numeric(d["取引価格（㎡単価）"], errors="coerce")
    else:
        d["㎡単価"] = None

    # 総額（万円）
    if "取引価格（総額）" in d.columns:
        d["price_man"] = pd.to_numeric(d["取引価格（総額）"], errors="coerce") / 10000
    else:
        d["price_man"] = None

    # 面積（㎡）
    def clean_area(x):
        try:
            s = str(x).replace(",", "").replace("㎡以上", "").replace("m^2", "").replace("m2", "")
            # 数字以外混ざる場合の保険
            nums = re.findall(r"[\d.]+", s)
            return float(nums[0]) if nums else None
        except Exception:
            return None

    if "面積（㎡）" in d.columns:
        d["area_m2"] = d["面積（㎡）"].apply(clean_area)
    else:
        d["area_m2"] = None

    # 坪単価（万円/坪） = 総額 / 坪面積
    if "取引価格（総額）" in d.columns and "area_m2" in d.columns:
        total = pd.to_numeric(d["取引価格（総額）"], errors="coerce")
        tsubo = d["area_m2"] / 3.30578
        d["tsubo_price"] = (total / tsubo) / 10000
        d["tsubo_price"] = d["tsubo_price"].round(1)
    else:
        d["tsubo_price"] = None

    # 取引時期（時系列用：文字列のままでもOK、ソートしやすい形へ）
    if "取引時期" in d.columns:
        d["period"] = d["取引時期"].astype(str).str.replace("年第", "-Q", regex=False).str.replace("四半期", "", regex=False)
    else:
        d["period"] = None

    # 駅徒歩（分）
    def clean_minutes(x):
        try:
            nums = re.findall(r"\d+", str(x))
            return int(nums[0]) if nums else None
        except Exception:
            return None

    if "最寄駅：距離（分）" in d.columns:
        d["minutes"] = d["最寄駅：距離（分）"].apply(clean_minutes)
    else:
        d["minutes"] = None

    # 築年数（建築年 → 年数）
    current_year = datetime.datetime.now().year
    def get_age(x):
        m = re.search(r"(\d{4})", str(x))
        if m:
            return max(0, current_year - int(m.group(1)))
        return None

    if "建築年" in d.columns:
        d["age"] = d["建築年"].apply(get_age)
    else:
        d["age"] = None

    return d


# --- サイドバー：分析設定 ---
st.sidebar.title("🛠️ 分析設定")

# 0. 利用可能な市町村のリストを取得
available_cities = get_available_cities()
if not available_cities:
    available_cities = ["川越市"]
    default_cities = ["川越市"]
else:
    default_cities = [available_cities[34]]

# 1. 分析対象エリアの選択（複数選択）
target_cities = st.sidebar.multiselect(
    "分析する市区町村を選択",
    options=available_cities,
    default=default_cities,
    help="複数の市を選ぶと、それら全てのエリアを横断して分析・比較できます。"
)

# 2. 地価データ（取引CSV）のアップロード（なければ test.csv）
uploaded_file = st.sidebar.file_uploader(
    "地価データ (CSV) をアップロード",
    type=["csv"],
    help="国土交通省の不動産取引価格情報など。未アップロードの場合は test.csv を使用します。"
)

uploaded_price_df = None
if uploaded_file is not None:
    uploaded_price_df = read_csv_flexible(uploaded_file, is_path=False)
    if uploaded_price_df.empty:
        st.sidebar.error("読み込みに失敗しました（文字コード/CSV形式を確認）")
    else:
        st.sidebar.success(f"✅ {uploaded_file.name} を読み込みました")
        st.sidebar.caption(f"行数: {len(uploaded_price_df):,}")
else:
    uploaded_price_df = read_csv_flexible("test.csv", is_path=True)
    if uploaded_price_df.empty:
        st.sidebar.warning("📄 test.csv を読み込めませんでした（存在チェック）")
    else:
        st.sidebar.info("📄 test.csv を使用しています（デフォルト）")
        st.sidebar.caption(f"行数: {len(uploaded_price_df):,}")

price_df_pre = preprocess_price_df(uploaded_price_df)

# --- データロード ---
@st.cache_data
def load_data(cities, price_df):
    if not cities:
        return pd.DataFrame(), {}
    return get_city_data(target_city_names=cities, uploaded_price_df=price_df)

if not target_cities:
    st.warning("左のサイドバーから、分析したい市区町村を選んでください。")
    st.stop()

df_city, city_summary = load_data(target_cities, uploaded_price_df)

# --- データチェック ---
if df_city.empty:
    st.error("選択されたエリアのデータが見つかりませんでした。")
    st.stop()

area_list = df_city.index.tolist()
numeric_cols = df_city.select_dtypes(include=["float", "int"]).columns.tolist()

# =========================
# メイン画面
# =========================
cities_str = "・".join(target_cities)
st.title(f"🗺️ {cities_str} エリア攻略＆分析")

tab_guide, tab_compare, tab_group = st.tabs(["🔰 攻略ガイド", "🔍 個別エリア比較", "🆚 グループ対抗"])


# ==========================================
# TAB 1: 🔰 攻略ガイド（ぱっと見＋市場要素追加）
# ==========================================
with tab_guide:
    st.header("エリア攻略ガイド（ぱっと見＋市場データ）")
    # デフォルトを「新富町」にしたい
    default_area = "新富町"

    if default_area in area_list:
        default_index = area_list.index(default_area)
    else:
        default_index = 0  # なければ先頭

    selected_area = st.selectbox(
        "担当エリアを選択",
        area_list,
        index=default_index,
        key="guide_area"
    )

    row = df_city.loc[selected_area]


    # ---- ① ストック（統計）主要KPI ----
    st.subheader("📌 主要指標サマリー（統計）")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("総人口", f"{row.get('総人口', 0):,.0f}")
    c2.metric("世帯数", f"{row.get('世帯総数', 0):,.0f}")
    c3.metric("地価中央値(㎡)", f"¥ {row.get('Median_Price_sqm', 0):,.0f}")
    c4.metric("高齢化率", f"{row.get('高齢化率', 0):.1%}")
    c5.metric("ファミリー率", f"{row.get('ファミリー世帯割合', 0):.1%}")

    st.subheader("📊 全体平均との差（統計ポジション）")
    def metric_vs_avg(label, value, avg, is_percent=True):
        delta = value - avg
        if is_percent:
            st.metric(label, f"{value:.1%}", delta=f"{delta:.1%}")
        else:
            st.metric(label, f"{value:,.0f}", delta=f"{delta:,.0f}")

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1: metric_vs_avg("持ち家率", row.get("持ち家率", 0), city_summary.get("持ち家率", 0), True)
    with d2: metric_vs_avg("借家率", row.get("借家率", 0), city_summary.get("借家率", 0), True)
    with d3: metric_vs_avg("一戸建率", row.get("一戸建率", 0), city_summary.get("一戸建率", 0), True)
    with d4: metric_vs_avg("共同住宅率", row.get("共同住宅率", 0), city_summary.get("共同住宅率", 0), True)
    with d5: metric_vs_avg("単身・少人数", row.get("単身・少人数世帯割合", 0), city_summary.get("単身・少人数世帯割合", 0), True)
    with d6: metric_vs_avg("ファミリー", row.get("ファミリー世帯割合", 0), city_summary.get("ファミリー世帯割合", 0), True)

    vs_avg = px.bar()
    st.plotly_chart()
    st.divider()

    # ---- ② フロー（市場/取引）: この町丁の取引データ抽出 ----
    st.subheader("💰 市場サマリー（取引データ）")

    market = price_df_pre.copy()
    # 市区町村で絞る（列があるときだけ）
    if not market.empty and "市区町村名" in market.columns:
        market = market[market["市区町村名"].isin(target_cities)].copy()
    # 地区名（町丁）で絞る（列があるときだけ）
    if not market.empty and "地区名" in market.columns:
        market_area = market[market["地区名"] == selected_area].copy()
    else:
        market_area = pd.DataFrame()

    if market_area.empty:
        st.info("この担当エリアに一致する取引データ（地区名）が見つかりませんでした。")
    else:
        m1, m2, m3, m4 = st.columns(4)

        # 平均取引価格（万円）
        if "price_man" in market_area.columns and market_area["price_man"].notna().any():
            m1.metric("平均取引価格", f"{market_area['price_man'].mean():,.0f} 万円")
        else:
            m1.metric("平均取引価格", "—")

        # 平均坪単価（万円/坪）
        if "tsubo_price" in market_area.columns and market_area["tsubo_price"].notna().any():
            m2.metric("平均坪単価", f"{market_area['tsubo_price'].mean():,.1f} 万円/坪")
        else:
            m2.metric("平均坪単価", "—")

        # 平均築年数
        if "age" in market_area.columns and market_area["age"].notna().any():
            m3.metric("平均築年数", f"{market_area['age'].mean():.1f} 年")
        else:
            m3.metric("平均築年数", "—")

        # 件数
        m4.metric("データ件数", f"{len(market_area):,} 件")

        # ---- 相場トレンド（時系列） ----
        st.subheader("📈 相場トレンド（時系列）")
        if "period" in market_area.columns and "tsubo_price" in market_area.columns and market_area["tsubo_price"].notna().any():
            trend = market_area.groupby("period")["tsubo_price"].mean().reset_index()
            fig_tr = px.line(trend, x="period", y="tsubo_price", markers=True, title="時期ごとの平均坪単価推移")
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.warning("時系列表示に必要な列（取引時期/坪単価）が不足しています。")

        # ---- 価格帯分布 ----
        st.subheader("📊 価格帯のボリュームゾーン")
        if "price_man" in market_area.columns and market_area["price_man"].notna().any():
            fig_hist = px.histogram(market_area, x="price_man", nbins=20, title="価格帯ごとの取引件数")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("価格帯分布に必要な列（取引価格（総額））が不足しています。")

        # ---- 建物構造分析（市場シェア＋価格レンジ） ----
        st.subheader("🏗️ 建物構造（シェア＆価格レンジ）")
        if "建物の構造" in market_area.columns and market_area["建物の構造"].notna().any() and "tsubo_price" in market_area.columns:
            struct_df = market_area.dropna(subset=["建物の構造"]).copy()

            s1, s2 = st.columns(2)
            with s1:
                fig_pie = px.pie(struct_df, names="建物の構造", title="構造割合（市場シェア）")
                st.plotly_chart(fig_pie, use_container_width=True)
            with s2:
                fig_box = px.box(struct_df, x="建物の構造", y="tsubo_price", color="建物の構造",
                                 title="構造別 坪単価レンジ（箱ひげ）", labels={"tsubo_price": "坪単価(万円/坪)"})
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("#### ⏳ 構造×築年数（経年でどれくらい落ちる？）")
            if "age" in struct_df.columns and struct_df["age"].notna().any():
                fig_sc = px.scatter(
                    struct_df,
                    x="age",
                    y="tsubo_price",
                    color="建物の構造",
                    size="area_m2" if "area_m2" in struct_df.columns else None,
                    hover_data=[c for c in ["最寄駅：名称", "minutes"] if c in struct_df.columns],
                    title="築年数と坪単価（構造別）",
                    labels={"age": "築年数(年)", "tsubo_price": "坪単価(万円/坪)"}
                )
                st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("構造分析に必要な列（建物の構造/坪単価）が不足しています。")

        # ---- 駅徒歩×築年数 ヒートマップ ----
        st.subheader("🟥 条件別ヒートマップ（駅徒歩 × 築年数）")
        if (
            "minutes" in market_area.columns and "age" in market_area.columns and "tsubo_price" in market_area.columns
            and market_area["minutes"].notna().any() and market_area["age"].notna().any() and market_area["tsubo_price"].notna().any()
        ):
            tmp = market_area.dropna(subset=["minutes", "age", "tsubo_price"]).copy()

            tmp["walk_bin"] = pd.cut(tmp["minutes"], bins=[0, 5, 10, 15, 20, 100],
                                     labels=["～5分", "6～10分", "11～15分", "16～20分", "20分～"])
            tmp["age_bin"] = pd.cut(tmp["age"], bins=[0, 5, 10, 20, 30, 100],
                                    labels=["築浅(～5年)", "築10年以内", "築20年以内", "築30年以内", "築古(30年～)"])

            heat = tmp.groupby(["walk_bin", "age_bin"], observed=True)["tsubo_price"].mean().reset_index()
            mat = heat.pivot(index="walk_bin", columns="age_bin", values="tsubo_price")

            fig_h = px.imshow(mat, text_auto=".1f", aspect="auto", title="条件別の平均坪単価マトリクス")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.warning("ヒートマップに必要な列（最寄駅距離/建築年/価格/面積 等）が不足しています。")

    st.divider()

    # ---- 最後：統計の割合系（この町の輪郭） ----
    st.subheader("🏘️ 住宅・世帯構成（統計・割合系）")
    chart_cols = [
        "持ち家率",
        "借家率",
        "一戸建率",
        "共同住宅率",
        "単身・少人数世帯割合",
        "ファミリー世帯割合",
        "高齢化率",
    ]
    chart_cols = [c for c in chart_cols if c in row.index]

    df_chart = pd.DataFrame({"指標": chart_cols, "割合": [row.get(c, 0) for c in chart_cols]})
    fig = px.bar(df_chart, x="指標", y="割合", text="割合", title="割合系の一覧（統計・このエリア）")
    fig.update_traces(texttemplate="%{y:.1%}")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 2: 🔍 個別エリア比較
# ==========================================
with tab_compare:
    st.header("マルチエリア比較")

    comps = st.multiselect(
        "比較エリア",
        area_list,
        default=area_list[:2] if len(area_list) >= 2 else area_list,
        key="comp_multi",
    )

    if comps:
        display_cols = [
            "総人口",
            "世帯総数",
            "Median_Price_sqm",
            "持ち家率",
            "借家率",
            "一戸建率",
            "共同住宅率",
            "高齢化率",
            "単身・少人数世帯割合",
            "ファミリー世帯割合",
        ]
        display_cols = [c for c in display_cols if c in df_city.columns]

        st.markdown("##### 📋 数値比較")
        st.dataframe(df_city.loc[comps, display_cols].T.style.format("{:,.4f}"), use_container_width=True)

        st.markdown("##### 📊 グラフ比較")
        cm = st.selectbox("グラフ指標", numeric_cols, key="comp_metric")

        df_chart = df_city.loc[comps].reset_index()
        fig_comp = px.bar(df_chart, x="AREA_NAME", y=cm, text=cm, title=f"{cm} の比較", color="AREA_NAME")

        if "率" in cm or "割合" in cm:
            fig_comp.update_traces(texttemplate="%{y:.1%}")
        elif cm == "Median_Price_sqm":
            fig_comp.update_traces(texttemplate="¥ %{y:,.0f}")
        else:
            fig_comp.update_traces(texttemplate="%{y:,.0f}")

        st.plotly_chart(fig_comp, use_container_width=True)


# ==========================================
# TAB 3: 🆚 グループ対抗
# ==========================================
with tab_group:
    st.header("グループ対抗分析")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        group_a = st.multiselect("🔴 チームA", area_list, key="ga")
    with col_g2:
        group_b = st.multiselect("🔵 チームB", area_list, key="gb")

    if group_a and group_b:
        st.divider()

        def agg_grp(df, areas, label):
            sub = df.loc[areas]
            agg = {}

            for c in ["総人口", "世帯総数", "持ち家", "民営借家", "一戸建", "共同住宅"]:
                if c in df.columns:
                    agg[c] = sub[c].sum()

            w_col = "世帯総数" if "世帯総数" in df.columns else None
            for c in [col for col in df.columns if ("率" in col or "割合" in col)]:
                if w_col and sub[w_col].sum() > 0:
                    agg[c] = (sub[c] * sub[w_col]).sum() / sub[w_col].sum()
                else:
                    agg[c] = sub[c].mean()

            if "Median_Price_sqm" in df.columns:
                agg["Median_Price_sqm"] = sub["Median_Price_sqm"].mean()

            agg["Team"] = label
            return agg

        res_a = agg_grp(df_city, group_a, "チームA")
        res_b = agg_grp(df_city, group_b, "チームB")

        st.subheader("⚔️ 対決結果")
        c1, c2, c3, c4 = st.columns(4)

        v_a, v_b = res_a.get("総人口", 0), res_b.get("総人口", 0)
        c1.metric("総人口", f"{v_a:,.0f}", delta=f"{v_a - v_b:,.0f}")

        v_a, v_b = res_a.get("Median_Price_sqm", 0), res_b.get("Median_Price_sqm", 0)
        c2.metric("平均地価", f"¥ {v_a:,.0f}", delta=f"{v_a - v_b:,.0f}")

        v_a, v_b = res_a.get("持ち家率", 0), res_b.get("持ち家率", 0)
        c3.metric("持ち家率", f"{v_a:.1%}", delta=f"{v_a - v_b:.1%}")

        v_a, v_b = res_a.get("高齢化率", 0), res_b.get("高齢化率", 0)
        c4.metric("高齢者世帯率", f"{v_a:.1%}", delta=f"{v_a - v_b:.1%}")

        st.markdown("##### 📋 詳細比較")
        st.dataframe(pd.DataFrame([res_a, res_b]).set_index("Team").T.style.format("{:,.4f}"), use_container_width=True)

        st.markdown("##### 📊 グラフ比較")
        vm = st.selectbox("比較指標", numeric_cols, key="vs_metric")
        ch_data = pd.DataFrame(
            [{"Team": "チームA", "Value": res_a.get(vm, 0)}, {"Team": "チームB", "Value": res_b.get(vm, 0)}]
        )

        fig_vs = px.bar(ch_data, x="Team", y="Value", color="Team", text="Value", title=f"{vm} のチーム比較")
        if "率" in vm or "割合" in vm:
            fig_vs.update_traces(texttemplate="%{y:.1%}")
        elif vm == "Median_Price_sqm":
            fig_vs.update_traces(texttemplate="¥ %{y:,.0f}")
        else:
            fig_vs.update_traces(texttemplate="%{y:,.0f}")

        st.plotly_chart(fig_vs, use_container_width=True)
