import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import re
from pathlib import Path

# ==========================================
# 1. Streamlit 初期設定 (必ず最初に実行)
# ==========================================
st.set_page_config(layout="wide", page_title="不動産エリア分析ツール")

# ベースディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. Backend Logic (旧 back.py + 修正版)
# ==========================================

# --- 定数 ---
DEFAULT_CITY_LIST = ["川越市"]
TOWN_CHOME_HYOSYO_FULL = [2, 3, 4]

NAME_NORMALIZATION_MAP = {
    '人口総数': '総人口',
    '一般世帯数（世帯人員６人以上含む）': '一般世帯数',
    '世帯人員１人': '世帯人員1人',
    '世帯人員２人': '世帯人員2人',
    '世帯人員４人': '世帯人員4人',
    '一般世帯総数': '一般世帯総数_家族',
    '１８歳未満世帯員のいる一般世帯総数': '子育て世帯数(仮)',
    '６５歳以上世帯員のいる一般世帯総数': '高齢者世帯数',
    '総数': '世帯総数_経済',
    '住宅に住む一般世帯': '住宅世帯'
}

def data_path(filename: str) -> str:
    return str(BASE_DIR / filename)

# --- CSV読み込みユーティリティ ---
def read_csv_safe(file_path, skiprows=None):
    """バックエンド用：統計CSV読み込み（文字コード自動判定）"""
    encodings = ['utf-8', 'cp932', 'utf-8-sig']
    for enc in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=enc,
                skiprows=skiprows,
                dtype={"KEY_CODE": "string"}
            )
        except Exception:
            continue
    return pd.DataFrame()

def normalize_key_code_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.replace(r"\D", "", regex=True)
    return s

def filter_key_code_len(df: pd.DataFrame, allowed_len: int = 9) -> pd.DataFrame:
    if df.empty or "KEY_CODE" not in df.columns:
        return df
    df = df.copy()
    df["KEY_CODE"] = normalize_key_code_series(df["KEY_CODE"])
    return df[df["KEY_CODE"].str.len() == allowed_len].copy()

# --- コード対応表読み込み ---
def load_column_mapping():
    df = read_csv_safe(data_path('code_mapping.csv'))
    if df.empty or 'CODE' not in df.columns or 'NAME' not in df.columns:
        return {}
    return dict(zip(df['CODE'], df['NAME']))

# --- 市区町村一覧取得 ---
def get_available_cities(file_name='population.csv'):
    df = read_csv_safe(data_path(file_name))
    if df.empty:
        return []
    df = filter_key_code_len(df, allowed_len=9)
    if 'CITYNAME' in df.columns:
        return sorted(df['CITYNAME'].dropna().unique().tolist())
    return []

# --- 統計データ集計ロジック ---
def load_and_aggregate(file_name, mapping_dict, target_cities):
    df = read_csv_safe(data_path(file_name))
    if df.empty or 'CITYNAME' not in df.columns:
        return pd.DataFrame()

    # 9桁のみ（11桁=丁目を無視）
    df = filter_key_code_len(df, allowed_len=9)

    df = df[df['CITYNAME'].isin(target_cities)].copy()

    if 'HYOSYO' in df.columns:
        df = df[df['HYOSYO'].isin(TOWN_CHOME_HYOSYO_FULL)].copy()

    # 列名変換
    df = df.rename(columns=mapping_dict)
    df = df.rename(columns=NAME_NORMALIZATION_MAP)

    if 'NAME' in df.columns:
        df['AREA_NAME'] = df['NAME']
    else:
        df['AREA_NAME'] = df['KEY_CODE']

    # 不要列削除
    cols_to_drop = ['KEY_CODE', 'HYOSYO', 'CITYNAME', 'NAME', 'HTKSYORI', 'HTKSAKI', 'GASSAN']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 数値化
    cols_to_convert = [c for c in df.columns if c != 'AREA_NAME']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df_agg = df.groupby('AREA_NAME')[cols_to_convert].sum().reset_index()
    return df_agg

# --- 取引データのフィルタリング ---
def filter_price_types(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or "種類" not in price_df.columns:
        return price_df

    out = price_df.copy()
    s = out["種類"].astype(str)

    # 除外ワード
    exclude_keywords = ["農地", "林地", "山林", "池沼", "原野"]
    mask_exclude = s.str.contains("|".join(exclude_keywords), na=False)
    out = out[~mask_exclude].copy()

    # 優先ワード（ただし全滅するなら戻す処理のためkeep_keywords定義）
    keep_keywords = ["宅地", "土地", "中古マンション", "マンション"]
    mask_keep = out["種類"].astype(str).str.contains("|".join(keep_keywords), na=False)
    kept = out[mask_keep].copy()

    return kept if not kept.empty else out

# --- 住民プロフィール推定 ---
def add_resident_profile(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()

    if "総人口" not in df.columns:
        return df

    denom = df["総人口"].replace(0, 1)
    cols = list(df.columns)

    def find_cols(patterns):
        hit = []
        for c in cols:
            s = str(c)
            if any(re.search(p, s) for p in patterns):
                hit.append(c)
        return hit

    child_cols = find_cols([r"0[-〜]?14", r"14歳以下", r"年少", r"年少人口", r"15歳未満"])
    work_cols  = find_cols([r"15[-〜]?64", r"生産年齢", r"生産年齢人口", r"15歳以上64歳以下"])
    elder_cols = find_cols([r"65歳以上", r"老年", r"老年人口", r"高齢", r"高齢者"])

    df["子ども人口_推定"] = df[child_cols].sum(axis=1) if child_cols else 0
    df["現役人口_推定"]   = df[work_cols].sum(axis=1) if work_cols else 0
    df["高齢人口_推定"]   = df[elder_cols].sum(axis=1) if elder_cols else 0

    df["子ども率"] = df["子ども人口_推定"] / denom
    df["現役率"]   = df["現役人口_推定"] / denom
    df["高齢者率"] = df["高齢人口_推定"] / denom

    return df

# --- メインデータ取得関数 ---
def get_city_data(target_city_names=DEFAULT_CITY_LIST, uploaded_price_df=None):
    if isinstance(target_city_names, str):
        target_city_names = [target_city_names]

    mapping = load_column_mapping()

    # 統計データ読み込み
    df_pop   = load_and_aggregate('population.csv',        mapping, target_city_names)
    df_age   = load_and_aggregate('age.csv',               mapping, target_city_names)
    df_size  = load_and_aggregate('household_size.csv',    mapping, target_city_names)
    df_family= load_and_aggregate('family_type.csv',       mapping, target_city_names)
    df_eco   = load_and_aggregate('economic_status.csv',   mapping, target_city_names)
    df_owner = load_and_aggregate('housing_ownership.csv', mapping, target_city_names)
    df_struct= load_and_aggregate('housing_structure.csv', mapping, target_city_names)

    # 派生指標計算
    if not df_size.empty and '一般世帯数' in df_size.columns:
        hh = df_size['一般世帯数'].replace(0, 1)
        p1 = df_size.get('世帯人員1人', 0)
        p2 = df_size.get('世帯人員2人', 0)
        p4 = df_size.get('世帯人員4人', 0)
        df_size['単身・少人数世帯割合'] = (p1 + p2) / hh
        df_size['ファミリー世帯割合'] = p4 / hh

    if not df_family.empty and '一般世帯総数_家族' in df_family.columns:
        fam_hh = df_family['一般世帯総数_家族'].replace(0, 1)
        if '高齢者世帯数' in df_family.columns:
            df_family['高齢化率'] = df_family['高齢者世帯数'] / fam_hh

    if not df_owner.empty and '住宅世帯' in df_owner.columns:
        house_hh = df_owner['住宅世帯'].replace(0, 1)
        if '持ち家' in df_owner.columns:
            df_owner['持ち家率'] = df_owner['持ち家'] / house_hh
        if '民営借家' in df_owner.columns:
            df_owner['借家率'] = df_owner['民営借家'] / house_hh

    if not df_struct.empty and '主世帯数' in df_struct.columns:
        main_hh = df_struct['主世帯数'].replace(0, 1)
        if '一戸建' in df_struct.columns:
            df_struct['一戸建率'] = df_struct['一戸建'] / main_hh
        if '共同住宅' in df_struct.columns:
            df_struct['共同住宅率'] = df_struct['共同住宅'] / main_hh

    # データ結合
    dfs = [d for d in [df_pop, df_age, df_size, df_family, df_eco, df_owner, df_struct] if not d.empty]
    
    if not dfs:
        # 統計データがない場合でも、地価データがあれば処理を続行するために空DataFrame作成
        merged_df = pd.DataFrame(columns=['AREA_NAME'])
    else:
        merged_df = dfs[0]
        for d in dfs[1:]:
            merged_df = pd.merge(merged_df, d, on='AREA_NAME', how='outer')

    if not merged_df.empty:
        merged_df = merged_df.set_index('AREA_NAME').fillna(0)
        merged_df.index.name = "AREA_NAME"

    if '総人口' in merged_df.columns and '世帯総数' in merged_df.columns:
        merged_df['1世帯当たり人員'] = merged_df['総人口'] / merged_df['世帯総数'].replace(0, 1)

    # 住民プロフィール追加
    merged_df = add_resident_profile(merged_df)

    # ---- 地価（取引）データ統合 ----
    price_df = uploaded_price_df.copy() if uploaded_price_df is not None else pd.DataFrame()

    if not price_df.empty:
        # 市区町村で絞る
        if '市区町村名' in price_df.columns:
            price_df = price_df[price_df['市区町村名'].isin(target_city_names)].copy()

        # 農地/林地等を除外
        price_df = filter_price_types(price_df)

        if '地区名' in price_df.columns:
            # --- ★ 修正箇所ここから：単価の自動計算 ---
            
            # 面積の数値化（「2000㎡以上」などの文字列対応）
            def _clean_area_local(x):
                try:
                    s = str(x).replace(",", "").replace("㎡以上", "").replace("m^2", "").replace("m2", "")
                    nums = re.findall(r"[\d.]+", s)
                    return float(nums[0]) if nums else None
                except:
                    return None
            
            area_col = '面積（㎡）' if '面積（㎡）' in price_df.columns else None
            price_col = '取引価格（総額）' if '取引価格（総額）' in price_df.columns else None

            # 計算用の一時列作成
            if area_col:
                price_df['area_calc'] = price_df[area_col].apply(_clean_area_local)
            else:
                price_df['area_calc'] = None
            
            if price_col:
                price_df['total_price'] = pd.to_numeric(price_df[price_col], errors='coerce')
            else:
                price_df['total_price'] = None

            # 単価計算（総額 / 面積）
            price_df['calc_unit_price'] = price_df['total_price'] / price_df['area_calc'].replace(0, np.nan)

            # 元々の「取引価格（㎡単価）」があれば読み込む
            if '取引価格（㎡単価）' in price_df.columns:
                price_df['orig_unit_price'] = pd.to_numeric(price_df['取引価格（㎡単価）'], errors='coerce')
                # 元の単価があれば使い、なければ計算値で埋める
                price_df['㎡単価'] = price_df['orig_unit_price'].fillna(price_df['calc_unit_price'])
            else:
                # 元の列がない場合は計算値を採用
                price_df['㎡単価'] = price_df['calc_unit_price']
            
            # --- ★ 修正箇所ここまで ---

            # 有効な単価と地区名があるデータのみ残す
            price_df = price_df.dropna(subset=['㎡単価', '地区名']).copy()

            if not price_df.empty:
                price_agg = price_df.groupby('地区名')['㎡単価'].median().reset_index()
                price_agg = price_agg.rename(columns={'地区名': 'AREA_NAME', '㎡単価': 'Median_Price_sqm'})

                # 統計データがない場合(merged_dfが空)の考慮
                if merged_df.empty:
                    merged_df = price_agg.set_index('AREA_NAME').fillna(0)
                else:
                    merged_df = merged_df.reset_index().merge(price_agg, on='AREA_NAME', how='left').set_index('AREA_NAME').fillna(0)
                
                merged_df.index.name = "AREA_NAME"
            else:
                if not merged_df.empty: merged_df['Median_Price_sqm'] = 0
        else:
            if not merged_df.empty: merged_df['Median_Price_sqm'] = 0
    else:
        if not merged_df.empty: merged_df['Median_Price_sqm'] = 0

    # サマリー作成
    city_summary = merged_df.mean(numeric_only=True).to_dict()

    return merged_df, city_summary


# ==========================================
# 3. Frontend Helper Functions (旧 app.py の関数)
# ==========================================

def read_csv_flexible(file_or_path, is_path: bool = False) -> pd.DataFrame:
    """フロントエンド用：ファイルアップロードオブジェクト対応のCSV読み込み"""
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

def preprocess_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """攻略ガイド表示用に取引データを加工"""
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
            nums = re.findall(r"[\d.]+", s)
            return float(nums[0]) if nums else None
        except Exception:
            return None

    if "面積（㎡）" in d.columns:
        d["area_m2"] = d["面積（㎡）"].apply(clean_area)
    else:
        d["area_m2"] = None

    # 坪単価（万円/坪）
    if "取引価格（総額）" in d.columns and "area_m2" in d.columns:
        total = pd.to_numeric(d["取引価格（総額）"], errors="coerce")
        tsubo = d["area_m2"] / 3.30578
        d["tsubo_price"] = (total / tsubo) / 10000
        d["tsubo_price"] = d["tsubo_price"].round(1)
    else:
        d["tsubo_price"] = None

    # 取引時期
    if "取引時期" in d.columns:
        d["period"] = d["取引時期"].astype(str).str.replace("年第", "-Q", regex=False).str.replace("四半期", "", regex=False)
    else:
        d["period"] = None

    # 駅徒歩
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

    # 築年数
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


# ==========================================
# 4. UI Logic (Streamlit Main App)
# ==========================================

# --- サイドバー：分析設定 ---
st.sidebar.title("🛠️ 分析設定")

available_cities = get_available_cities()
if not available_cities:
    # 統計データがない場合でも地価データだけで動くようにデフォルトを設定
    available_cities = ["川越市"]
    default_cities = ["川越市"]
else:
    default_cities = [available_cities[34]]

target_cities = st.sidebar.multiselect(
    "分析する市区町村を選択",
    options=available_cities,
    default=default_cities,
    help="複数の市を選ぶと、それら全てのエリアを横断して分析・比較できます。"
)

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
    default_test_path = str(BASE_DIR / "test.csv")
    uploaded_price_df = read_csv_flexible(default_test_path, is_path=True)
    if uploaded_price_df.empty:
        st.sidebar.warning("📄 test.csv を読み込めませんでした（存在チェック）")
        st.sidebar.caption(f"path: {default_test_path}")
    else:
        st.sidebar.info("📄 test.csv を使用しています（デフォルト）")
        st.sidebar.caption(f"行数: {len(uploaded_price_df):,}")

price_df_pre = preprocess_price_df(uploaded_price_df)

# --- データロード (キャッシュ使用) ---
@st.cache_data
def load_data(cities, price_df):
    # citiesが空でもprice_dfがあれば動くように緩和
    if not cities and price_df.empty:
        return pd.DataFrame(), {}
    return get_city_data(target_city_names=cities, uploaded_price_df=price_df)

if not target_cities and uploaded_price_df.empty:
    st.warning("左のサイドバーから、分析したい市区町村を選んでください。")
    st.stop()

df_city, city_summary = load_data(target_cities, uploaded_price_df)

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
# TAB 1: 🔰 攻略ガイド
# ==========================================
with tab_guide:
    st.header("エリア攻略ガイド（ぱっと見＋市場データ）")

    default_area = "新富町"
    default_index = area_list.index(default_area) if default_area in area_list else 0

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
        delta = (value or 0) - (avg or 0)
        if is_percent:
            st.metric(label, f"{(value or 0):.1%}", delta=f"{delta:.1%}")
        else:
            st.metric(label, f"{(value or 0):,.0f}", delta=f"{delta:,.0f}")

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1: metric_vs_avg("持ち家率", row.get("持ち家率", 0), city_summary.get("持ち家率", 0), True)
    with d2: metric_vs_avg("借家率", row.get("借家率", 0), city_summary.get("借家率", 0), True)
    with d3: metric_vs_avg("一戸建率", row.get("一戸建率", 0), city_summary.get("一戸建率", 0), True)
    with d4: metric_vs_avg("共同住宅率", row.get("共同住宅率", 0), city_summary.get("共同住宅率", 0), True)
    with d5: metric_vs_avg("単身・少人数", row.get("単身・少人数世帯割合", 0), city_summary.get("単身・少人数世帯割合", 0), True)
    with d6: metric_vs_avg("ファミリー", row.get("ファミリー世帯割合", 0), city_summary.get("ファミリー世帯割合", 0), True)

    st.markdown("##### 📊 ポジション差分（平均との差：%ポイント）")
    pos_items = [
        ("持ち家率", "持ち家率"),
        ("借家率", "借家率"),
        ("一戸建率", "一戸建率"),
        ("共同住宅率", "共同住宅率"),
        ("単身・少人数世帯割合", "単身・少人数"),
        ("ファミリー世帯割合", "ファミリー"),
    ]
    rows_ = []
    for key, label in pos_items:
        area_val = float(row.get(key, 0) or 0)
        avg_val = float(city_summary.get(key, 0) or 0)
        rows_.append({
            "指標": label,
            "平均との差(ポイント)": (area_val - avg_val) * 100,
        })
    pos_df = pd.DataFrame(rows_)
    fig_pos = px.bar(pos_df, x="指標", y="平均との差(ポイント)", text="平均との差(ポイント)", title="市平均との差（＋なら平均より高い）")
    fig_pos.update_traces(texttemplate="%{text:.1f}pt")
    st.plotly_chart(fig_pos, use_container_width=True)

    st.divider()

    # ---- ★ 住民プロフィール（市場サマリーの上に追加） ----
    st.subheader("👥 住民プロフィール（どんな人が住んでる？）")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("子ども率", f"{row.get('子ども率', 0):.1%}")
    p2.metric("現役率", f"{row.get('現役率', 0):.1%}")
    p3.metric("高齢者率", f"{row.get('高齢者率', 0):.1%}")
    p4.metric("1世帯あたり人員", f"{row.get('1世帯当たり人員', 0):.2f}")

    prof_df = pd.DataFrame({
        "区分": ["子ども", "現役", "高齢者"],
        "割合": [
            float(row.get("子ども率", 0) or 0),
            float(row.get("現役率", 0) or 0),
            float(row.get("高齢者率", 0) or 0),
        ]
    })
    fig_prof = px.bar(prof_df, x="区分", y="割合", text="割合", title="年齢構成（推定）")
    fig_prof.update_traces(texttemplate="%{y:.1%}")
    st.plotly_chart(fig_prof, use_container_width=True)

    st.divider()

    # ---- ② フロー（市場/取引） ----
    st.subheader("💰 市場サマリー（取引データ）")

    market = price_df_pre.copy()

    if not market.empty and "市区町村名" in market.columns:
        market = market[market["市区町村名"].isin(target_cities)].copy()

    if not market.empty and "地区名" in market.columns:
        market_area = market[market["地区名"] == selected_area].copy()
    else:
        market_area = pd.DataFrame()

    if market_area.empty:
        st.info("この担当エリアに一致する取引データ（地区名）が見つかりませんでした。")
    else:
        m1, m2, m3, m4 = st.columns(4)

        if "price_man" in market_area.columns and market_area["price_man"].notna().any():
            m1.metric("平均取引価格", f"{market_area['price_man'].mean():,.0f} 万円")
        else:
            m1.metric("平均取引価格", "—")

        if "tsubo_price" in market_area.columns and market_area["tsubo_price"].notna().any():
            m2.metric("平均坪単価", f"{market_area['tsubo_price'].mean():,.1f} 万円/坪")
        else:
            m2.metric("平均坪単価", "—")

        if "age" in market_area.columns and market_area["age"].notna().any():
            m3.metric("平均築年数", f"{market_area['age'].mean():.1f} 年")
        else:
            m3.metric("平均築年数", "—")

        m4.metric("データ件数", f"{len(market_area):,} 件")

        st.subheader("📈 相場トレンド（時系列）")
        if "period" in market_area.columns and "tsubo_price" in market_area.columns and market_area["tsubo_price"].notna().any():
            trend = market_area.groupby("period")["tsubo_price"].mean().reset_index()
            fig_tr = px.line(trend, x="period", y="tsubo_price", markers=True, title="時期ごとの平均坪単価推移")
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.warning("時系列表示に必要な列（取引時期/坪単価）が不足しています。")

        st.subheader("📊 価格帯のボリュームゾーン")
        if "price_man" in market_area.columns and market_area["price_man"].notna().any():
            fig_hist = px.histogram(market_area, x="price_man", nbins=20, title="価格帯ごとの取引件数")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("価格帯分布に必要な列（取引価格（総額））が不足しています。")

        st.subheader("🏗️ 建物構造（シェア＆価格レンジ）")
        if "建物の構造" in market_area.columns and market_area["建物の構造"].notna().any() and "tsubo_price" in market_area.columns:
            struct_df = market_area.dropna(subset=["建物の構造"]).copy()
            s1, s2 = st.columns(2)
            with s1:
                fig_pie = px.pie(struct_df, names="建物の構造", title="構造割合（市場シェア）")
                st.plotly_chart(fig_pie, use_container_width=True)
            with s2:
                fig_box = px.box(
                    struct_df,
                    x="建物の構造",
                    y="tsubo_price",
                    color="建物の構造",
                    title="構造別 坪単価レンジ（箱ひげ）",
                    labels={"tsubo_price": "坪単価(万円/坪)"}
                )
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

    st.subheader("🏘️ 住宅・世帯構成（統計・割合系）")
    chart_cols = [
        "持ち家率",
        "借家率",
        "一戸建率",
        "共同住宅率",
        "単身・少人数世帯割合",
        "ファミリー世帯割合",
        "高齢化率",
        "子ども率",
        "現役率",
        "高齢者率",
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
            "子ども率",
            "現役率",
            "高齢者率",
        ]
        display_cols = [c for c in display_cols if c in df_city.columns]

        st.markdown("##### 📋 数値比較")
        st.dataframe(df_city.loc[comps, display_cols].T.style.format("{:,.4f}"), use_container_width=True)

        st.markdown("##### 📊 グラフ比較")
        cm = st.selectbox("グラフ指標", numeric_cols, key="comp_metric")

        df_tmp = df_city.loc[comps].reset_index()  # index名は "AREA_NAME" に設定済み
        fig_comp = px.bar(df_tmp, x="AREA_NAME", y=cm, text=cm, title=f"{cm} の比較", color="AREA_NAME")

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