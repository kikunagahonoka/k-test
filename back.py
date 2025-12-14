import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================
# パス
# =========================
BASE_DIR = Path(__file__).resolve().parent

def data_path(filename: str) -> str:
    return str(BASE_DIR / filename)

# =========================
# 定数
# =========================
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

# =========================
# CSV 読み込み（KEY_CODEは文字列）
# =========================
def read_csv_safe(file_path, skiprows=None):
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

# =========================
# コード対応表
# =========================
def load_column_mapping():
    # ※英数字ファイル名推奨（本番で日本語ファイル名が事故りやすい）
    df = read_csv_safe(data_path('code_mapping.csv'))
    if df.empty or 'CODE' not in df.columns or 'NAME' not in df.columns:
        return {}
    return dict(zip(df['CODE'], df['NAME']))

# =========================
# 市区町村一覧
# =========================
def get_available_cities(file_name='population.csv'):
    df = read_csv_safe(data_path(file_name))
    if df.empty:
        return []
    df = filter_key_code_len(df, allowed_len=9)
    if 'CITYNAME' in df.columns:
        return sorted(df['CITYNAME'].dropna().unique().tolist())
    return []

# =========================
# 統計CSV読み込み→集計
# =========================
def load_and_aggregate(file_name, mapping_dict, target_cities):
    df = read_csv_safe(data_path(file_name))
    if df.empty or 'CITYNAME' not in df.columns:
        return pd.DataFrame()

    # ★ 9桁のみ（11桁=丁目を無視）
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

# =========================
# ★ 取引データの「種類」フィルタ（農地・林地/山林・池沼など除外）
# =========================
def filter_price_types(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or "種類" not in price_df.columns:
        return price_df

    out = price_df.copy()
    s = out["種類"].astype(str)

    # 除外したい（ユーザー要望）
    exclude_keywords = ["農地", "林地", "山林", "池沼", "原野"]
    mask_exclude = s.str.contains("|".join(exclude_keywords), na=False)
    out = out[~mask_exclude].copy()

    # 「宅地/土地」系を優先（ただし全滅するなら戻す）
    keep_keywords = ["宅地", "土地", "中古マンション", "マンション"]
    mask_keep = out["種類"].astype(str).str.contains("|".join(keep_keywords), na=False)
    kept = out[mask_keep].copy()

    return kept if not kept.empty else out

# =========================
# ★ 住民プロフィール（age.csv）から推定列を追加
# =========================
def add_resident_profile(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    age.csv 由来の列から「どんな人が住んでるか」指標を推定して追加。
    列名が多少違っても拾えるようにパターンで抽出。
    """
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

# =========================
# メイン
# =========================
def get_city_data(target_city_names=DEFAULT_CITY_LIST, uploaded_price_df=None):
    if isinstance(target_city_names, str):
        target_city_names = [target_city_names]

    mapping = load_column_mapping()

    # 統計データ
    df_pop   = load_and_aggregate('population.csv',        mapping, target_city_names)
    df_age   = load_and_aggregate('age.csv',               mapping, target_city_names)
    df_size  = load_and_aggregate('household_size.csv',    mapping, target_city_names)
    df_family= load_and_aggregate('family_type.csv',       mapping, target_city_names)
    df_eco   = load_and_aggregate('economic_status.csv',   mapping, target_city_names)
    df_owner = load_and_aggregate('housing_ownership.csv', mapping, target_city_names)
    df_struct= load_and_aggregate('housing_structure.csv', mapping, target_city_names)

    # 派生指標
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

    # 統合
    dfs = [d for d in [df_pop, df_age, df_size, df_family, df_eco, df_owner, df_struct] if not d.empty]
    if not dfs:
        return pd.DataFrame(), {}

    merged_df = dfs[0]
    for d in dfs[1:]:
        merged_df = pd.merge(merged_df, d, on='AREA_NAME', how='outer')

    merged_df = merged_df.set_index('AREA_NAME').fillna(0)
    merged_df.index.name = "AREA_NAME"

    if '総人口' in merged_df.columns and '世帯総数' in merged_df.columns:
        merged_df['1世帯当たり人員'] = merged_df['総人口'] / merged_df['世帯総数'].replace(0, 1)

    # 住民プロフィール（age.csv 推定）
    merged_df = add_resident_profile(merged_df)

    # ---- 地価（取引）処理 ----
    price_df = uploaded_price_df.copy() if uploaded_price_df is not None else pd.DataFrame()

    if not price_df.empty:
        # 市区町村で絞る
        if '市区町村名' in price_df.columns:
            price_df = price_df[price_df['市区町村名'].isin(target_city_names)].copy()

        # 農地/林地/山林/池沼/原野 を除外（＋宅地/土地優先）
        price_df = filter_price_types(price_df)

        if '取引価格（㎡単価）' in price_df.columns and '地区名' in price_df.columns:
            price_df['㎡単価'] = pd.to_numeric(price_df['取引価格（㎡単価）'], errors='coerce')
            price_df = price_df.dropna(subset=['㎡単価', '地区名']).copy()

            price_agg = price_df.groupby('地区名')['㎡単価'].median().reset_index()
            price_agg = price_agg.rename(columns={'地区名': 'AREA_NAME', '㎡単価': 'Median_Price_sqm'})

            merged_df = merged_df.reset_index().merge(price_agg, on='AREA_NAME', how='left').set_index('AREA_NAME').fillna(0)
            merged_df.index.name = "AREA_NAME"
        else:
            merged_df['Median_Price_sqm'] = 0
    else:
        merged_df['Median_Price_sqm'] = 0

    # ★ city_summary は「全部追加し終えたあと」に作る（ここ重要）
    city_summary = merged_df.mean(numeric_only=True).to_dict()

    return merged_df, city_summary
