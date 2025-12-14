import pandas as pd
import numpy as np
import os

# --- 定数 ---
DEFAULT_CITY_LIST = ["川越市"]
TOWN_CHOME_HYOSYO_FULL = [2, 3, 4] 

# --- 変換マップ ---
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

def read_csv_safe(file_name, skiprows=None):
    """エンコーディングを自動判別して読み込む"""
    encodings = ['utf-8', 'cp932', 'utf_8_sig']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_name, encoding=encoding, skiprows=skiprows)
            return df
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return pd.DataFrame()

def load_column_mapping():
    """コード対応表を読み込み辞書化"""
    df = read_csv_safe('コード対応表.csv')
    if df.empty or 'CODE' not in df.columns:
        return {}
    return dict(zip(df['CODE'], df['NAME']))

def get_available_cities(file_name='population.csv'):
    """CSVに含まれる市区町村のリストを取得する"""
    df = read_csv_safe(file_name)
    if df.empty: return []
    
    # ヘッダー処理
    if 'KEY_CODE' in df.columns:
        first_val = df.iloc[0]['KEY_CODE']
        is_numeric_code = str(first_val).replace('.', '').isdigit()
        if pd.isna(first_val) or str(first_val).strip() == '' or not is_numeric_code:
            df = df.iloc[1:].reset_index(drop=True)
            
    if 'CITYNAME' in df.columns:
        return sorted(df['CITYNAME'].dropna().unique().tolist())
    return []

def load_and_aggregate(file_name, mapping_dict, target_cities):
    """CSVを読み込み、指定された複数の都市で絞り込み、合算する"""
    
    df = read_csv_safe(file_name)
    
    if df.empty or 'CITYNAME' not in df.columns:
        return pd.DataFrame()

    # ヘッダー行判定
    if not df.empty and 'KEY_CODE' in df.columns:
        first_val = df.iloc[0]['KEY_CODE']
        is_numeric_code = str(first_val).replace('.', '').isdigit()
        if pd.isna(first_val) or str(first_val).strip() == '' or not is_numeric_code:
            df = df.iloc[1:].reset_index(drop=True)

    # ★複数都市でフィルタリング★
    df = df[df['CITYNAME'].isin(target_cities)].copy()
    df = df[df['HYOSYO'].isin(TOWN_CHOME_HYOSYO_FULL)].copy()
    
    df = df.rename(columns=mapping_dict)
    df = df.rename(columns=NAME_NORMALIZATION_MAP)
    df['AREA_NAME'] = df['NAME']
    
    cols_to_drop = ['KEY_CODE', 'HYOSYO', 'CITYNAME', 'NAME', 'HTKSYORI', 'HTKSAKI', 'GASSAN']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    cols_to_convert = [c for c in df.columns if c != 'AREA_NAME']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 町丁名でグループ化（異なる市の同じ町名は合算されてしまうが、通常重複しない前提）
    # もし市をまたいで同じ町名がある場合は 'CITYNAME' も含めてgroupする必要があるが、
    # 今回はシンプルにAREA_NAMEで合算
    df_agg = df.groupby('AREA_NAME')[cols_to_convert].sum().reset_index()
    return df_agg

def get_city_data(target_city_names=DEFAULT_CITY_LIST, uploaded_price_df=None):
    """
    メイン関数：指定された都市（リスト）のデータを生成する
    """
    if isinstance(target_city_names, str):
        target_city_names = [target_city_names]
        
    mapping = load_column_mapping()
    
    # 1. 統計データの読み込み
    df_pop = load_and_aggregate('population.csv', mapping, target_city_names)
    if not df_pop.empty and '総人口' in df_pop.columns:
        df_pop['総人口'] = df_pop['総人口'].replace(0, 1)

    # age.csv は補足データとして扱う（高齢化率は世帯ベースを使うため）
    df_age = load_and_aggregate('age.csv', mapping, target_city_names)
    
    df_size = load_and_aggregate('household_size.csv', mapping, target_city_names)
    if not df_size.empty and '一般世帯数' in df_size.columns:
        hh = df_size['一般世帯数'].replace(0, 1)
        p1 = df_size.get('世帯人員1人', 0)
        p2 = df_size.get('世帯人員2人', 0)
        p4 = df_size.get('世帯人員4人', 0)
        df_size['単身・少人数世帯割合'] = (p1 + p2) / hh
        df_size['ファミリー世帯割合'] = p4 / hh

    df_family = load_and_aggregate('family_type.csv', mapping, target_city_names)
    if not df_family.empty and '一般世帯総数_家族' in df_family.columns:
        fam_hh = df_family['一般世帯総数_家族'].replace(0, 1)
        if '核家族世帯' in df_family.columns: df_family['核家族率'] = df_family['核家族世帯'] / fam_hh
        if '子育て世帯数(仮)' in df_family.columns: df_family['子育て世帯率'] = df_family['子育て世帯数(仮)'] / fam_hh
        
        # ★【変更】高齢化率を「世帯ベース」で計算★
        if '高齢者世帯数' in df_family.columns: 
            df_family['高齢化率'] = df_family['高齢者世帯数'] / fam_hh

    df_eco = load_and_aggregate('economic_status.csv', mapping, target_city_names)
    if not df_eco.empty and '世帯総数_経済' in df_eco.columns:
        df_eco['非就業者世帯率'] = df_eco.get('非就業者世帯', 0) / df_eco['世帯総数_経済'].replace(0, 1)

    df_owner = load_and_aggregate('housing_ownership.csv', mapping, target_city_names)
    if not df_owner.empty and '住宅世帯' in df_owner.columns:
        house_hh = df_owner['住宅世帯'].replace(0, 1)
        if '持ち家' in df_owner.columns: df_owner['持ち家率'] = df_owner['持ち家'] / house_hh
        if '民営借家' in df_owner.columns: df_owner['借家率'] = df_owner['民営借家'] / house_hh

    df_struct = load_and_aggregate('housing_structure.csv', mapping, target_city_names)
    if not df_struct.empty and '主世帯数' in df_struct.columns:
        main_hh = df_struct['主世帯数'].replace(0, 1)
        if '一戸建' in df_struct.columns: df_struct['一戸建率'] = df_struct['一戸建'] / main_hh
        if '共同住宅' in df_struct.columns: df_struct['共同住宅率'] = df_struct['共同住宅'] / main_hh

    # --- 統合 ---
    dfs = [d for d in [df_pop, df_age, df_size, df_family, df_eco, df_owner, df_struct] if not d.empty]
    if not dfs: return pd.DataFrame(), {}
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='AREA_NAME', how='outer')
    
    merged_df = merged_df.set_index('AREA_NAME').fillna(0)
    
    # 最終的な高齢化率カラムの調整（もし計算できていなければ0）
    if '高齢化率' not in merged_df.columns:
        merged_df['高齢化率'] = 0

    if '総人口' in merged_df.columns and '世帯総数' in merged_df.columns:
        merged_df['1世帯当たり人員'] = merged_df['総人口'] / merged_df['世帯総数'].replace(0, 1)

    # --- 地価データ処理 ---
    price_df = pd.DataFrame()
    if uploaded_price_df is not None:
        price_df = uploaded_price_df
    else:
        price_df = read_csv_safe('real_estate_tx.csv')

    kawagoe_summary = merged_df.mean().to_dict()

    if not price_df.empty:
        if '市区町村名' in price_df.columns:
             # 地価データも指定された複数の市町村で絞る
             price_df = price_df[price_df['市区町村名'].isin(target_city_names)].copy()
        
        if '取引価格（㎡単価）' in price_df.columns and '地区名' in price_df.columns:
            price_df['㎡単価'] = pd.to_numeric(price_df['取引価格（㎡単価）'], errors='coerce')
            if '種類' in price_df.columns:
                price_df = price_df[price_df['種類'].isin(['宅地(土地と建物)', '土地'])]
            
            price_agg = price_df.groupby('地区名')['㎡単価'].median().reset_index()
            price_agg = price_agg.rename(columns={'地区名': 'AREA_NAME', '㎡単価': 'Median_Price_sqm'})
            
            merged_df = merged_df.merge(price_agg, on='AREA_NAME', how='left').set_index('AREA_NAME').fillna(0)
            
            if 'Median_Price_sqm' in price_agg.columns:
                kawagoe_summary['Median_Price_sqm'] = price_agg['Median_Price_sqm'].median()
        else:
            merged_df['Median_Price_sqm'] = 0
            kawagoe_summary['Median_Price_sqm'] = 0
    else:
        merged_df['Median_Price_sqm'] = 0
        kawagoe_summary['Median_Price_sqm'] = 0

    return merged_df, kawagoe_summary