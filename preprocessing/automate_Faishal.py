import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import argparse

binary_cat = {'No': 0, 'Yes': 1}
target_cat = {"Low cost": 0, "Medium cost": 1,
              "High cost": 2, "Very High cost": 3}

encoders = {'Binary_Cat': binary_cat,
            'Target_Cat': target_cat}

# Fungsi loading data
def load_data(root_path: str,
              filename: str = 'Mobile.csv') -> pd.DataFrame:
    data_path = os.path.join(root_path, filename)
    df = pd.read_csv(data_path)

    return df

def cleaning_data(df: pd.DataFrame, 
                  encoders: dict) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include='object')
    col_with_digits = []

    def digit_cleaning(obj_cols: pd.DataFrame = obj_cols,
                       col_with_digits: list = col_with_digits,
                       df: pd.DataFrame = df) -> pd.DataFrame:
        for col in obj_cols:
            first_row = df[col].iloc[0]
            if re.search(r'\d+', first_row):
                # Simpan nama col
                col_with_digits.append(col)
                # Maka menyisakan nilai numerik itu, dan hapus teks non numerik
                df[col] = df[col].str.replace(r'\D+', '', regex=True)
            else:
                continue
        return df
    
    def na_detection(total_na: pd.Series,
                     df: pd.DataFrame) -> pd.DataFrame:
        if len(total_na) > 0:
            df = df.dropna(axis=0)
        return df
    
    def dups_detection(total_dups: pd.Series,
                       df: pd.DataFrame) -> pd.DataFrame:
        if total_dups > 0:
            df = df.drop_duplicates(axis=0)

        return df

    def encoding_obj_feats(remain_obj_cols: pd.DataFrame,
                           target_col: str,
                           encoders: dict = encoders) -> pd.DataFrame:
        for_encode_cols = remain_obj_cols.columns
    
        for col in for_encode_cols:
            scaler = encoders['Binary_Cat'] if col != target_col else encoders['Target_Cat']
            df[col] = df[col].map(scaler)

        return df

    df = digit_cleaning()
    # Memastikan tipe data
    df[col_with_digits] = df[col_with_digits].astype('float64')

    # Hapus missing values
    total_na = df.isna().sum()
    df = na_detection(total_na, df)
    # Hapus duplicated data
    total_dups = df.duplicated().sum()
    df = dups_detection(total_dups, df)
    
    # Encoding fitur objek
    remain_obj_cols = df.select_dtypes(include='object')
    target_col = 'price_range'
    df = encoding_obj_feats(remain_obj_cols, target_col, encoders)

    return df

def standarization(scaler: StandardScaler,
                   cols: list,
                   df: pd.DataFrame,
                   ) -> pd.DataFrame:
    df[cols] = scaler.fit_transform(df[cols])
    return df

def remove_outliers_iqr(data: pd.DataFrame, 
                        cols: list) -> pd.DataFrame:
    cleaned_data = data.copy()
    for col in cols:
        Q1 = cleaned_data[col].quantile(0.25)
        Q3 = cleaned_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        cleaned_data = cleaned_data[(cleaned_data[col] >= lower) & (cleaned_data[col] <= upper)]
    return cleaned_data

def lda_dim_reduction(X: pd.Series, 
                       y: pd.Series,
                       n_comp: int = 3) -> pd.DataFrame:
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_lda = lda.fit_transform(X, y)

    return X_lda

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automating Data Preprocessing")
    parser.add_argument("--datapath", type=str, default="../data", 
                        help="Masukkan relative path dataset", required=True)
    parser.add_argument("--ncomps", type=int, default=3,
                        help="Masukkan jumlah komponen LDA")
    args = parser.parse_args()

    ROOT_PATH = args.datapath
    n_comps = args.ncomps

    # Load data
    df = load_data(root_path=ROOT_PATH)

    # Cleaning data
    cleaned_df = cleaning_data(df=df, encoders=encoders)

    # Standarization
    scaler = StandardScaler()
    col_to_scale = ['Ram_mb', 'Battery_power_mAh', 'Pixel_width', 'px_height']
    cleaned_df = standarization(scaler=scaler, cols=col_to_scale,
                                df=cleaned_df)

    # Remove Outliers
    num_cols = cleaned_df.select_dtypes(exclude='object').columns
    cleaned_df = remove_outliers_iqr(data=cleaned_df,
                                     cols=num_cols)

    # Dimentional Reduction (LDA)
    target_col = 'price_range'
    X = cleaned_df.drop(columns=[target_col])
    y = cleaned_df[target_col]

    X_lda = lda_dim_reduction(X, y, n_comp=n_comps)

    output_path = os.path.join(os.path.dirname(__file__), "cleaned_data.csv")
    X_lda.to_csv(output_path, index=False)