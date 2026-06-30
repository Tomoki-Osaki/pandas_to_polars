この記事に掲載しているコードは, 以下のリンク先からnotebook形式のファイルとしてダウンロードできます。
https://github.com/Tomoki-Osaki/pandas_to_polars

## この記事の特長
pandasからpolarsへの乗り換えがスムーズにできるように, pandasでの処理をpolarsで実現するための関数の対応をまとめました。

なお, この記事ではあくまでpandasとpolarsの対応コードに焦点を当てているため, pandasではできないpolarsのすごい機能については触れませんし, polarsの特徴についても最低限しか触れません。そちらは参考記事セクションに載せている他記事をご覧ください。

polarsは既に参考になる記事が多くありますが, 一応この記事の差別化点は, pandasでの処理をpolarsで再現するということを重点に置き, 両者のコードと処理結果の併記を徹底していることです (自分がpolarsに乗り換えるときに意外に少なかったので)。
また, 実データを想定したダミーデータを用意することで, 実際にpolarsを使うイメージが掴みやすくなっているかと思います。別途csvをダウンロードする必要もありません。
```python
import numpy as np
import pandas as pd
import polars as pl
```

```python
print("pandas version: ", pd.__version__)
print("polars version: ", pl.__version__)
# [out]
# pandas version:  3.0.3
# polars version:  1.41.2
```


## ダミーデータの作成
今回は顧客の購買行動のダミーデータを利用します。(ダミーデータを生成するコードはGeminiに書いてもらいました)
この関数は, write_csvにパスを指定してcsvに書き出すか, そうでなければpandasやpolarsのDataFrame関数に渡すことのできる辞書型データを返します。
データ列は, 顧客番号, 契約年月日, 出身地, 年齢, 性別, 年齢, 購買金額, 購買年月日があり, 欠損値がランダムに含まれるようにしています。

```python
def generate_dict_for_dataframe(n_rows=10000, seed=42, write_csv=None):
    """
    データフレーム作成用のダミーデータの辞書を返す。
    write_csvにパスを指定すると、そのパスにcsvとして出力する。
    """
    num_unique_customers = 2000
    
    # 乱数シードの固定
    np.random.seed(42)
    
    # 一般的な地域分類のリスト
    regions = ["北海道", "東北", "関東", "中部", "近畿", "中国", "四国", "九州"]
    
    # 顧客マスタデータの作成（ユニーク2000人分）
    unique_customer_ids = [f"C{i:04d}" for i in range(1, num_unique_customers + 1)]
    
    # 各顧客の一意な属性を生成
    genders_master = np.random.choice([1, 2], size=num_unique_customers)
    ages_master = np.random.randint(18, 81, size=num_unique_customers)
    regions_master = np.random.choice(regions, size=num_unique_customers)  # 地域分類マスタ
    
    # 契約年月日の生成
    start_contract = pd.to_datetime("2020-01-01")
    end_contract = pd.to_datetime("2024-12-31")
    contract_range = (end_contract - start_contract).days
    random_days_contract = np.random.randint(
        0, contract_range + 1, size=num_unique_customers
    )
    contract_dates_master = (
        (start_contract + pd.to_timedelta(random_days_contract, unit="D"))
        .strftime("%Y%m%d")
        .astype(int)
        .tolist()
    )
    
    # マスタ辞書（顧客番号をキーに、各属性を固定）
    customer_master = {
        cid: {"性別": g, "年齢": a, "契約年月日": c, "出身地": r}
        for cid, g, a, c, r in zip(
            unique_customer_ids,
            genders_master,
            ages_master,
            contract_dates_master,
            regions_master,
        )
    }
    
    # 1万行の購買トランザクションデータの作成
    # 1万行分の顧客番号をランダムに割り当て
    customer_ids = np.random.choice(unique_customer_ids, size=n_rows)
    
    # 購買金額のベース生成
    purchase_amounts_base = np.random.randint(-5000, 50000, size=n_rows).astype(float)
    
    # 購買金額の1%（100行）をランダムに np.nan にする
    missing_amount_indices = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    purchase_amounts = [
        np.nan if i in missing_amount_indices else amt
        for i, amt in enumerate(purchase_amounts_base)
    ]
    
    # 購買年月日（整数8桁）のベース生成
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2025-12-31")
    date_range = (end_date - start_date).days
    random_days_purchase = np.random.randint(0, date_range + 1, size=n_rows)
    purchase_dates_base = (
        (start_date + pd.to_timedelta(random_days_purchase, unit="D"))
        .strftime("%Y%m%d")
        .astype(int)
        .tolist()
    )
    
    # 購買年月日の1%（100行）をランダムに欠損値（None）にする
    missing_purchase_indices = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    purchase_dates = [
        None if i in missing_purchase_indices else date
        for i, date in enumerate(purchase_dates_base)
    ]
    
    # 顧客番号をベースにマスタの属性を1万行に紐付け
    genders = [customer_master[cid]["性別"] for cid in customer_ids]
    ages = [customer_master[cid]["年齢"] for cid in customer_ids]
    contract_dates = [customer_master[cid]["契約年月日"] for cid in customer_ids]
    regions_base = [customer_master[cid]["出身地"] for cid in customer_ids]
    
    # 出身地（地域）の1%（100行）をランダムに欠損値（None）にする
    missing_region_indices = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    birthplaces = [
        None if i in missing_region_indices else rg for i, rg in enumerate(regions_base)
    ]
    
    # pandasに渡せる辞書データの作成
    dummy_data_dict = {
        "顧客番号": customer_ids,
        "契約年月日": contract_dates,
        "年齢": ages,
        "性別": genders,
        "出身地": birthplaces,
        "購入年月日": purchase_dates,
        "購買金額": purchase_amounts,
    }
        
    if write_csv:
        pl.DataFrame(dummy_data_dict).write_csv(write_csv) # デフォルトの１万行で約500KB
    elif not write_csv:
        return dummy_data_dict
```

## 1. データフレームの作成
pandasでもpolarsでもDataFrame関数が利用できます。
```python
# 辞書型のデータの雛形
dummy_data_dict = generate_dict_for_dataframe(n_rows=10000, seed=42)
```

```python
# pandas
df_pd = pd.DataFrame(dummy_data_dict)
print(df_pd)

# [out]
       顧客番号     契約年月日  年齢  性別  出身地       購入年月日     購買金額
0     C0699  20230711  75   2   関東  20250824.0  28044.0
1     C0965  20240723  36   1   関東  20251116.0  22883.0
2     C1476  20241223  44   2   中部  20250119.0   9063.0
3     C0661  20240212  73   2  北海道  20250803.0  28649.0
4     C1128  20201110  50   2   近畿  20250424.0  29828.0
...     ...       ...  ..  ..  ...         ...      ...
9995  C0788  20200618  25   2   中部  20250129.0   5219.0
9996  C0907  20240712  36   2   九州  20250118.0  18481.0
9997  C0864  20220714  69   2   東北  20250901.0  29536.0
9998  C1404  20220121  19   2   関東  20250624.0   1096.0
9999  C0667  20200722  68   2   東北  20250330.0  32953.0

[10000 rows x 7 columns]
```

```python
# polars
df_pl = pl.DataFrame(dummy_data_dict)
print(df_pl)

# [out]
shape: (10_000, 7)
┌──────────┬────────────┬──────┬──────┬────────┬────────────┬──────────┐
│ 顧客番号 ┆ 契約年月日 ┆ 年齢 ┆ 性別 ┆ 出身地 ┆ 購入年月日 ┆ 購買金額 │
│ ---      ┆ ---        ┆ ---  ┆ ---  ┆ ---    ┆ ---        ┆ ---      │
│ str      ┆ i64        ┆ i64  ┆ i64  ┆ str    ┆ i64        ┆ f64      │
╞══════════╪════════════╪══════╪══════╪════════╪════════════╪══════════╡
│ C0699    ┆ 20230711   ┆ 75   ┆ 2    ┆ 関東   ┆ 20250824   ┆ 28044.0  │
│ C0965    ┆ 20240723   ┆ 36   ┆ 1    ┆ 関東   ┆ 20251116   ┆ 22883.0  │
│ C1476    ┆ 20241223   ┆ 44   ┆ 2    ┆ 中部   ┆ 20250119   ┆ 9063.0   │
│ C0661    ┆ 20240212   ┆ 73   ┆ 2    ┆ 北海道 ┆ 20250803   ┆ 28649.0  │
│ C1128    ┆ 20201110   ┆ 50   ┆ 2    ┆ 近畿   ┆ 20250424   ┆ 29828.0  │
│ …        ┆ …          ┆ …    ┆ …    ┆ …      ┆ …          ┆ …        │
│ C0788    ┆ 20200618   ┆ 25   ┆ 2    ┆ 中部   ┆ 20250129   ┆ 5219.0   │
│ C0907    ┆ 20240712   ┆ 36   ┆ 2    ┆ 九州   ┆ 20250118   ┆ 18481.0  │
│ C0864    ┆ 20220714   ┆ 69   ┆ 2    ┆ 東北   ┆ 20250901   ┆ 29536.0  │
│ C1404    ┆ 20220121   ┆ 19   ┆ 2    ┆ 関東   ┆ 20250624   ┆ 1096.0   │
│ C0667    ┆ 20200722   ┆ 68   ┆ 2    ┆ 東北   ┆ 20250330   ┆ 32953.0  │
└──────────┴────────────┴──────┴──────┴────────┴────────────┴──────────┘
```

以下, 公式の関数の説明です。
### [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
```python
class pandas.DataFrame(
    data = None, # ndarray (structured or homogeneous), Iterable, dict, or DataFrame
    index = None, # Index or array-like
    columns = None, # Index or array-like
    dtype = None, # dtype, default None
    copy = None # bool or None, default None
)
"""
2次元で、サイズ変更が可能であり、潜在的に異なる型（ヘテロジニアス）のデータを保持し得る表形式のデータ。

データ構造には、ラベル付きの軸（行および列）も含まれる。
算術演算は、行と列の両方のラベルに基づいて整列される。
Seriesオブジェクトを格納する、辞書（dict）風のコンテナと捉えることが可能。
pandasにおける主要なデータ構造。
"""
```
### [polars.DataFrame](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html)
```python
class polars.DataFrame(
    data: FrameInitTypes | None = None, # dict, Sequence, ndarray, Series, or pandas.DataFrame
    schema: SchemaDefinition | None = None, # Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
    *,
    schema_overrides: SchemaDict | None = None, # dict, default None
    strict: bool = True, # bool, default True
    orient: Orientation | None = None, # {‘col’, ‘row’}, default None
    infer_schema_length: int | None = 100, # int or None
    nan_to_null: bool = False, # bool, default False
    height: int | None = None, # int or None, default None
)
"""
2次元で、データを行と列を伴う表として表現するデータ構造。
"""
```

## 2. データフレームの作成: csv読み込み
pandasでもpolarsでもread_csv関数が利用できます。(出力はDataFrameの時と同じなので割)愛)
```python
# csvファイルの生成
generate_dict_for_dataframe(n_rows=10000, seed=42, write_csv="dummy_data.csv")
```
```python
# pandas
df_pd = pd.read_csv("dummy_data.csv")
print(df_pd)
```

```python
# polars
df_pl = pl.read_csv("dummy_data.csv")
print(df_pl)
```
以下, 公式の関数の説明です (長いです) 。

### [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
```python
pandas.read_csv(
    filepath_or_buffer, # str, path object or file-like object
    *, 
    sep = <no_default>, # str, default ‘,’
    delimiter = None, # str, optional
    header = 'infer', # int, Sequence of int, ‘infer’ or None, default ‘infer’
    names = <no_default>, # Sequence of Hashable, optional
    index_col = None, # Hashable, Sequence of Hashable or False, optional
    usecols = None, # Sequence of Hashable or Callable, optional
    dtype = None, # dtype or dict of {Hashable : dtype}, optional
    engine = None, # {‘c’, ‘python’, ‘pyarrow’}, optional
    converters = None, # dict of {Hashable : Callable}, optional
    true_values = None, # list, optional
    false_values = None, # list, optional
    skipinitialspace = False, # bool, default False
    skiprows = None, # int, list of int or Callable, optional
    skipfooter = 0, # int, default 0
    nrows = None, # int, optional
    na_values = None, # Hashable, Iterable of Hashable or dict of {Hashable : Iterable},
    keep_default_na = True, # bool, default True
    na_filter = True, # bool, default True
    skip_blank_lines = True, # bool, default True
    parse_dates = None, # bool, None, list of Hashable, default None
    date_format = None, # str or dict of column -> format, optional
    dayfirst = False, # bool, default False
    cache_dates = True, # bool, default True
    iterator = False, # bool, default False
    chunksize = None, # int, optional 
    compression = 'infer', # str or dict, default ‘infer’
    thousands = None, # str (length 1), optional
    decimal = '.', # str (length 1), default ‘.’
    lineterminator = None, # str (length 1), optional
    quotechar = '"', # str (length 1), optional
    quoting = 0, # {0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL
    doublequote = True, # bool, default True
    escapechar = None, # str (length 1), optional
    comment = None, # str (length 1), optional
    encoding = None, # str, optional, default ‘utf-8’
    encoding_errors = 'strict', # str, optional, default ‘strict’
    dialect = None, # str or csv.Dialect, optional
    on_bad_lines = 'error', # {‘error’, ‘warn’, ‘skip’} or Callable, default ‘error’
    low_memory = True, # bool, default True
    memory_map = False, # bool, default False
    float_precision = None, # {‘high’, ‘legacy’, ‘round_trip’}, optional
    storage_options = None, # dict, optional
    dtype_backend = <no_default> # {‘numpy_nullable’, ‘pyarrow’}
) → DataFrame or TextFileReader
"""
カンマ区切り値（CSV）ファイルを読み込んで、DataFrameに変換する。

オプションで、ファイルの反復処理（イテレーション）や、チャンク（塊）への分割読み込みにも対応している。
"""
```

### [polars.read_csv](https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html)

```python
polars.read_csv(
    source: str | Path | IO[str] | IO[bytes] | bytes,
    *,
    has_header: bool = True,
    columns: Sequence[int] | Sequence[str] | None = None,
    new_columns: Sequence[str] | None = None,
    separator: str = ',',
    comment_prefix: str | None = None,
    quote_char: str | None = '"',
    skip_rows: int = 0,
    skip_lines: int = 0,
    schema: SchemaDict | None = None,
    schema_overrides: Mapping[str, polarsDataType] | Sequence[polarsDataType] | None = None,
    null_values: str | Sequence[str] | dict[str, str] | None = None,
    missing_utf8_is_empty_string: bool = False,
    ignore_errors: bool = False,
    try_parse_dates: bool = False,
    n_threads: int | None = None,
    infer_schema: bool = True,
    infer_schema_length: int | None = 100,
    batch_size: int = 8192,
    n_rows: int | None = None,
    encoding: CsvEncoding | str = 'utf8',
    low_memory: bool = False,
    rechunk: bool = False,
    use_pyarrow: bool = False,
    storage_options: StorageOptionsDict | None = None,
    skip_rows_after_header: int = 0,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    sample_size: int = 1024,
    eol_char: str = '\n',
    raise_if_empty: bool = True,
    truncate_ragged_lines: bool = False,
    decimal_comma: bool = False,
    glob: bool = True,
) → DataFrame
"""
CSVファイルを読み込んで、DataFrameに変換する。

ドキュメントに特記されていない限り、polarsはCSVデータがRFC 4180に厳格に準拠していることを前提としている。
不整形な（フォーマットが崩れた）データは、一般によく見られるものではあるが、未定義の動作を引き起こす可能性がある。
"""
```

## 3. インデックス列の追加
pandasとpolarsのデータフレームの出力を比較すると分かるように, **polarsにはインデックスが存在しません。** インデックスに相当する列が必要な場合は, with_row_indexを利用して再代入します。

**この記事では, pandasとの出力結果の比較がしやすいように, インデックスを追加したデータフレームを使います。**
ご自身でpolarsのデータフレームを新しく作成された場合はインデックス列は存在しませんので, ご注意ください。

```python
df_pl = pl.DataFrame(dummy_data_dict).with_row_index()
print(df_pl)

# [out]
shape: (10_000, 8)
┌───────┬──────────┬────────────┬──────┬──────┬────────┬────────────┬──────────┐
│ index ┆ 顧客番号 ┆ 契約年月日 ┆ 年齢 ┆ 性別 ┆ 出身地 ┆ 購入年月日 ┆ 購買金額 │
│ ---   ┆ ---      ┆ ---        ┆ ---  ┆ ---  ┆ ---    ┆ ---        ┆ ---      │
│ u32   ┆ str      ┆ i64        ┆ i64  ┆ i64  ┆ str    ┆ i64        ┆ f64      │
╞═══════╪══════════╪════════════╪══════╪══════╪════════╪════════════╪══════════╡
│ 0     ┆ C0699    ┆ 20230711   ┆ 75   ┆ 2    ┆ 関東   ┆ 20250824   ┆ 28044.0  │
│ 1     ┆ C0965    ┆ 20240723   ┆ 36   ┆ 1    ┆ 関東   ┆ 20251116   ┆ 22883.0  │
│ 2     ┆ C1476    ┆ 20241223   ┆ 44   ┆ 2    ┆ 中部   ┆ 20250119   ┆ 9063.0   │
│ 3     ┆ C0661    ┆ 20240212   ┆ 73   ┆ 2    ┆ 北海道 ┆ 20250803   ┆ 28649.0  │
│ 4     ┆ C1128    ┆ 20201110   ┆ 50   ┆ 2    ┆ 近畿   ┆ 20250424   ┆ 29828.0  │
│ …     ┆ …        ┆ …          ┆ …    ┆ …    ┆ …      ┆ …          ┆ …        │
│ 9995  ┆ C0788    ┆ 20200618   ┆ 25   ┆ 2    ┆ 中部   ┆ 20250129   ┆ 5219.0   │
│ 9996  ┆ C0907    ┆ 20240712   ┆ 36   ┆ 2    ┆ 九州   ┆ 20250118   ┆ 18481.0  │
│ 9997  ┆ C0864    ┆ 20220714   ┆ 69   ┆ 2    ┆ 東北   ┆ 20250901   ┆ 29536.0  │
│ 9998  ┆ C1404    ┆ 20220121   ┆ 19   ┆ 2    ┆ 関東   ┆ 20250624   ┆ 1096.0   │
│ 9999  ┆ C0667    ┆ 20200722   ┆ 68   ┆ 2    ┆ 東北   ┆ 20250330   ┆ 32953.0  │
└───────┴──────────┴────────────┴──────┴──────┴────────┴────────────┴──────────┘
```

### [polars.DataFrame.with_row_index](https://docs.pola.rs/api/python/version/0.20/reference/dataframe/api/polars.DataFrame.with_row_index.html)
```python
# polars
DataFrame.with_row_index(
    name: str = 'index', 
    offset: int = 0
) → Self
"""
データフレームの最初の列にインデックスを追加する。
"""
```

## 4. 列の追加
pandasではassignで, polarsではwith_columnsで列を追加できます。
その際, .aliasで列名を指定できます。

```python
# pandas
df_pd.assign(
    集計用=1, 
    出身地方=df_pd["出身地"] + "地方",
)

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額	集計用	出身地方
0	C0699	20230711	75	2	関東	20250824.0	28044.0	1	関東地方
1	C0965	20240723	36	1	関東	20251116.0	22883.0	1	関東地方
2	C1476	20241223	44	2	中部	20250119.0	9063.0	1	中部地方
3	C0661	20240212	73	2	北海道	20250803.0	28649.0	1	北海道地方
4	C1128	20201110	50	2	近畿	20250424.0	29828.0	1	近畿地方
...	...	...	...	...	...	...	...	...	...
9995	C0788	20200618	25	2	中部	20250129.0	5219.0	1	中部地方
9996	C0907	20240712	36	2	九州	20250118.0	18481.0	1	九州地方
9997	C0864	20220714	69	2	東北	20250901.0	29536.0	1	東北地方
9998	C1404	20220121	19	2	関東	20250624.0	1096.0	1	関東地方
9999	C0667	20200722	68	2	東北	20250330.0	32953.0	1	東北地方
10000 rows × 9 columns
```

```python
# polars
df_pl.with_columns(
    pl.lit(1).alias("集計用"),
    (pl.col("出身地") + "地方").alias("出身地方"),
)

# [out]
shape: (10_000, 10)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額	集計用	出身地方
u32	str	i64	i64	i64	str	i64	f64	i32	str
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0	1	"関東地方"
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0	1	"関東地方"
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0	1	"中部地方"
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0	1	"北海道地方"
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0	1	"近畿地方"
…	…	…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0	1	"中部地方"
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0	1	"九州地方"
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0	1	"東北地方"
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0	1	"関東地方"
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0	1	"東北地方"
```
以下, 公式の関数説明です。
### [pandas.DataFrame.assign](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html)
```python
DataFrame.assign(
    **kwargs # callable or Series
) → DataFrame
"""
DataFrameに新しい列を割り当てる。

新しい列に加え、すべての元の列を含んだ新しいオブジェクトを返す。
再割り当てされた既存の列は上書きされる。
"""
```
### [polars.DataFrame.with_columns](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_columns.html)
```python
DataFrame.with_columns(
    *exprs: IntoExpr | Iterable[IntoExpr],
    **named_exprs: IntoExpr,
) → DataFrame
"""
このDataFrameに列を追加する。

追加された列は、同じ名前を持つ既存の列を置き換える。
"""
```

## 5. 列名の変更
pandasでもpolarsでもrename関数を使って列名を変更できます。
```python
# pandas
df_pd.rename(columns={"購買金額": "利用額"})

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	利用額
0	C0699	20230711	75	2	関東	20250824.0	28044.0
1	C0965	20240723	36	1	関東	20251116.0	22883.0
2	C1476	20241223	44	2	中部	20250119.0	9063.0
3	C0661	20240212	73	2	北海道	20250803.0	28649.0
4	C1128	20201110	50	2	近畿	20250424.0	29828.0
...	...	...	...	...	...	...	...
9995	C0788	20200618	25	2	中部	20250129.0	5219.0
9996	C0907	20240712	36	2	九州	20250118.0	18481.0
9997	C0864	20220714	69	2	東北	20250901.0	29536.0
9998	C1404	20220121	19	2	関東	20250624.0	1096.0
9999	C0667	20200722	68	2	東北	20250330.0	32953.0
10000 rows × 7 columns
```

```python
# polars
df_pl.rename({"購買金額": "利用額"})

# [out]
shape: (10_000, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	利用額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0
```

polarsでは, 関数やデータフレームのメソッドで列を指定するときは, エクスプレッション (pl.Expr) という記法を用いることで, 柔軟で高速な処理を実現することができます (列同士の計算結果を新しい列として追加したり)。
これは例えば, pl.col("列名")や, pl.all()といった書き方です。
この記事で詳細には解説しませんが, [こちらの記事](https://qiita.com/Johannyjm/items/125736c186507a12eda0)で詳しく説明されていますので, ご参照ください。

```python
# polars pl.colを用いた列名変更(推奨)
df_pl.with_columns(
    pl.col("購買金額").alias("利用額") # .aliasで新しい列名を指定
)
# 出力はrenameと同じなので割愛
```
以下, 公式の関数の説明です。
### [pandas.DataFrame.rename](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)
```python
DataFrame.rename(
    mapper = None, # dict-like or function
    *,  
    index = None, # dict-like or function
    columns = None, # dict-like or function
    axis = None, # {0 or ‘index’, 1 or ‘columns’}, default 0
    copy = <no_default>, # bool, default False
    inplace = False, # bool, default False
    level = None, # int or level name, default None
    errors = 'ignore' # {‘ignore’, ‘raise’}, default ‘ignore’
) → DataFrame or None
"""
列名またはインデックスのラベルを変更する。

関数または辞書（dict）の値はユニーク（1対1）である必要がある。
辞書またはSeriesに含まれないラベルはそのまま残される。
指定された余分なラベルによってエラーが発生することはない。
"""
```

### [polars.DataFrame.rename](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rename.html)
```python
DataFrame.rename(
    mapping: Mapping[str, str] | Callable[[str], str],
    *,
    strict: bool = True,
) → DataFrame
"""
列名を変更する。
"""
```

## 6. 列の選択
pandasではfilter, polarsではselectが利用できます。polarsでは, pl.colを用いる記法により, 既存の列に処理を行った新しい列を作成することもできます。

```python
# pandas
df_pd.filter(items=["性別", "顧客番号"])

# [out]
	性別	顧客番号
0	2	C0699
1	1	C0965
2	2	C1476
3	2	C0661
4	2	C1128
...	...	...
9995	2	C0788
9996	2	C0907
9997	2	C0864
9998	2	C1404
9999	2	C0667
10000 rows × 2 columns
```

```python
# polars
df_pl.select(
    pl.col("性別"), 
    pl.col("顧客番号"),
)

# [out]
shape: (10_000, 2)
性別	顧客番号
i64	str
2	"C0699"
1	"C0965"
2	"C1476"
2	"C0661"
2	"C1128"
…	…
2	"C0788"
2	"C0907"
2	"C0864"
2	"C1404"
2	"C0667"
```

```python
# polars
# こういう書き方もできる
df_pl.select(["性別", "顧客番号"])
df_pl[["性別", "顧客番号"]]
```

以下, 公式の関数の説明です。
### [pandas.DataFrame.filter](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html)
```python
DataFrame.filter(
    items = None, # list-like
    like = None, # str
    regex = None, # str (regular expression)
    axis = None # {0 or ‘index’, 1 or ‘columns’, None}, default None
) → Series or DataFrame
"""
指定されたインデックスのラベルに従って、DataFrameまたはSeriesの部分集合（サブセット）を抽出する。

DataFrameの場合、axis引数に応じて行または列をフィルタリングする。
なお、このルーチンはデータの内容（コンテンツ）に基づいてフィルタリングを行うものではないことに注意。
フィルターはインデックスのラベルに対して適用される。
"""
```
### [polars.select](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.select.html)
```python
polars.select(
    *exprs: IntoExpr | Iterable[IntoExpr],
    eager: bool = True,
    **named_exprs: IntoExpr,
) → DataFrame | LazyFrame
"""
コンテキスト（DataFrameなどの枠組み）を伴わずにpolarsの式（Expressions）を実行する。

これは、空のDataFrame（または eager=False の場合はLazyFrame）に対して df.select を実行するためのシンタックスシュガー（構文糖衣）。
"""
```

## 7. 列の削除
pandasではdropを利用して, polarsではpl.selectとpl.excludeを組み合わせて列の削除を実現します。
```python
# pandas
df_pd.drop(columns=["性別"])

# [out]
	顧客番号	契約年月日	年齢	出身地	購入年月日	購買金額
0	C0699	20230711	75	関東	20250824.0	28044.0
1	C0965	20240723	36	関東	20251116.0	22883.0
2	C1476	20241223	44	中部	20250119.0	9063.0
3	C0661	20240212	73	北海道	20250803.0	28649.0
4	C1128	20201110	50	近畿	20250424.0	29828.0
...	...	...	...	...	...	...
9995	C0788	20200618	25	中部	20250129.0	5219.0
9996	C0907	20240712	36	九州	20250118.0	18481.0
9997	C0864	20220714	69	東北	20250901.0	29536.0
9998	C1404	20220121	19	関東	20250624.0	1096.0
9999	C0667	20200722	68	東北	20250330.0	32953.0
10000 rows × 6 columns
```

```python
# polars
df_pl.select(pl.exclude("列"))

# [out]
shape: (10_000, 7)
index	顧客番号	契約年月日	年齢	出身地	購入年月日	購買金額
u32	str	i64	i64	str	i64	f64
0	"C0699"	20230711	75	"関東"	20250824	28044.0
1	"C0965"	20240723	36	"関東"	20251116	22883.0
2	"C1476"	20241223	44	"中部"	20250119	9063.0
3	"C0661"	20240212	73	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	"近畿"	20250424	29828.0
…	…	…	…	…	…	…
9995	"C0788"	20200618	25	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	"東北"	20250330	32953.0
```

以下, 公式の関数の説明です。
### [pandas.DataFrame.drop](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html)
```python
DataFrame.drop(
    labels = None, # single label or iterable of labels
    *, 
    axis = 0, # {0 or ‘index’, 1 or ‘columns’}, default 0
    index = None, # single label or iterable of labels
    columns = None, # single label or iterable of labels
    level = None, # int or level name, optional
    inplace = False, # bool, default False
    errors = 'raise' # {‘ignore’, ‘raise’}, default ‘raise’
) → DataFrame or None
"""
行または列から指定されたラベルを削除する。

ラベル名と対応する軸を指定するか、あるいはインデックス名や列名を直接指定することによって、行または列を削除する。
マルチインデックスを使用する場合、レベルを指定することで異なるレベルのラベルを削除できる。
"""
```

### [polars.exclude](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.exclude.html)
```python
polars.exclude(
    columns: str | polarsDataType | Collection[str] | Collection[polarsDataType],
    *more_columns: str | polarsDataType,
) → Expr
"""
指定された列を除く、すべての列を表現する。

pl.all().exclude(columns) を実行するためのシンタックスシュガー（構文糖衣）。
"""
```

## 8. 行の絞り込み
pandasではquery, polarsではfilter関数が利用できます。

```python
# pandas
df_pd.query('性別 == 1 & 年齢 <= 50')
# ↓ query関数を使わない形
# df_pd[
#     (df_pd["性別"] == 1) & (df_pd["年齢"] <= 50)
# ]

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
1	C0965	20240723	36	1	関東	20251116.0	22883.0
17	C1735	20200112	20	1	NaN	20251223.0	19187.0
25	C0495	20200518	37	1	四国	20251106.0	45290.0
33	C0616	20240617	37	1	近畿	20250909.0	46571.0
36	C0206	20201126	26	1	中部	20250131.0	6278.0
...	...	...	...	...	...	...	...
9975	C0011	20221218	40	1	九州	20250707.0	13755.0
9976	C1606	20230520	25	1	中国	20250904.0	27311.0
9983	C1283	20210104	33	1	近畿	20250624.0	40370.0
9990	C1962	20210607	48	1	関東	20250512.0	656.0
9992	C0619	20211011	23	1	近畿	20251002.0	37203.0
2557 rows × 7 columns
```

```python
# polars
df_pl.filter(
    (pl.col("性別") == 1) & (pl.col("年齢") <= 50)
)

# [out]
shape: (2_557, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
17	"C1735"	20200112	20	1	null	20251223	19187.0
25	"C0495"	20200518	37	1	"四国"	20251106	45290.0
33	"C0616"	20240617	37	1	"近畿"	20250909	46571.0
36	"C0206"	20201126	26	1	"中部"	20250131	6278.0
…	…	…	…	…	…	…	…
9975	"C0011"	20221218	40	1	"九州"	20250707	13755.0
9976	"C1606"	20230520	25	1	"中国"	20250904	27311.0
9983	"C1283"	20210104	33	1	"近畿"	20250624	40370.0
9990	"C1962"	20210607	48	1	"関東"	20250512	656.0
9992	"C0619"	20211011	23	1	"近畿"	20251002	37203.0
```

以下, 公式の関数の説明です。

### [pandas.DataFrame.query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
```python
DataFrame.query(
    expr, # str
    *, 
    parser = 'pandas', # {‘pandas’, ‘python’}, default ‘pandas’
    engine = None, # {‘python’, ‘numexpr’}, default ‘numexpr’
    local_dict = None, # dict or None, optional
    global_dict = None, # dict or None, optional
    resolvers = None, # list of dict-like or None, optional
    level = 0, # int, optional
    inplace = False # bool
) → DataFrame or None
"""
ブール式（真偽値を返す条件式）を使って、DataFrameの列をクエリ（条件指定による行の抽出）する。

このメソッドは任意のコードを実行できるため、関数の引数にユーザー入力をそのまま渡すと、コードインジェクション（不正コードの注入・実行）に対して脆弱になる危険性がある。
"""
```

### [polars.DataFrame.filter](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.filter.html)
```python
DataFrame.filter(
    *predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any],
    **constraints: Any,
) → DataFrame
"""
指定された述語式（条件式）にマッチする行を保持し、行をフィルタリングする。

残された行の元の順序は維持される。
述語（条件）の結果が True になる行のみが保持され、結果が False または null（欠損値）の行は破棄される。
"""
```

## 9. 重複の削除
pandasではdrop_duplicates, polarsではunique関数が利用できます。
```python
# pandas
df_pd.drop_duplicates(subset=["年齢"])

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
0	C0699	20230711	75	2	関東	20250824.0	28044.0
1	C0965	20240723	36	1	関東	20251116.0	22883.0
2	C1476	20241223	44	2	中部	20250119.0	9063.0
3	C0661	20240212	73	2	北海道	20250803.0	28649.0
4	C1128	20201110	50	2	近畿	20250424.0	29828.0
...	...	...	...	...	...	...	...
219	C1658	20200130	35	1	関東	20250911.0	19102.0
232	C0921	20241230	80	1	中国	20251216.0	2377.0
239	C0334	20200406	30	1	中国	20250305.0	49920.0
249	C1117	20210425	41	1	中国	20250721.0	49607.0
260	C0576	20240923	19	2	北海道	20250702.0	18130.0
63 rows × 7 columns
```

```python
# polars
df_pl.unique(subset=pl.col("年齢"), keep="first", maintain_order=True)

# [out]
shape: (63, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
219	"C1658"	20200130	35	1	"関東"	20250911	19102.0
232	"C0921"	20241230	80	1	"中国"	20251216	2377.0
239	"C0334"	20200406	30	1	"中国"	20250305	49920.0
249	"C1117"	20210425	41	1	"中国"	20250721	49607.0
260	"C0576"	20240923	19	2	"北海道"	20250702	18130.0
```

以下, 公式の関数説明です。
### [pandas.DataFrame.drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)
```python
DataFrame.drop_duplicates(
    subset = None, # column label or iterable of labels, optional
    *, 
    keep = 'first', # {‘first’, ‘last’, False}, default ‘first’
    inplace = False, # bool, default False
    ignore_index = False # bool, default False
) → DataFrame or None
"""
重複した行が削除されたDataFrameを返す。

特定の列のみを考慮して（重複判定を行って）削除することも可能（オプション）。
タイムインデックスを含むインデックス情報は無視される。
"""
```

### [polars.DataFrame.unique](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.unique.html)
```python
DataFrame.unique(
    subset: IntoExpr | Collection[IntoExpr] | None = None,
    *,
    keep: UniqueKeepStrategy = 'any', # {‘first’, ‘last’, ‘any’, ‘none’}
    maintain_order: bool = False,
) → DataFrame
"""
このデータフレームから重複した行を削除する。
"""
```

## 10. 欠損値: nullとNaNの説明
polarsでは欠損値にnullとNaNがあり, これらは区別されています。
[公式の説明はこちら。](https://docs.pola.rs/user-guide/expressions/missing-data/)
nullは全てのデータ型に対して用いられる欠損値です。
NaN(Not a Number)は, 小数点の値を持つ列に対して用いられる値で, 厳密には欠損値ではなく, 有効な小数点の値としてみなされます。 
今回のダミーデータでいうと, 購買年月日や出身地はデータ生成時にNoneを含むようにしていて, これらはpolarsではnullになります。一方, 購買金額はnp.nanを含むようにしていて, これはNaNとして扱われます。
なお, pandasのデータフレームやシリーズのメソッドでは.isnan()とisnull()がありますが, これらは単なる別名で, 処理は全く同じです。

## 11. 欠損値の検出
pandasではis_nullおよびisnanが利用できます。これらはブール値のデータフレームやシリーズを返すので, 合計することで欠損値の個数を数えることができます。
polarsでは, pl.Expr表現とis_nullおよびis_nanを組み合わせます。注意点として, is_nullは数値型でない列に対して実行しようとするとエラーが発生するので, polars.selectorsを用いたり列を明示的に選択することで, データフレームのうち数値型の列だけに対して処理を行うようにします。

```python
# pandas
df_pd.isnull().sum()

# [out]
顧客番号       0
契約年月日      0
年齢         0
性別         0
出身地      100
購入年月日    100
購買金額     100
dtype: int64
```

```python
# polars: is_null どのデータ型に対しても利用できる
df_pl.select(
    pl.all().is_null()
).sum()

# [out]
shape: (1, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	u32	u32	u32	u32	u32	u32	u32
0	0	0	0	0	100	100	0
```

```python
# polars: is_nan 数値型に対してのみ利用できる
import polars.selectors as cs
df_pl.select(
    cs.numeric().is_nan()
).sum()

# [out]
shape: (1, 6)
index	契約年月日	年齢	性別	購入年月日	購買金額
u32	u32	u32	u32	u32	u32
0	0	0	0	0	100
```
以下, 公式の関数の説明です。

### [pandas.DataFrame.isnull](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isnull.html)
```python
DataFrame.isnull() → Series or DataFrame
"""
欠損値を検出する。

DataFrame.isnull は DataFrame.isna のエイリアス（別名）である。
値がNA（欠損値）であるかどうかを示す、元のオブジェクトと同じサイズのブール値（真偽値）のオブジェクトを返す。
None や numpy.NaN などのNA値は True にマッピングされ、それ以外のすべての値は False にマッピングされる。
空文字列 '' や numpy.inf（無限大）などの文字や値は、NA値とはみなされない。
"""
```

### [polars.Expr.is_null](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_null.html)
```python
Expr.is_null() → Expr
"""
値がnull（欠損値）であるかどうかを示す、ブール値（真偽値）のSeriesを返す。
"""
```

### [polars.Expr.is_nan](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_nan.html)
```python
Expr.is_nan() → Expr
"""
値がNaN（Not a Number / 非数）であるかどうかを示す、ブール値（真偽値）のSeriesを返す。
"""
```

## 12. 欠損値の削除
pandasではdropna, polarsでは例によってdrop_nullsとdrop_nansがあります。

```python
# pandas
df_pd.dropna()

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
0	C0699	20230711	75	2	関東	20250824.0	28044.0
1	C0965	20240723	36	1	関東	20251116.0	22883.0
2	C1476	20241223	44	2	中部	20250119.0	9063.0
3	C0661	20240212	73	2	北海道	20250803.0	28649.0
4	C1128	20201110	50	2	近畿	20250424.0	29828.0
...	...	...	...	...	...	...	...
9995	C0788	20200618	25	2	中部	20250129.0	5219.0
9996	C0907	20240712	36	2	九州	20250118.0	18481.0
9997	C0864	20220714	69	2	東北	20250901.0	29536.0
9998	C1404	20220121	19	2	関東	20250624.0	1096.0
9999	C0667	20200722	68	2	東北	20250330.0	32953.0
9702 rows × 7 columns
```

```python
# polars (drop_nulls)
df_pl.drop_nulls()

# [out]
shape: (9_800, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0
```

```python
# polars (drop_nans)
df_pl.drop_nans()

# [out]
shape: (9_900, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0
```

```python
# polars (drop_nulls + drop_nans) pandasのdropnaと同じ
df_pl.drop_nulls().drop_nans()

# [out]
shape: (9_702, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東地方"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東地方"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部地方"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道地方"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿地方"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部地方"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州地方"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北地方"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東地方"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北地方"	20250330	32953.0
```


以下, 公式の関数の説明です。
### [pandas.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
```python
DataFrame.dropna(
    *, 
    axis = 0, # {0 or ‘index’, 1 or ‘columns’}, default 0
    how = <no_default>, # {‘any’, ‘all’}, default ‘any’
    thresh = <no_default>, # int, optional
    subset = None, # column label or iterable of labels, optional
    inplace = False, # bool, default False
    ignore_index = False # bool, default False
) → DataFrame or None
"""
欠損値を除去する。
"""
```

### [polars.DataFrame.drop_nulls](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.drop_nulls.html)
```python
DataFrame.drop_nulls(
    subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
) → DataFrame
"""
1つ以上のnull値を含むすべての行を削除する。

残された行の元の順序は維持される。
"""
```

### [polars.DataFrame.drop_nans](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.drop_nans.html)
```python
DataFrame.drop_nans(
    subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
) → DataFrame
"""
1つ以上のNaN値を含むすべての行を削除する。

残された行の元の順序は維持される。
"""
```

## 13. 欠損値の値埋め
pandasではfillna, polarsではNaN値とnull値についてそれぞれfill_nan, fill_nullが利用できます。
```python
# pandas
df_pd.fillna(0)

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
0	C0699	20230711	75	2	関東	20250824.0	28044.0
1	C0965	20240723	36	1	関東	20251116.0	22883.0
2	C1476	20241223	44	2	中部	20250119.0	9063.0
3	C0661	20240212	73	2	北海道	20250803.0	28649.0
4	C1128	20201110	50	2	近畿	20250424.0	29828.0
...	...	...	...	...	...	...	...
9995	C0788	20200618	25	2	中部	20250129.0	5219.0
9996	C0907	20240712	36	2	九州	20250118.0	18481.0
9997	C0864	20220714	69	2	東北	20250901.0	29536.0
9998	C1404	20220121	19	2	関東	20250624.0	1096.0
9999	C0667	20200722	68	2	東北	20250330.0	32953.0
10000 rows × 7 columns
```

```python
# polars (fill_null)
df_pl.fill_null(0)

# [out]
shape: (10_000, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0
```

```python
# polars (fill_nan)
df_pl.fill_nan(0)

# [out]
shape: (10_000, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0
```

以下, 公式の関数の説明です。
### [pandas.fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)
```python
DataFrame.fillna(
    value, # scalar, dict, Series, or DataFrame
    *, 
    axis = None, # {0 or ‘index’} for Series, {0 or ‘index’, 1 or ‘columns’} for DataFrame
    inplace = False, # bool, default False
    limit = None # int, default None
) → Series or DataFrame
"""
NA/NaN値を指定した値で埋める。
"""
```

### [polars.DataFrame.fill_null](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.fill_null.html)
```python
DataFrame.fill_null(
    value: Any | Expr | None = None,
    strategy: FillNullStrategy | None = None, # {None, ‘forward’, ‘backward’, ‘min’, ‘max’, ‘mean’, ‘zero’, ‘one’}
    limit: int | None = None,
    *,
    matches_supertype: bool = True,
) → DataFrame
"""
指定された値またはストラテジー（手法）を用いて、null値を埋める。
"""
```

### [polars.DataFrame.fill_nan](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.fill_nan.html)
```python
polars.DataFrame.fill_nan

DataFrame.fill_nan(
    value: Expr | int | float | None
) → DataFrame
"""
式の評価によって、浮動小数点数のNaN値（欠損値）を埋める。
"""
```

## 14. データフレームの結合： 縦方向
pandasでもpolarsでもconcatが利用できます。

```python
# 結合用の新しいデータフレームの用意
dummy_dict_data2 = generate_dict_for_dataframe(seed=1)
df_pd2 = pd.DataFrame(dummy_dict_data2)
df_pl2 = pl.DataFrame(dummy_dict_data2).with_row_index()
```

```python
# pandas
pd.concat([df_pd, df_pd2])

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
0	C0699	20230711	75	2	関東	20250824.0	28044.0
1	C0965	20240723	36	1	関東	20251116.0	22883.0
2	C1476	20241223	44	2	中部	20250119.0	9063.0
3	C0661	20240212	73	2	北海道	20250803.0	28649.0
4	C1128	20201110	50	2	近畿	20250424.0	29828.0
...	...	...	...	...	...	...	...
9995	C0788	20200618	25	2	中部	20250129.0	5219.0
9996	C0907	20240712	36	2	九州	20250118.0	18481.0
9997	C0864	20220714	69	2	東北	20250901.0	29536.0
9998	C1404	20220121	19	2	関東	20250624.0	1096.0
9999	C0667	20200722	68	2	東北	20250330.0	32953.0
20000 rows × 7 columns
```

```python
# polars
pl.concat([df_pl, df_pl2])

# [out]
shape: (20_000, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0
…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0
```

以下, 公式の関数説明です。
### [pandas.concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)
```python
pandas.concat(
    objs, # an iterable or mapping of Series or DataFrame objects
    *, 
    axis = 0, # {0/’index’, 1/’columns’}, default 0
    join = 'outer', # {‘inner’, ‘outer’}, default ‘outer’
    ignore_index = False, # bool, default False
    keys = None, # sequence, default None
    levels = None, # list of sequences, default None
    names = None, # list, default None
    verify_integrity = False, # bool, default False
    sort = <no_default>, # bool, default False
    copy = <no_default> # bool, default False
) → Series or DataFrame
"""
特定の軸に沿ってpandasオブジェクトを結合（連結）する。

他の軸に沿った任意の集合論理（set logic、積集合や和集合など）を適用できる。
また、結合する軸に対して階層型インデックス（MultiIndex）のレイヤーを追加することも可能。
これは、渡された軸の番号上でラベルが同じ（または重複している）場合に有用。
"""
```

### [polars.concat](https://docs.pola.rs/api/python/stable/reference/api/polars.concat.html)
```python
polars.concat(
    items: Iterable[polarsType],
    *,
    how: ConcatMethod = 'vertical', # {‘vertical’, ‘vertical_relaxed’, ‘diagonal’, ‘diagonal_relaxed’, ‘horizontal’, ‘align’, ‘align_full’, ‘align_inner’, ‘align_left’, ‘align_right’}
    rechunk: bool = False,
    parallel: bool = True,
    strict: bool = False,
) → polarsType
"""
複数のDataFrame、LazyFrame、またはSeriesを単一のオブジェクトに結合（連結）する。
"""
```

## 15. データフレームの結合： 横方向
pandasではmerge, polarsではjoinが利用できます。

```python
# pandas
pd.merge(df_pd, df_pd2, on="顧客番号", how="left")

# [out}
	顧客番号	契約年月日_x	年齢_x	性別_x	出身地_x	購入年月日_x	購買金額_x	契約年月日_y	年齢_y	性別_y	出身地_y	購入年月日_y	購買金額_y
0	C0699	20230711	75	2	関東	20250824.0	28044.0	20230711	75	2	関東	20250824.0	28044.0
1	C0699	20230711	75	2	関東	20250824.0	28044.0	20230711	75	2	関東	20250710.0	42899.0
2	C0699	20230711	75	2	関東	20250824.0	28044.0	20230711	75	2	関東	20251008.0	15055.0
3	C0699	20230711	75	2	関東	20250824.0	28044.0	20230711	75	2	関東	20251023.0	27726.0
4	C0965	20240723	36	1	関東	20251116.0	22883.0	20240723	36	1	関東	20251116.0	22883.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...
59895	C1404	20220121	19	2	関東	20250624.0	1096.0	20220121	19	2	関東	20250804.0	7442.0
59896	C1404	20220121	19	2	関東	20250624.0	1096.0	20220121	19	2	関東	20250624.0	1096.0
59897	C0667	20200722	68	2	東北	20250330.0	32953.0	20200722	68	2	東北	20250204.0	4808.0
59898	C0667	20200722	68	2	東北	20250330.0	32953.0	20200722	68	2	東北	20250930.0	23874.0
59899	C0667	20200722	68	2	東北	20250330.0	32953.0	20200722	68	2	東北	20250330.0	32953.0
59900 rows × 13 columns
```

```python
# polars
df_pl.join(df_pl2, on="顧客番号", how="left")

# [out]
shape: (59_900, 15)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額	index_right	契約年月日_right	年齢_right	性別_right	出身地_right	購入年月日_right	購買金額_right
u32	str	i64	i64	i64	str	i64	f64	u32	i64	i64	i64	str	i64	f64
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0	0	20230711	75	2	"関東"	20250824	28044.0
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0	621	20230711	75	2	"関東"	20250710	42899.0
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0	8480	20230711	75	2	"関東"	20251008	15055.0
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0	9560	20230711	75	2	"関東"	20251023	27726.0
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0	1	20240723	36	1	"関東"	20251116	22883.0
…	…	…	…	…	…	…	…	…	…	…	…	…	…	…
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0	6048	20220121	19	2	"関東"	20250804	7442.0
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0	9998	20220121	19	2	"関東"	20250624	1096.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0	1463	20200722	68	2	"東北"	20250204	4808.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0	2623	20200722	68	2	"東北"	20250930	23874.0
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0	9999	20200722	68	2	"東北"	20250330	32953.0
```

以下, 公式の関数説明です。
### [pandas.merge](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
```python
pandas.merge(
    left, # DataFrame or named Series
    right, # DataFrame or named Series
    how = 'inner', # {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’, ‘left_anti’, ‘right_anti}
    on = None, # Hashable or a sequence of the previous
    left_on = None, # Hashable or a sequence of the previous, or array-like
    right_on = None, # Hashable or a sequence of the previous, or array-like
    left_index = False, # bool, default False
    right_index = False, # bool, default False
    sort = False, # bool, default False
    suffixes = ('_x', '_y'), # list-like, default is (“_x”, “_y”)
    copy = <no_default>, # bool, default False
    indicator = False, # bool or str, default False
    validate = None # str, optional {'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'}
) → DataFrame
"""
DataFrameまたは名前付きSeriesオブジェクトを、データベーススタイルの結合（Join）でマージする。

名前付きSeriesオブジェクトは、名前付きの列を1つだけ持つDataFrameとして処理される。
結合は列またはインデックスに基づいて行われる。
列同士を結合する場合、DataFrameのインデックスは無視される。
インデックス同士、またはインデックスと列を結合する場合は、インデックスの情報が引き継がれる。
クロス結合（クロス・マージ）を実行する場合、結合キーとなる列の指定は不可。
"""
```

### [polars.DataFrame.join](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.join.html)
```python
DataFrame.join(
    other: DataFrame,
    on: str | Expr | Sequence[str | Expr] | None = None,
    how: JoinStrategy = 'inner', # {‘inner’, ‘left’, ‘right’, ‘full’, ‘semi’, ‘anti’, ‘cross’}
    *,
    left_on: str | Expr | Sequence[str | Expr] | None = None,
    right_on: str | Expr | Sequence[str | Expr] | None = None,
    suffix: str = '_right',
    validate: JoinValidation = 'm:m', # {‘m:m’, ‘m:1’, ‘1:m’, ‘1:1’}
    nulls_equal: bool = False,
    coalesce: bool | None = None,
    maintain_order: MaintainOrderJoin | None = None, # {‘none’, ‘left’, ‘right’, ‘left_right’, ‘right_left’}
) → DataFrame
"""
データフレームをSQL風に結合する。
"""
```

## 16. 並び替え
pandasではsort_values, polarsではsortが利用できます。pandasは関数の引数がascending(昇順)なのに対し, polarsではdescending(降順)のブール値を指定することに気をつけてください。

また, **NaN値がpolarsのデータフレームに存在する場合, pandasのsort_valuesでna_position="first" (デフォルト値) を指定した時と, polarsのsortでnulls_last=False (デフォルト値) を指定したときで, 出力が一致しません。**

これは, pandasではNaNはna_positionに反応して頭に来ますが, polarsではNaNはnullではないので, nulls_lastには反応しないからです。

このように, polarsでは欠損値がnullなのかNaNなのかが重要な違いを持ちます。

```python
# pandas
df_pd.sort_values(
    ["年齢", "購買金額"], 
    ascending=[True, False],
    na_position="first",
)

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
9887	C0873	20241105	18	2	四国	20250602.0	NaN
6037	C0825	20220731	18	2	東北	20250715.0	49422.0
3420	C0791	20221103	18	2	中国	20250906.0	48494.0
2912	C0456	20200711	18	1	中国	20250322.0	47913.0
3697	C0884	20200328	18	2	九州	20251129.0	47627.0
...	...	...	...	...	...	...	...
2455	C1071	20211220	80	1	四国	20250905.0	-2221.0
6787	C0623	20230921	80	1	東北	20250109.0	-2301.0
4193	C0183	20200208	80	2	東北	20250111.0	-3690.0
5610	C0623	20230921	80	1	東北	20250609.0	-3822.0
9958	C0168	20211005	80	1	近畿	20250517.0	-4865.0
10000 rows × 7 columns
```

```python
# polars
df_pl.sort(
    [pl.col("年齢"), pl.col("購買金額")], 
    descending=[False, True],
    maintain_order=True,
    nulls_last=True
)

# [out]
shape: (10_000, 8)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
u32	str	i64	i64	i64	str	i64	f64
9887	"C0873"	20241105	18	2	"四国"	20250602	NaN
6037	"C0825"	20220731	18	2	"東北"	20250715	49422.0
3420	"C0791"	20221103	18	2	"中国"	20250906	48494.0
2912	"C0456"	20200711	18	1	"中国"	20250322	47913.0
3697	"C0884"	20200328	18	2	"九州"	20251129	47627.0
…	…	…	…	…	…	…	…
2455	"C1071"	20211220	80	1	"四国"	20250905	-2221.0
6787	"C0623"	20230921	80	1	"東北"	20250109	-2301.0
4193	"C0183"	20200208	80	2	"東北"	20250111	-3690.0
5610	"C0623"	20230921	80	1	"東北"	20250609	-3822.0
9958	"C0168"	20211005	80	1	"近畿"	20250517	-4865.0
```

以下, 公式の関数の説明です。

### [pandas.DataFrame.sort_values](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html)
```python
DataFrame.sort_values(
    by, # str or list of str
    *, 
    axis = 0, # “{0 or ‘index’, 1 or ‘columns’}”, default 0
    ascending = True, # bool or list of bool, default True
    inplace = False, # bool, default False
    kind = 'quicksort', # {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, default ‘quicksort’
    na_position = 'last', # {‘first’, ‘last’}, default ‘last’
    ignore_index = False, # bool, default False
    key = None # callable, optional
) → DataFrame or None
"""
いずれかの軸に沿って、値ベースでソート（並べ替え）を行う。
"""
```

### [polars.DataFrame.sort](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.sort.html)
```python
DataFrame.sort(
    by: IntoExpr | Iterable[IntoExpr],
    *more_by: IntoExpr,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    multithreaded: bool = True,
    maintain_order: bool = False,
) → DataFrame
"""
指定した列に沿ってデータフレームをソートする。
"""
```

## 17. グループバイ
pandasではgroupby, polarsではgroup_byが利用できます。polarsでは, 列の追加時と同様に.aliasで列名を指定できます。
なお, polarsではmaintain_order=Trueを設定しないと, 実行のたびに行の順番が変わります (その分早い)。


```python
# pandas
df_pd.groupby("顧客番号", as_index=False, sort=False).agg(
    年齢=("年齢", "last"),
    契約年月日=("契約年月日", "last"),
    合計購買金額=("購買金額", "sum"),
)

# [out]
	顧客番号	年齢	契約年月日	合計購買金額
0	C0699	75	20230711	113724.0
1	C0965	36	20240723	180273.0
2	C1476	44	20241223	120477.0
3	C0661	73	20240212	147906.0
4	C1128	50	20201110	173258.0
...	...	...	...	...
1981	C1842	73	20240618	31415.0
1982	C0034	43	20240209	47087.0
1983	C0370	21	20230624	80221.0
1984	C0120	33	20230107	63534.0
1985	C0136	62	20201027	30809.0
1986 rows × 4 columns
```

```python
# polars
df_pl.group_by(
    pl.col("顧客番号"),
    maintain_order=True
).agg(
    pl.last("年齢"),
    pl.last("契約年月日"),
    pl.sum("購買金額").alias("合計購入金額"),
)

# [out]
shape: (1_986, 4)
顧客番号	年齢	契約年月日	合計購入金額
str	i64	i64	f64
"C0699"	75	20230711	113724.0
"C0965"	36	20240723	180273.0
"C1476"	44	20241223	120477.0
"C0661"	73	20240212	147906.0
"C1128"	50	20201110	173258.0
…	…	…	…
"C1842"	73	20240618	31415.0
"C0034"	43	20240209	47087.0
"C0370"	21	20230624	80221.0
"C0120"	33	20230107	63534.0
"C0136"	62	20201027	30809.0

```
以下, 公式の関数の説明です。
### [pandas.DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)
```python
DataFrame.groupby(
    by = None, # mapping, function, label, pd.Grouper or list of such
    level = None, # int, level name, or sequence of such, default None
    *, 
    as_index = True, # bool, default True
    sort = True, # bool, default True
    group_keys = True, # bool, default True
    observed = True, # bool, default True
    dropna = True # bool, default True
) → pandas.api.typing.DataFrameGroupBy
"""
マッパー（関数や辞書など）またはSeriesの列を用いてDataFrameをグループ化する。

groupby操作は、オブジェクトの分割（splitting）、関数の適用（applying）、および結果の結合（combining）の組み合わせを伴う。
これは、大量のデータをグループ化し、それらのグループに対して演算を計算するために使用できる。
"""
```
### [polars.DataFrame.group_by](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.group_by.html)
```python
polars.DataFrame.group_by

DataFrame.group_by(
    *by: IntoExpr | Iterable[IntoExpr],
    maintain_order: bool = False,
    **named_by: IntoExpr,
) → GroupBy
"""
groupby操作を開始する。
"""
```

## 18. 記述統計
pandasでもpolarsでもdescribeが利用できます。
ただし, polarsのdescribeは将来的に挙動が変更される可能性があるため, 定期実行を行う処理などには組み込まないようにと公式から注意があります。
なお, polarsのdescribeでは文字列も集計対象で, 最小値はアルファベット順で一番最初, 最大値は一番最後の値となります。

また, **polarsではNaNが含まれる列の集計では, 平均値や標準偏差はNaNになります。**

これは, polarsではNaNは欠損値ではなく有効な値とみなされるからです。

```python
# pandas
df_pd.describe()

# [out]
	契約年月日	年齢	性別	購入年月日	購買金額
count	1.000000e+04	10000.000000	10000.000000	9.900000e+03	9900.000000
mean	2.022051e+07	48.714800	1.498000	2.025066e+07	22289.060606
std	1.398674e+04	17.922683	0.500021	3.478658e+02	15827.597946
min	2.020010e+07	18.000000	1.000000	2.025010e+07	-4995.000000
25%	2.021040e+07	33.000000	1.000000	2.025033e+07	8724.000000
50%	2.022062e+07	49.000000	1.000000	2.025063e+07	21808.000000
75%	2.023092e+07	64.000000	2.000000	2.025093e+07	36058.250000
max	2.024123e+07	80.000000	2.000000	2.025123e+07	49986.000000

```

```python
# polars
df_pl.describe()

# [out]
shape: (9, 9)
statistic	index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額
str	f64	str	f64	f64	f64	str	f64	f64
"count"	10000.0	"10000"	10000.0	10000.0	10000.0	"9900"	9900.0	10000.0
"null_count"	0.0	"0"	0.0	0.0	0.0	"100"	100.0	0.0
"mean"	4999.5	null	2.0221e7	48.7148	1.498	null	2.0251e7	NaN
"std"	2886.89568	null	13986.740832	17.922683	0.500021	null	347.865757	NaN
"min"	0.0	"C0001"	2.0200103e7	18.0	1.0	"中国"	2.0250101e7	-4995.0
"25%"	2500.0	null	2.0210404e7	33.0	1.0	null	2.0250329e7	8890.0
"50%"	5000.0	null	2.0220615e7	49.0	1.0	null	2.0250629e7	22115.0
"75%"	7499.0	null	2.0230916e7	64.0	2.0	null	2.0250929e7	36462.0
"max"	9999.0	"C2000"	2.0241231e7	80.0	2.0	"関東"	2.0251231e7	49986.0
```

以下, 公式の関数説明です。
### [pandas.DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
```python
DataFrame.describe(
    percentiles = None, # list-like of numbers, optional
    include = None, # ‘all’, list-like of dtypes or None (default), optional
    exclude = None # list-like of dtypes or None (default), optional
) → Series or DataFrame
"""
記述統計（要約統計量）を生成する。

記述統計には、データセットの分布の中心傾向（中央値や平均値など）、分散（ばらつき）、および形状を要約する統計量が含まれ、NaN値（欠損値）は除外される。
数値データとオブジェクト（文字列など）データの両方のSeries、および異なるデータ型が混在するDataFrameの列セットを分析できる。
出力結果は提供されたデータの内容によって異なる。
"""
```

### [polars.DataFrame.descirbe](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.describe.html)
```python
DataFrame.describe(
    percentiles: Sequence[float] | float | None = (0.25, 0.5, 0.75),
    *,
    interpolation: QuantileMethod = 'nearest', # {‘nearest’, ‘higher’, ‘lower’, ‘midpoint’, ‘linear’, ‘equiprobable’}
) → DataFrame
"""
データフレームの記述統計。

describe の出力結果は、将来にわたって不変（安定的）であることを保証していない。
polars開発チームが有益であると判断した統計量が表示され、これは将来的に更新（変更）される可能性がある。
そのため、プログラム内（自動化された処理など）で describe を使用することは推奨されない（対話的なデータ探索・確認での使用にとどめるべき）。
"""
```

## 19. 関数の適用
pandasではapply関数, polarsではwith_columnsやselectでmap_elementsを使用することにより実現できます。
ただし, いずれも実行速度が遅くなるため, 必要性を見極めることが重要。
```python
# 適用する関数
def label_sex(sex):
    if sex == 1: return "男性"
    elif sex == 2: return "女性"
```

```python
# pandas
df_pd.assign(
    性別_男女=df_pd["性別"].apply(label_sex)
)

# [out]
	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額	性別_男女
0	C0699	20230711	75	2	関東	20250824.0	28044.0	女性
1	C0965	20240723	36	1	関東	20251116.0	22883.0	男性
2	C1476	20241223	44	2	中部	20250119.0	9063.0	女性
3	C0661	20240212	73	2	北海道	20250803.0	28649.0	女性
4	C1128	20201110	50	2	近畿	20250424.0	29828.0	女性
...	...	...	...	...	...	...	...	...
9995	C0788	20200618	25	2	中部	20250129.0	5219.0	女性
9996	C0907	20240712	36	2	九州	20250118.0	18481.0	女性
9997	C0864	20220714	69	2	東北	20250901.0	29536.0	女性
9998	C1404	20220121	19	2	関東	20250624.0	1096.0	女性
9999	C0667	20200722	68	2	東北	20250330.0	32953.0	女性
10000 rows × 8 columns
```

```python
# polars
df_pl.with_columns(
    pl.col("性別").map_elements(label_sex, return_dtype=pl.String).alias("性別_男女")
)

# [out]
shape: (10_000, 9)
index	顧客番号	契約年月日	年齢	性別	出身地	購入年月日	購買金額	性別_男女
u32	str	i64	i64	i64	str	i64	f64	str
0	"C0699"	20230711	75	2	"関東"	20250824	28044.0	"女性"
1	"C0965"	20240723	36	1	"関東"	20251116	22883.0	"男性"
2	"C1476"	20241223	44	2	"中部"	20250119	9063.0	"女性"
3	"C0661"	20240212	73	2	"北海道"	20250803	28649.0	"女性"
4	"C1128"	20201110	50	2	"近畿"	20250424	29828.0	"女性"
…	…	…	…	…	…	…	…	…
9995	"C0788"	20200618	25	2	"中部"	20250129	5219.0	"女性"
9996	"C0907"	20240712	36	2	"九州"	20250118	18481.0	"女性"
9997	"C0864"	20220714	69	2	"東北"	20250901	29536.0	"女性"
9998	"C1404"	20220121	19	2	"関東"	20250624	1096.0	"女性"
9999	"C0667"	20200722	68	2	"東北"	20250330	32953.0	"女性"
```

polarsではreturn_dtypeを指定することが強く推奨されます。
また, 例えば以下のようにできるだけpolarsの既存の表現で処理を行うことが進められています。

```python
# できるだけpolarsの既存の表現で処理できるようにする
df_pl.with_columns(
    pl.when(pl.col("性別") == 1).then(pl.lit("男性"))
      .when(pl.col("性別") == 2).then(pl.lit("女性"))
    .alias("性別_男女")
)
# 出力は割愛
```

以下, 公式の関数の説明です。
### [pandas.DataFrame.apply](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html)
```python
DataFrame.apply(
    func, # function
    axis = 0, # {0 or ‘index’, 1 or ‘columns’}, default 0
    raw = False, # bool, default False
    result_type = None, # {‘expand’, ‘reduce’, ‘broadcast’, None}, default None
    args = (), # tuple
    by_row = 'compat', # False or “compat”, default “compat”
    engine = None, # decorator or {‘python’, ‘numba’}, optional
    engine_kwargs = None, # dict
    **kwargs
) → Series or DataFrame
"""
DataFrameの軸（axis）に沿って関数を適用する。

関数に渡されるオブジェクトは、インデックスがDataFrameのインデックス（axis=0）またはDataFrameの列（axis=1）のいずれかであるSeriesオブジェクト。
デフォルト（result_type=None）では、最終的な戻り値の型は適用された関数の戻り値の型から推論される。
それ以外の場合は、result_type 引数に依存する。
適用された関数の戻り値の型は、Seriesオブジェクトに関数を適用した後に得られる最初の計算結果に基づいて推論される。
"""
```

### [polars.Expr.map_elements](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_elements.html)
```python
polars.Expr.map_elements

Expr.map_elements(
    function: Callable[[Any], Any],
    return_dtype: polarsDataType | DataTypeExpr | None = None,
    *,
    skip_nulls: bool = True,
    pass_name: bool = False,
    strategy: MapElementsStrategy = 'thread_local' # {‘thread_local’, ‘threading’}
) → Expr
"""
列の各要素に対して、カスタム関数（ユーザー定義関数 / UDF）をマッピング（適用）する。

このメソッドは、ネイティブのExpressions APIよりも大幅に低速。
ほかの方法でロジックを実装できない場合にのみ使用すること。
"""
```

## おわりに
pandasからpolarsにすることで前処理のサイクルのスピードが速くなるだけでなく, Amazon SageMakerなどのクラウドコンピューティングサービスを使っている人なら, 利用するインスタンスのグレードを下げて料金を節約できる可能性もあります。

実用性が高いだけでなく非常に楽しいライブラリなので, ぜひこの機会に移行を検討してみてください。

## 参考文献
https://qiita.com/_jinta/items/fac13f09e8e8a5769b79
↑ (最終更新日: 2023/11/06) polarsとpandasのコードだけでなく基本的な仕様の違いにも触れながら, polarsに乗り換えることによるメリットを理解できる記事。他の記事へのリンクも非常に充実しています。また, 文章が非常に読みやすいです。

https://qiita.com/nkay/items/9cfb2776156dc7e054c8
↑ (最終更新日: 2024/08/06) 非常に丁寧に, 網羅的にpolarsで利用できる機能を紹介した記事。大量のコード例が載っているので, 自分の行いたい処理を実現しているコード例に出会えるはず。

https://zenn.dev/bee2/articles/e8623a603752ff
↑ (最終更新日: 2023/05/19) 簡潔にpandasとpolarsの関数の対応を確認することができます。

https://knowledge.insight-lab.co.jp/aidx/-python-comparison-of-pandas-and-polars
↑ (最終更新日: 2024/12/25) 1.6GBのKaggleデータを使って, pandasとpolarsのコードの書き方および処理速度を比較した非常に実践的な記事。

https://qiita.com/yknoguchi/items/be49932d7f9061333954
↑ (最終更新日: 2023/12/12) 列や行に対しての操作が分かりやすく簡潔に説明されています。

https://qiita.com/Johannyjm/items/125736c186507a12eda0

↑ (最終更新日: 2023/12/25) polars特有のエクスプレッション (pl.Expr) について詳細に解説した記事。