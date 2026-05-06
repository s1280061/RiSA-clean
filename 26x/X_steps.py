import pandas as pd

# ファイルのパス
file_path = r"C:\Users\s1280\Downloads\traj_dataset_p30_f45_s5_px_with_ego_speed_val.csv"

try:
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)

    # 'id' 列の重複を除いた（ユニークな）個数をカウント
    # 例：id 11 が複数あっても 1 とカウントされます
    unique_id_count = df['id'].nunique()

    print(f"データの総行数: {len(df)}")
    print(f"IDの個数（車両数）: {unique_id_count} 台")

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりませんでした。パスが正しいか確認してください。\n{file_path}")
except KeyError:
    print("エラー: CSVファイル内に 'id' という名前の列が見つかりませんでした。")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")