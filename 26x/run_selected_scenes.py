# run_selected_scenes.py
"""
複数ディレクトリから特定シーンを処理するバッチスクリプト
使い方: python run_selected_scenes.py
"""

import os
import subprocess
from pathlib import Path

# ========== 設定 ==========
BASE_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata"
MAIN_SCRIPT = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\2-25_all_scripts_vx2xxx.py"

# 処理するシーン（フォーマット: (サブディレクトリ番号, シーン番号)）
SCENES_TO_PROCESS = [
    (5, 19),  # 5/scene_019
    (5, 20),  # 5/scene_020
    (3, 7),  # 3/scene_007
    (6, 105),  # 6/scene_105
    (6, 104),  # 6/scene_104
    (3, 9),  # 3/scene_009
    (4, 22),  # 4/scene_022
    (4, 36),  # 4/scene_036
    (4, 117),  # 4/scene_117
    (4, 40),  # 4/scene_040
    (4, 73),  # 4/scene_073
    (4, 110),  # 4/scene_110
    (4, 168),  # 4/scene_168
    (5, 69),  # 5/scene_069
    (6, 103),  # 6/scene_103
]


# ========== ファイルパス構築 ==========
def build_paths(subdir, scene_num):
    """
    指定されたサブディレクトリとシーン番号からファイルパスを構築

    Args:
        subdir: サブディレクトリ番号 (3, 4, 5, 6)
        scene_num: シーン番号 (7, 19, 105など)

    Returns:
        dict: video_path, csv_path, output_dirを含む辞書
    """
    scene_str = f"{scene_num:03d}"  # 3桁ゼロ埋め (例: 7 -> 007)

    subdir_path = os.path.join(BASE_DIR, str(subdir))
    video_dir = os.path.join(subdir_path, "new_divided")
    csv_dir = os.path.join(subdir_path, "csv_divided")

    video_path = os.path.join(video_dir, f"scene_{scene_str}.mp4")
    csv_path = os.path.join(csv_dir, f"scene_{scene_str}.csv")

    return {
        'subdir': subdir,
        'scene_num': scene_num,
        'scene_str': scene_str,
        'video_path': video_path,
        'csv_path': csv_path,
        'output_dir': video_dir
    }


# ========== ファイル存在チェック ==========
def check_files_exist(paths):
    """入力ファイルの存在を確認"""
    errors = []

    if not os.path.exists(paths['video_path']):
        errors.append(f"❌ 動画ファイルが見つかりません: {paths['video_path']}")

    if not os.path.exists(paths['csv_path']):
        errors.append(f"❌ CSVファイルが見つかりません: {paths['csv_path']}")

    return errors


# ========== メイン処理 ==========
def main():
    print("=" * 70)
    print("🚀 選択シーンのバッチ処理開始")
    print("=" * 70)

    # 処理対象のシーンを整理
    scenes_info = []
    missing_files = []

    for subdir, scene_num in SCENES_TO_PROCESS:
        paths = build_paths(subdir, scene_num)
        errors = check_files_exist(paths)

        if errors:
            missing_files.extend(errors)
            print(f"⚠️ {subdir}/{scene_num:03d}: ファイルが不足")
        else:
            scenes_info.append(paths)

    # ファイルチェック結果
    if missing_files:
        print("\n" + "=" * 70)
        print("⚠️ 以下のファイルが見つかりません:")
        print("=" * 70)
        for error in missing_files:
            print(error)
        print()
        response = input("見つかったファイルのみ処理を続行しますか？ (y/n): ")
        if response.lower() != 'y':
            print("❌ 処理を中断しました")
            return

    if not scenes_info:
        print("❌ 処理可能なシーンがありません")
        return

    # 処理予定のシーン一覧
    print(f"\n📊 処理予定のシーン: {len(scenes_info)}個")
    print("=" * 70)
    for info in scenes_info:
        print(f"  {info['subdir']}/{info['scene_str']} - {os.path.basename(info['video_path'])}")

    # 確認
    print("=" * 70)
    response = input("\n実行しますか？ (y/n): ")
    if response.lower() != 'y':
        print("❌ 処理を中断しました")
        return

    # 各シーンを順次処理
    success_count = 0
    fail_count = 0
    failed_scenes = []

    for i, info in enumerate(scenes_info, 1):
        subdir = info['subdir']
        scene_str = info['scene_str']

        print("\n" + "=" * 70)
        print(f"▶️ [{i}/{len(scenes_info)}] {subdir}/scene_{scene_str} を処理中...")
        print("=" * 70)
        print(f"📹 動画: {info['video_path']}")
        print(f"📊 CSV:  {info['csv_path']}")
        print(f"📁 出力: {info['output_dir']}")

        # メインスクリプトを実行
        cmd = [
            "python", MAIN_SCRIPT,
            "--video", info['video_path'],
            "--csv", info['csv_path']
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ {subdir}/scene_{scene_str} 完了")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {subdir}/scene_{scene_str} エラー")
            fail_count += 1
            failed_scenes.append(f"{subdir}/scene_{scene_str}")

            # エラー時の対応
            response = input("\n続行しますか？ (y/n): ")
            if response.lower() != 'y':
                print("❌ バッチ処理を中断しました")
                break
        except KeyboardInterrupt:
            print("\n⏹️ ユーザーによって中断されました")
            fail_count += 1
            failed_scenes.append(f"{subdir}/scene_{scene_str}")
            break

    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 バッチ処理結果")
    print("=" * 70)
    print(f"✅ 成功: {success_count}件")
    print(f"❌ 失敗: {fail_count}件")
    print(f"📁 合計: {success_count + fail_count}件")

    if failed_scenes:
        print("\n失敗したシーン:")
        for scene in failed_scenes:
            print(f"  - {scene}")

    print("\n🏁 バッチ処理完了")


if __name__ == "__main__":
    main()