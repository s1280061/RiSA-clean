# -*- coding: utf-8 -*-
"""
make_fired_image_index.py
- 発火画像（軌跡付きで保存した JPG）の保存先パスを JSON にまとめます。
- 走査対象: C:\Users\s1280\Desktop\SHRP2rawdata\{3,4,5,6}\new_divided\scene_XXX\
- 出力:
  1) 全体インデックス: C:\Users\s1280\Desktop\SHRP2rawdata\fired_images_index.json
  2) 各シーン:         ...\new_divided\scene_XXX\scene_XXX_fired_images.json
- オプション:
  --augment-covla を付けると、scene_XXX_covla.json が存在する場合に
  fired_images を追記（上書きではなく追加/重複排除）します。
"""

import os
import re
import json
import argparse
from datetime import datetime

# ==== 固定ルート（必要なら調整） ====
ROOT = r"C:\Users\s1280\Desktop\SHRP2rawdata"

# ---- ユーティリティ ----
def natural_key(s: str):
    """人間にとって自然な並び順（数字を数値としてソート）"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def dump_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json_or_none(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def collect_scene_images(scene_dir: str):
    # scene_dir: ...\new_divided\scene_XXX
    if not os.path.isdir(scene_dir):
        return []
    imgs = []
    for name in os.listdir(scene_dir):
        if name.lower().endswith(".jpg"):
            imgs.append(os.path.join(scene_dir, name))
    imgs.sort(key=natural_key)
    return imgs

def augment_covla_json(scene_dir: str, fired_images: list):
    """scene_XXX_covla.json が隣接フォルダ（new_divided と同じ階層）にある前提で追記を試みる。"""
    # video と同じフォルダ（= new_divided）と同階層に JSON を作っている前提なら：
    # 実際の保存先は run スクリプトの base_dir (= video と同じフォルダ) なので、
    # scene_XXX_covla.json は new_divided と同じ階層にあります。
    new_divided = os.path.dirname(scene_dir)          # ...\new_divided
    base_dir     = os.path.dirname(new_divided)       # ...\ (3/4/5/6)
    scene_name   = os.path.basename(scene_dir)        # scene_XXX
    covla_path   = os.path.join(new_divided, f"{scene_name}_covla.json")
    # もし run スクリプトが new_divided と同じフォルダに出していない場合は、
    # ここを base_dir に切り替えてください。
    if not os.path.exists(covla_path):
        # 念のため base_dir 側も探す
        fallback = os.path.join(base_dir, f"{scene_name}_covla.json")
        if os.path.exists(fallback):
            covla_path = fallback
        else:
            return False, None

    data = load_json_or_none(covla_path)
    if data is None:
        return False, covla_path

    # data がリスト（フレームごとの記録配列）の場合と、辞書の場合の両対応
    # どちらでも top-level に "fired_images"（重複なし）を付与
    fired_set = set(fired_images)

    if isinstance(data, list):
        # 既存の各レコードに image_path が入っている想定なので、それもマージして重複除去
        old_paths = []
        for rec in data:
            p = rec.get("image_path")
            if isinstance(p, str):
                old_paths.append(p)
        fired_set.update(old_paths)
        out_obj = {
            "records": data,
            "fired_images": sorted(fired_set, key=natural_key),
        }
    elif isinstance(data, dict):
        prev = data.get("fired_images", [])
        if isinstance(prev, list):
            fired_set.update(prev)
        data["fired_images"] = sorted(fired_set, key=natural_key)
        out_obj = data
    else:
        # 想定外形式 → ラップして保存
        out_obj = {
            "raw": data,
            "fired_images": sorted(fired_set, key=natural_key),
        }

    dump_json(covla_path, out_obj)
    return True, covla_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment-covla", action="store_true",
                        help="scene_XXX_covla.json へ fired_images を追記します")
    args = parser.parse_args()

    bases = [(3, 268), (4, 237), (5, 187), (6, 337)]
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bases": {}
    }

    for base_num, _max in bases:
        base_root = os.path.join(ROOT, str(base_num))
        new_divided = os.path.join(base_root, "new_divided")
        if not os.path.isdir(new_divided):
            print(f"[WARN] {new_divided} が見つかりません。スキップします。")
            continue

        base_entry = {
            "root": new_divided,
            "total_images": 0,
            "scenes": {}
        }

        for i in range(_max + 1):
            scene_name = f"scene_{i:03d}"
            scene_dir = os.path.join(new_divided, scene_name)
            imgs = collect_scene_images(scene_dir)
            if not imgs:
                continue

            base_entry["total_images"] += len(imgs)
            base_entry["scenes"][scene_name] = {
                "count": len(imgs),
                "first": imgs[0],
                "last": imgs[-1],
                "images": imgs
            }

            # シーン別のインデックスも出す
            per_scene_out = os.path.join(scene_dir, f"{scene_name}_fired_images.json")
            dump_json(per_scene_out, {
                "scene": scene_name,
                "count": len(imgs),
                "images": imgs
            })

            # 既存の scene_XXX_covla.json を拡張（希望時）
            if args.augment_covla:
                ok, path = augment_covla_json(scene_dir, imgs)
                if ok:
                    print(f"[OK] {scene_name}: covla を追記しました → {path}")
                else:
                    if path:
                        print(f"[WARN] {scene_name}: covla 読み込み失敗 → {path}")
                    # 見つからない場合は無言スキップ

        summary["bases"][str(base_num)] = base_entry

    # 全体インデックスを ROOT 直下へ
    master_path = os.path.join(ROOT, "fired_images_index.json")
    dump_json(master_path, summary)
    print(f"\n✅ 出力しました: {master_path}")
    # ベースごとの集計も軽く表示
    for bkey, entry in summary["bases"].items():
        print(f"  - base {bkey}: {entry['total_images']} images, {len(entry['scenes'])} scenes")

if __name__ == "__main__":
    main()
