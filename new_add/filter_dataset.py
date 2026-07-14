"""
筛选数据集，保留指定数量的样本。

用法：
python filter_dataset.py \
  --train_dir /data/project_m2f/Mask2Former/datasets/balanced_dataset/train \
  --val_dir /data/project_m2f/Mask2Former/datasets/balanced_dataset/val \
  --train_count 2520 \
  --val_count 280
"""
import argparse
import json
import os
import random


def filter_split(split_dir, target_count, seed=42):
    """筛选一个分割，保留指定数量的样本。"""
    image_dir = os.path.join(split_dir, "encoded_views")
    mask_dir = os.path.join(split_dir, "masks")
    class_map_path = os.path.join(split_dir, "class_map.json")

    # 获取所有 PNG 文件
    all_images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith('.png')
    ])

    print(f"  原始图片数: {len(all_images)}")
    print(f"  目标数量: {target_count}")

    if len(all_images) <= target_count:
        print(f"  图片数量已满足要求，跳过")
        return

    # 随机采样
    random.seed(seed)
    selected = set(random.sample(all_images, target_count))
    to_delete = [f for f in all_images if f not in selected]

    print(f"  保留: {len(selected)} 张")
    print(f"  删除: {len(to_delete)} 张")

    # 删除图片和对应的掩码
    deleted_count = 0
    for img_file in to_delete:
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(mask_path):
            os.remove(mask_path)

        deleted_count += 1
        if deleted_count % 1000 == 0:
            print(f"    已删除 {deleted_count}/{len(to_delete)}...")

    print(f"  已删除 {deleted_count} 个文件")

    # 更新 class_map.json
    if os.path.exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)

        new_class_map = {k: v for k, v in class_map.items() if k in selected}

        with open(class_map_path, "w", encoding="utf-8") as f:
            json.dump(new_class_map, f, ensure_ascii=False, indent=2)

        print(f"  class_map.json 已更新: {len(new_class_map)} 条记录")


def main():
    parser = argparse.ArgumentParser(description="筛选数据集")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="训练集目录")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="验证集目录")
    parser.add_argument("--train_count", type=int, default=2520,
                        help="训练集目标数量")
    parser.add_argument("--val_count", type=int, default=280,
                        help="验证集目标数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    print("=" * 50)
    print("筛选训练集")
    print("=" * 50)
    filter_split(args.train_dir, args.train_count, args.seed)

    print()
    print("=" * 50)
    print("筛选验证集")
    print("=" * 50)
    filter_split(args.val_dir, args.val_count, args.seed)

    print()
    print("筛选完成!")


if __name__ == "__main__":
    main()
