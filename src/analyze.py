import json
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def calculate_iou(box1, box2, format1='xyxy', format2='xywh'):
    """
    Calculate IoU between two boxes.
    format1: format of box1 ('xyxy' = [x1, y1, x2, y2] or 'xywh' = [x, y, w, h])
    format2: format of box2 ('xyxy' = [x1, y1, x2, y2] or 'xywh' = [x, y, w, h])
    """
    # Convert to xyxy format
    if format1 == 'xywh':
        x1_min, y1_min, w1, h1 = box1
        x1_max = x1_min + w1
        y1_max = y1_min + h1
    else:  # xyxy
        x1_min, y1_min, x1_max, y1_max = box1
    
    if format2 == 'xywh':
        x2_min, y2_min, w2, h2 = box2
        x2_max = x2_min + w2
        y2_max = y2_min + h2
    else:  # xyxy
        x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    # Calculate areas
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='MMDetection Result Analysis')
    parser.add_argument('--gt', type=str, required=True, help='Path to coco annotation json')
    parser.add_argument('--det_dir', type=str, required=True, help='Directory containing detection jsons')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--score_thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--output', type=str, default='analysis_report.txt', help='Output text file')
    parser.add_argument('--img_dir', type=str, default=None, help='Directory containing original images (optional)')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save visualized images (optional)')
    args = parser.parse_args()

    # 1. 正解データの読み込み
    with open(args.gt, 'r') as f:
        coco = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    images = {img['file_name']: img['id'] for img in coco['images']}
    
    # 画像IDごとのGTアノテーションを整理
    gt_by_img = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_by_img:
            gt_by_img[img_id] = []
        gt_by_img[img_id].append(ann)

    # 2. 推論結果の解析
    stats = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in categories.values()}
    
    for img_name, img_id in images.items():
        # 推論結果JSONの特定 (画像名.json を想定)
        det_json_path = os.path.join(args.det_dir, Path(img_name).stem + ".json")
        
        gts = gt_by_img.get(img_id, [])
        dts = []
        
        print(f"[DEBUG] Looking for: {det_json_path}")
        
        if os.path.exists(det_json_path):
            with open(det_json_path, 'r') as f:
                det_data = json.load(f)
                print(f"[DEBUG] Loaded detection data, keys: {det_data.keys() if isinstance(det_data, dict) else 'list'}")
                
                # instances形式の推論結果に対応
                if isinstance(det_data, dict) and 'instances' in det_data:
                    for inst in det_data['instances']:
                        if inst['score'] >= args.score_thr:
                            # bboxは [x1, y1, x2, y2] 形式と仮定
                            dts.append({
                                'bbox': inst['bbox'],
                                'label_id': inst['label_id'],
                                'label_name': inst['label'],
                                'score': inst['score']
                            })
                # MMDetection形式の出力にも対応 (labels, bboxes, scores)
                elif isinstance(det_data, dict) and 'bboxes' in det_data:
                    for i in range(len(det_data['bboxes'])):
                        if det_data['scores'][i] >= args.score_thr:
                            dts.append({
                                'bbox': det_data['bboxes'][i][:4], # [x1, y1, x2, y2]
                                'label_id': det_data['labels'][i],
                                'score': det_data['scores'][i]
                            })
        
        # デバッグ情報を出力
        print(f"\n{'='*80}")
        print(f"Image: {img_name} (ID: {img_id})")
        print(f"GT Objects: {len(gts)}, Detection Objects: {len(dts)}")
        print(f"{'='*80}")

        # 可視化処理: 元画像フォルダが指定されていればGT/DTを描画して保存
        if args.img_dir and args.vis_dir:
            img_path = os.path.join(args.img_dir, img_name)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    # optional font (fallback if not available)
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None

                    # draw GT boxes (COCO xywh)
                    for gt in gts:
                        x, y, w, h = gt['bbox']
                        xy = [x, y, x + w, y + h]
                        draw.rectangle(xy, outline='lime', width=2)
                        cat_name = categories.get(gt['category_id'], str(gt['category_id']))
                        if font:
                            draw.text((x, max(0, y - 10)), cat_name, fill='lime', font=font)

                    # draw Detection boxes (assumed xyxy)
                    for dt in dts:
                        bx = dt['bbox']
                        # ensure 4 values
                        if len(bx) >= 4:
                            x1, y1, x2, y2 = bx[:4]
                            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                            label = categories.get(dt.get('label_id'), dt.get('label_name', ''))
                            score = dt.get('score', 0.0)
                            txt = f"{label}:{score:.2f}"
                            if font:
                                draw.text((x1, max(0, y1 - 10)), txt, fill='red', font=font)

                    os.makedirs(args.vis_dir, exist_ok=True)
                    vis_name = Path(img_name).stem + '_vis.jpg'
                    vis_path = os.path.join(args.vis_dir, vis_name)
                    img.save(vis_path)
                    print(f"[VIS] Saved visualization: {vis_path}")
                except Exception as e:
                    print(f"[VIS] Failed to visualize {img_name}: {e}")
        
        # クラスごとにTP/FP/FNを計算
        for cat_id, cat_name in categories.items():
            current_gts = [g for g in gts if g['category_id'] == cat_id]
            current_dts = [d for d in dts if d['label_id'] == cat_id]
            
            if current_gts or current_dts:
                print(f"\n[{cat_name}] GT: {len(current_gts)}, DT: {len(current_dts)}")
            
            matched_gt = set()
            tp_count = 0
            
            for dt_idx, dt in enumerate(current_dts):
                best_iou = 0
                best_gt_idx = -1
                for i, gt in enumerate(current_gts):
                    if i in matched_gt: continue
                    # dt['bbox'] is [x1, y1, x2, y2], gt['bbox'] is [x, y, w, h]
                    iou = calculate_iou(dt['bbox'], gt['bbox'], format1='xyxy', format2='xywh')
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                # IoU値をデバッグ情報として出力
                result_str = "TP" if (best_iou >= args.iou_thr) else "FP"
                print(f"  DT[{dt_idx}] (score={dt['score']:.4f}) -> Best IoU={best_iou:.4f} [{result_str}]")
                
                if best_iou >= args.iou_thr:
                    tp_count += 1
                    matched_gt.add(best_gt_idx)
                else:
                    stats[cat_name]['FP'] += 1
            
            # マッチされなかったGTをFNとして表示
            fn_count = len(current_gts) - tp_count
            if fn_count > 0:
                print(f"  FN: {fn_count} unmatched GT objects")
            
            stats[cat_name]['TP'] += tp_count
            stats[cat_name]['FN'] += fn_count

    # 3. 結果の出力
    with open(args.output, 'w') as f:
        f.write(f"Detection Analysis Report (IoU Thr: {args.iou_thr}, Score Thr: {args.score_thr})\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10}\n")
        f.write("-" * 60 + "\n")
        
        for name, s in stats.items():
            prec = s['TP'] / (s['TP'] + s['FP']) if (s['TP'] + s['FP']) > 0 else 0
            rec = s['TP'] / (s['TP'] + s['FN']) if (s['TP'] + s['FN']) > 0 else 0
            f.write(f"{name:<15} | {s['TP']:<5} | {s['FP']:<5} | {s['FN']:<5} | {prec:<10.4f} | {rec:<10.4f}\n")

    print(f"Done! Report saved to {args.output}")

if __name__ == '__main__':
    main()