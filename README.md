# r7_inference_analyze
## パッケージ概要
深江用．mmdetectionのtest.pyによる推論処理を解析するためのパッケージ．
## 同梱物
- README.md : 本ファイル．説明書．  
- analyze.py : 推論処理解析用スクリプト  
## ディレクトリ構成
- r7_inference_analyze : ルートディレクトリ
  - src : スクリプトソースコード配置場所
  - data : 解析対象データ群配置場所
## 使い方
### analyze.py
1. dataディレクトリ内に専用の作業ディレクトリを作成する（例：data/ann01）
2. 作業ディレクトリ内にresultsディレクトリを作成する.resultsディレクトリには推論結果であるmmdetectionのtest.pyが出力したjsonファイル群を全てコピーする
3. 作業ディレクトリ内にvalidディレクトリを作成する．validディレクトリにはアノテーションデータであるjsonファイルと元画像全てをコピーする
4. 作業ディレクトリ内にvisディレクトリを作成する．
5. ルートディレクトリ上でsrc/analyze.pyを引数をつけて実行する．実行例を以下に示す．
```
python ./src/analyze.py --gt ./data/ann01/valid/_annotations.coco.json --det_dir ./data/ann01/results --img_dir ./data/ann01/valid --vis_dir ./data/ann01/vis --output ./data/ann01/report.txt
```
6. report.txtにはTP, FP, FN，適合率（Precision），再現率（Recall）がまとめられている．visディレクトリには元画像に対してアノテーションのbboxを緑，推論結果のbboxを赤で描画した結果が保存されている．
