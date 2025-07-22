

# 混合変数（連続＋離散）最適化のためのサロゲートモデル

---

## 概要

FMQAOptimizerは、**連続変数・離散変数・混合変数を含む最適化問題**に対して、
**Factorization Machine (FM)** をサロゲートモデル（代理関数）とし、
そのパラメータをQUBO形式に変換して\*\*量子アニーリング（またはシミュレーテッドアニーリング）\*\*で効率的に最適解を探索するPython実装です。
GpyOptのようなインターフェースでいつも通り使えるようにしています。

* **連続・離散・カテゴリ変数をビット列に自動エンコード**
* **QUBO最適化部は D-Wave 実機 or 古典nealアニーラーで切り替え可能**
* **サロゲート学習はPyTorchで回帰学習**
* **現実のブラックボックス最適化・材料設計・実験計画に最適**

---

## 特徴

* **連続／離散／カテゴリ混合問題を一発で最適化**
* サロゲートは**FM（線形＋2次）モデル**、学習はPyTorch
* QUBO（Quadratic Unconstrained Binary Optimization）への自動変換
* 量子アニーリングまたはシミュレーテッドアニーリング（neal）で最適化

---

## インストール

```bash
pip install numpy pandas matplotlib torch tqdm neal
# D-Wave QPU使う場合のみ（アカウント必要）
pip install dwave-system
```

---

## 使い方

### 1. ドメイン（最適化パラメータ）を定義

```python
domain = [
    {'name': 'temp',    'type': 'continuous', 'domain': (500, 900)},         # 温度
    {'name': 'time',    'type': 'continuous', 'domain': (1, 8)},             # 時間
    {'name': 'press',   'type': 'continuous', 'domain': (0.5, 3)},           # 圧力
    {'name': 'catalyst','type': 'discrete',   'domain': ('Ni', 'Pt', 'Fe')}, # 触媒
    {'name': 'method',  'type': 'discrete',   'domain': ('A法', 'B法', 'C法')}, # 処理法
]
```

### 2. 目的関数（ブラックボックス評価関数）を定義

```python
def func(x):
    temp, time, press, catalyst, method = x
    strength = -((temp-720)**2)/800 - ((time-5)**2)/4 - ((press-1.5)**2)*2
    if catalyst == 'Pt':
        strength += 2.0
    if catalyst == 'Ni' and method == 'B法':
        strength += 1.5
    if method == 'C法':
        strength += np.sin(press*2) * 1.2
    cost = ((temp-500)/400)*2 + (time-1)/7 + (1 if catalyst=='Pt' else 0.5) - (0.7 if method=='C法' else 0)
    return -(strength - cost)  # FMQAは「最小化」仕様
```

### 3. 最適化の実行

```python
optimizer = FMQAOptimizer(
    domain, bits=5, k=4, epochs=400, verbose=True, use_qpu=False)
best_x, best_y, history_df = optimizer.run_optimization(
    func, n_init=10, n_iter=20, csv_path="history_material.csv"
)
print("Best X:", best_x)
print("Best (強度-コスト):", -best_y)
```

* **`bits`**: 各連続変数の量子化ビット数（例:5で32分割）
* **`k`**: FMの2次項因子数（通常3～8程度）
* **`epochs`**: FM回帰モデルのエポック数
* **`use_qpu`**: TrueにするとD-Wave QPU実機でQUBO最適化（アカウント必要）

---

## 出力

* **最適パラメータ (`Best X`) とその評価値 (`Best Y`) を表示**
* **探索履歴（全評価データ）はCSV（例:`history_material.csv`）として保存**
* **進捗グラフも自動表示**

---

## 応用例

* 材料・化学プロセス最適化
* 複雑な実験条件決定
* カテゴリ＋連続混合のブラックボックス最適化全般
* AIハイパーパラメータ探索、組合せ最適設計
* 産業現場の現実制約付き最適化タスク

---

## 注意事項・限界

* FMサロゲートは2次相互作用までしか近似しないため、「強い非線形・高次」は苦手
* QUBO最適化はbit数が多いと指数的に重くなるため、変数数やbitsの設定には注意
* 本質的にベイズ最適化よりも外挿・不確実性探索には弱い
* 連続変数だけの滑らかな最適化にはガウス過程BO等も検討ください


## 参考

tsudalab/fmqa inspired

https://unit.aist.go.jp/g-quat/ja/events/img/CAE_20240509-10/20240510_01_Tamura.pdf
---

