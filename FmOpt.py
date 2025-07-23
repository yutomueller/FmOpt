import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import trange
import copy
import matplotlib.pyplot as plt
import neal
import pandas as pd

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
except ImportError:
    DWaveSampler = None
    EmbeddingComposite = None

class FMQAOptimizer:
    """
    FM(QA)最適化クラス
    - 連続/離散混合変数対応
    - ファクタリゼーションマシン＋アニーリング
    - 重複探索点は「1ビットだけ反転」で回避
    """
    def __init__(self, domain, bits=6, k=6, epochs=1000, verbose=True, use_qpu=False, qpu_kwargs=None, l2_lambda=0.0):
        """
        各種パラメータをセット
        - domain: 各変数の仕様リスト
        - bits: 連続値のビット数（離散化粒度）
        - k: FMの潜在因子次元
        - epochs: FM学習エポック
        - l2_lambda: L2正則化強度
        """
        self.domain = domain
        self.bits = bits
        self.k = k
        self.epochs = epochs
        self.verbose = verbose
        self.use_qpu = use_qpu
        self.qpu_kwargs = qpu_kwargs if qpu_kwargs is not None else {}
        self.l2_lambda = l2_lambda

        self.var_types = []
        self.bounds = []
        self.discrete_vals = []
        self.total_bits = 0
        # 各変数をビット空間で定義
        for var in domain:
            if var['type'] == 'continuous':
                self.var_types.append('continuous')
                self.bounds.append(var['domain'])
                self.discrete_vals.append(None)
                self.total_bits += bits
            elif var['type'] == 'discrete':
                self.var_types.append('discrete')
                self.bounds.append(None)
                self.discrete_vals.append(list(var['domain']))
                self.total_bits += int(np.ceil(np.log2(len(var['domain']))))
            else:
                raise ValueError("Unknown variable type: {}".format(var['type']))

    def encode(self, x):
        """
        連続・離散変数を全てビット列にエンコード
        - 連続値: 0-1に正規化→ビット離散化
        - 離散値: インデックスをビット化
        """
        binvec = []
        for val, typ, bound, dvals in zip(x, self.var_types, self.bounds, self.discrete_vals):
            if typ == 'continuous':
                a, b = bound
                norm_val = (val - a) / (b - a)  # 正規化
                v = int(round(norm_val * (2 ** self.bits - 1)))
                v = np.clip(v, 0, 2 ** self.bits - 1)
                bits = [int(b) for b in format(v, '0{}b'.format(self.bits))]
                binvec.extend(bits)
            elif typ == 'discrete':
                idx = dvals.index(val)
                nbits = int(np.ceil(np.log2(len(dvals))))
                bits = [int(b) for b in format(idx, '0{}b'.format(nbits))]
                binvec.extend(bits)
        return np.array(binvec, dtype=int)

    def decode(self, binvec):
        """
        ビット列から元の変数値へ逆変換
        - 連続値: 0-1逆正規化→元スケール復元
        - 離散値: インデックス復元
        """
        x = []
        pos = 0
        for typ, bound, dvals in zip(self.var_types, self.bounds, self.discrete_vals):
            if typ == 'continuous':
                bits = binvec[pos:pos + self.bits]
                pos += self.bits
                intval = int(''.join(map(str, bits)), 2)
                norm_val = intval / (2 ** self.bits - 1)
                a, b = bound
                val = a + (b - a) * norm_val
                x.append(val)
            elif typ == 'discrete':
                nbits = int(np.ceil(np.log2(len(dvals))))
                bits = binvec[pos:pos + nbits]
                pos += nbits
                intval = int(''.join(map(str, bits)), 2)
                idx = np.clip(intval, 0, len(dvals) - 1)
                x.append(dvals[idx])
        return list(x)

    class TorchFM(nn.Module):
        """
        シンプルなファクタリゼーションマシン（FM）モデル
        """
        def __init__(self, d, k):
            super().__init__()
            self.d = d
            self.v = torch.randn((d, k), requires_grad=True)
            self.w = torch.randn((d,), requires_grad=True)
            self.w0 = torch.randn((), requires_grad=True)
        def forward(self, x):
            out_linear = torch.matmul(x, self.w) + self.w0
            out_1 = torch.matmul(x, self.v).pow(2).sum(1)
            out_2 = torch.matmul(x.pow(2), self.v.pow(2)).sum(1)
            out_quadratic = 0.5 * (out_1 - out_2)
            return out_linear + out_quadratic
        def get_parameters(self):
            # numpy配列でパラメータを取り出し
            np_v = self.v.detach().numpy().copy()
            np_w = self.w.detach().numpy().copy()
            np_w0 = self.w0.detach().numpy().copy()
            return np_v, np_w, float(np_w0)

    def _train_fm(self, x, y):
        """
        FMを教師あり回帰で学習
        - L2正則化込み
        - Early stoppingは無し（ベスト検証損失で保存）
        """
        model = self.TorchFM(self.total_bits, self.k)
        optimizer = torch.optim.AdamW([model.v, model.w, model.w0], lr=0.1)
        loss_func = nn.MSELoss()
        x_tensor, y_tensor = (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
        )
        dataset = TensorDataset(x_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)
        min_loss = 1e18
        best_state = copy.deepcopy(model.state_dict())
        for _ in trange(self.epochs, leave=False, disable=not self.verbose):
            for x_train, y_train in train_loader:
                optimizer.zero_grad()
                pred_y = model(x_train)
                mse_loss = loss_func(pred_y, y_train)
                # L2正則化
                l2_reg = 0.0
                for param in [model.v, model.w]:
                    l2_reg += torch.sum(param ** 2)
                loss = mse_loss + self.l2_lambda * l2_reg
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                loss = 0
                for x_valid, y_valid in valid_loader:
                    out_valid = model(x_valid)
                    loss += loss_func(out_valid, y_valid)
                if loss < min_loss:
                    best_state = copy.deepcopy(model.state_dict())
                    min_loss = loss
        model.load_state_dict(best_state)
        return model

    def _anneal(self, fm_model, num_reads=10):
        """
        FMをQUBOに変換してアニーリング解をサンプリング
        - 実機QPUまたはneal（シミュレーテッドアニーリング）
        """
        v, w, w0 = fm_model.get_parameters()
        d, k = v.shape
        Q = np.zeros((d, d))
        # 線形項
        for i in range(d):
            Q[i, i] = w[i]
        # 2次項
        for i in range(d):
            for j in range(i+1, d):
                Q[i, j] += np.dot(v[i], v[j])
        Q_dict = {}
        for i in range(d):
            if Q[i, i] != 0:
                Q_dict[(i, i)] = Q[i, i]
            for j in range(i+1, d):
                if Q[i, j] != 0:
                    Q_dict[(i, j)] = Q[i, j]
        # QPU・nealいずれかでアニーリング
        if self.use_qpu and DWaveSampler is not None:
            print("[FMQA] D-Wave QPUで実行します")
            sampler = EmbeddingComposite(DWaveSampler(**self.qpu_kwargs))
            sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
            sample = sampleset.first.sample
        else:
            if self.verbose:
                print("[FMQA] neal（シミュレーテッドアニーリング）で実行します")
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
            sample = sampleset.first.sample
        # 結果ビット列を返却
        x_bin = np.array([sample[i] for i in range(d)])
        return x_bin

    def run_optimization(
            self, func, n_init=5, n_iter=20, plot=True, csv_path="history.csv", save_model_path=None,
            X_init=None, y_init=None, init_data_path=None):
        """
        最適化のメインループ
        - 既存データ・CSV初期化対応
        - Tabuリスト無し！探索点重複時は「1ビットのみ」ランダム反転
        - best履歴グラフ、CSV保存、モデルセーブ
        """
        rng = np.random.default_rng()
        x_data = []
        y_data = []
        X_continuous = []

        # --- 既存データ・CSVの取り込み ---
        if init_data_path is not None:
            df = pd.read_csv(init_data_path)
            X_init = df.iloc[:, :-1].values
            y_init = df.iloc[:, -1].values

        if X_init is not None and y_init is not None:
            for x_, y_ in zip(X_init, y_init):
                binvec = self.encode(x_)
                x_data.append(binvec)
                y_data.append(y_)
                X_continuous.append(np.array(x_))
            n_additional = max(0, n_init - len(X_init))
        else:
            n_additional = n_init

        # --- ランダム初期点で不足を補完 ---
        for _ in range(n_additional):
            x_sample = []
            for typ, bound, dvals in zip(self.var_types, self.bounds, self.discrete_vals):
                if typ == 'continuous':
                    a, b = bound
                    x_sample.append(rng.uniform(a, b))
                elif typ == 'discrete':
                    x_sample.append(rng.choice(dvals))
            binvec = self.encode(x_sample)
            x_data.append(binvec)
            y_val = func(list(x_sample))
            y_data.append(y_val)
            X_continuous.append(np.array(x_sample))

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        X_continuous = np.array(X_continuous)
        y_hist = list(y_data)
        best_hist = [np.min(y_hist)]
        best_idx = np.argmin(y_data)
        best_fm_model = None

        # === 最適化メインループ ===
        for it in trange(n_iter, disable=not self.verbose):
            fm_model = self._train_fm(x_data, y_data)
            x_bin = self._anneal(fm_model)
            # --- tsudalab/fmqa式の重複回避：1ビットだけランダム反転 ---
            max_attempts = 20  # 最大20回までリトライ（十分広い空間ならこれでOK）
            for attempt in range(max_attempts):
                if not any((np.all(x_bin == row) for row in x_data)):
                    break  # 未探索点ならOK
                flip = rng.integers(0, len(x_bin))
                x_bin[flip] ^= 1  # 1ビットだけ反転
            x_new = self.decode(x_bin)
            y_new = func(list(x_new))
            x_data = np.vstack((x_data, x_bin))
            y_data = np.append(y_data, y_new)
            X_continuous = np.vstack((X_continuous, x_new))
            y_hist.append(y_new)
            # best値の履歴管理
            if y_new < np.min(y_data[:-1]):
                best_idx = len(y_data) - 1
                best_fm_model = copy.deepcopy(fm_model)
            best_hist.append(np.min(y_data))
            if self.verbose:
                print(f"[FMQA iter {it}] x={x_new}, y={y_new:.4f}, best={np.min(y_data):.4f}")

        best_x = self.decode(x_data[best_idx])
        best_y = y_data[best_idx]

        # --- モデル保存（必要なら） ---
        if save_model_path is not None and best_fm_model is not None:
            torch.save(best_fm_model.state_dict(), save_model_path)
            if self.verbose:
                print(f"[INFO] Best FM model saved to {save_model_path}")

        # --- 結果グラフ・CSV出力 ---
        if plot:
            plt.plot(y_hist, 'o-', label="y")
            plt.plot(best_hist, 'r--', label="best so far")
            plt.xlabel('iteration')
            plt.ylabel('f(x)')
            plt.title('FMQA Optimization history')
            plt.legend()
            plt.show()
        columns = [v["name"] for v in self.domain] + ["y"]
        df = pd.DataFrame(np.column_stack([X_continuous, y_data]), columns=columns)
        df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"[INFO] Optimization history saved to {csv_path}")
        return best_x, best_y, df

# ========== 使い方例 ==========

if __name__ == '__main__':
    # --- 最適化対象のドメイン定義 ---
    domain = [
        {'name': 'amt_A', 'type': 'continuous', 'domain': (10, 100)},
        {'name': 'amt_B', 'type': 'continuous', 'domain': (10, 100)},
        {'name': 'temp',  'type': 'continuous', 'domain': (100, 250)},
        {'name': 'method','type': 'discrete',   'domain': ('normal', 'fast', 'eco')},
        ]
    def func(x):
        amt_A, amt_B, temp, method = x
        # sin, cosで周期的に谷を発生
        valley = np.sin((amt_A-10)/20*np.pi) * np.cos((amt_B-10)/20*np.pi)
        # 正負反転とコスト項追加
        cost = 10 - 5 * valley
        if method == 'fast':
            cost -= (0.3 * amt_A - 0.4 * amt_B)
        elif method == 'eco':
            cost -= (0.3 * amt_B - 0.4 * amt_A)
        else:
            cost -= (0.05 * amt_A + 0.05 * amt_B)
        
        return cost


    # --- インスタンス生成 ---
    optimizer = FMQAOptimizer(
        domain, bits=10, k=5, epochs=1000, verbose=True,
        use_qpu=False, l2_lambda=1e-4
    )

    # --- 既存データ例---
    X_init = [
        [10, 80, 180, 'normal'],
        [50, 50, 220, 'eco'],
    ]
    y_init = [func(x) for x in X_init]

    # --- 最適化実行 ---
    best_x, best_y, history_df = optimizer.run_optimization(
        func, n_init=5, n_iter=50,
        csv_path="history_material.csv",
        X_init=X_init, y_init=y_init
    )
    print("Best X:", best_x)
    print("Best Cost:", best_y)











