import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import tqdm, trange
import copy
import matplotlib.pyplot as plt
import neal
import pandas as pd

# 実機QPU用ライブラリは未インストールでもOK
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
except ImportError:
    DWaveSampler = None
    EmbeddingComposite = None

class FMQAOptimizer:
    def __init__(self, domain, bits=6, k=6, epochs=1000, verbose=True, use_qpu=False, qpu_kwargs=None):
        self.domain = domain
        self.bits = bits
        self.k = k
        self.epochs = epochs
        self.verbose = verbose
        self.use_qpu = use_qpu
        self.qpu_kwargs = qpu_kwargs if qpu_kwargs is not None else {}

        self.var_types = []
        self.bounds = []
        self.discrete_vals = []
        self.total_bits = 0
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
        binvec = []
        for val, typ, bound, dvals in zip(x, self.var_types, self.bounds, self.discrete_vals):
            if typ == 'continuous':
                a, b = bound
                v = int(round((val - a) / (b - a) * (2 ** self.bits - 1)))
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
        x = []
        pos = 0
        for typ, bound, dvals in zip(self.var_types, self.bounds, self.discrete_vals):
            if typ == 'continuous':
                bits = binvec[pos:pos + self.bits]
                pos += self.bits
                intval = int(''.join(map(str, bits)), 2)
                a, b = bound
                val = a + (b - a) * intval / (2 ** self.bits - 1)
                x.append(val)
            elif typ == 'discrete':
                nbits = int(np.ceil(np.log2(len(dvals))))
                bits = binvec[pos:pos + nbits]
                pos += nbits
                intval = int(''.join(map(str, bits)), 2)
                idx = np.clip(intval, 0, len(dvals) - 1)
                x.append(dvals[idx])
        return list(x)   # ここだけ修正（np.array→list）

    class TorchFM(nn.Module):
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
            np_v = self.v.detach().numpy().copy()
            np_w = self.w.detach().numpy().copy()
            np_w0 = self.w0.detach().numpy().copy()
            return np_v, np_w, float(np_w0)

    def _train_fm(self, x, y):
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
                loss = loss_func(pred_y, y_train)
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
        v, w, w0 = fm_model.get_parameters()
        d, k = v.shape
        Q = np.zeros((d, d))
        for i in range(d):
            Q[i, i] = w[i]
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
        # --- 実機QPUとnealで切り替え ---
        if self.use_qpu and DWaveSampler is not None:
            print("[FMQA] D-Wave QPUで実行します")
            sampler = EmbeddingComposite(DWaveSampler(**self.qpu_kwargs))
            sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
            sample = sampleset.first.sample
        else:
            print("[FMQA] neal（シミュレーテッドアニーリング）で実行します")
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
            sample = sampleset.first.sample
        x_bin = np.array([sample[i] for i in range(d)])
        return x_bin

    def run_optimization(self, func, n_init=5, n_iter=20, plot=True, csv_path="history.csv"):
        rng = np.random.default_rng()
        x_data = []
        y_data = []
        X_continuous = []
        for _ in range(n_init):
            x_sample = []
            for typ, bound, dvals in zip(self.var_types, self.bounds, self.discrete_vals):
                if typ == 'continuous':
                    a, b = bound
                    x_sample.append(rng.uniform(a, b))
                elif typ == 'discrete':
                    x_sample.append(rng.choice(dvals))
            binvec = self.encode(x_sample)
            x_data.append(binvec)
            y_val = func(list(x_sample))      # ← ここ
            y_data.append(y_val)
            X_continuous.append(np.array(x_sample))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        X_continuous = np.array(X_continuous)
        y_hist = list(y_data)
        for it in trange(n_iter, disable=not self.verbose):
            fm_model = self._train_fm(x_data, y_data)
            x_bin = self._anneal(fm_model)
            x_new = self.decode(x_bin)
            # 重複対策
            while any((np.all(x_bin == row) for row in x_data)):
                flip = rng.integers(0, len(x_bin))
                x_bin[flip] ^= 1
                x_new = self.decode(x_bin)
            y_new = func(list(x_new))         # ← ここ
            x_data = np.vstack((x_data, x_bin))
            y_data = np.append(y_data, y_new)
            X_continuous = np.vstack((X_continuous, x_new))
            y_hist.append(y_new)
            if self.verbose:
                print(f"[FMQA iter {it}] x={x_new}, y={y_new:.4f}, best={np.min(y_data):.4f}")
        best_idx = np.argmin(y_data)
        best_x = self.decode(x_data[best_idx])
        best_y = y_data[best_idx]
        if plot:
            plt.plot(y_hist, 'o-')
            plt.xlabel('iteration')
            plt.ylabel('f(x)')
            plt.title('FMQA Optimization history')
            plt.show()
        # === CSV出力 ===
        columns = [v["name"] for v in self.domain] + ["y"]
        df = pd.DataFrame(np.column_stack([X_continuous, y_data]), columns=columns)
        df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"[INFO] Optimization history saved to {csv_path}")
        return best_x, best_y, df

# ---- USAGE EXAMPLE ----
if __name__ == '__main__':
    # ▼ ドメイン定義
    domain = [
        {'name': 'temp',    'type': 'continuous', 'domain': (500, 900)},         # 温度
        {'name': 'time',    'type': 'continuous', 'domain': (1, 8)},             # 時間
        {'name': 'press',   'type': 'continuous', 'domain': (0.5, 3)},           # 圧力
        {'name': 'catalyst','type': 'discrete',   'domain': ('Ni', 'Pt', 'Fe')}, # 触媒
        {'name': 'method',  'type': 'discrete',   'domain': ('A法', 'B法', 'C法')}, # 処理法
    ]

    # ▼ 目的関数（ダミーの複雑なブラックボックス想定）
    def func(x):
        temp, time, press, catalyst, method = x
        # 強度（多峰性と組み合わせ依存）
        strength = -((temp-720)**2)/800 - ((time-5)**2)/4 - ((press-1.5)**2)*2
        if catalyst == 'Pt':
            strength += 2.0
        if catalyst == 'Ni' and method == 'B法':
            strength += 1.5
        if method == 'C法':
            strength += np.sin(press*2) * 1.2
        # コスト（温度高・時間長・Ptはコスト高、C法は安い）
        cost = ((temp-500)/400)*2 + (time-1)/7 + (1 if catalyst=='Pt' else 0.5) - (0.7 if method=='C法' else 0)
        # 評価値＝強度 - コスト（最大化したい）
        return -(strength - cost)  # FMQA仕様で「最小化」

    # ▼ 最適化実行
    optimizer = FMQAOptimizer(
        domain, bits=5, k=4, epochs=400, verbose=True, use_qpu=False)
    best_x, best_y, history_df = optimizer.run_optimization(
        func, n_init=10, n_iter=20, csv_path="history_material.csv"
    )
    print("Best X:", best_x)
    print("Best (強度-コスト):", -best_y)  # 「-best_y」で“本来の最大化値”を表示











