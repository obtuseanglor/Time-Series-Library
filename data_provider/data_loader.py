import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')
from dateutil import tz

class Dataset_SaintHilaire10min(Dataset):
    """
    读取 Saint‑Hilaire 场站 10‑minute SCADA 数据（Excel），
    支持：
      • 按风机筛选 (T2 / T3 / T5 / T6)
      • 剔除 LIST_STOPS 中的停机区间
      • 单/多特征、标准化、时间特征编码
    """
    def __init__(self,
                 root_path,
                 data_path='SaintHilaire_SCADA.xlsx',
                 flag='train',               # 'train' | 'val' | 'test'
                 size=None,                  # [seq_len, label_len, pred_len]
                 turbine_id='T2',
                 features='S',               # 'S' | 'M' | 'MS'
                 target='ACTIVE POWER [Watt]',
                 scale=True,
                 timeenc=0,                  # 0: 简单手工; 1: 调用 time_features()
                 freq='10min',
                 drop_stop=False,            # 是否剔除停机区间
                 cp_curve_path='CP-CT.dat',  # 功率曲线文件（可选）
                 load_cp_curve=False):
        super().__init__()

        # ---------- 序列长度 ----------
        if size is None:
            self.seq_len  = 6*24*4       # 6 天输入 (6*24h*4=576)
            self.label_len = 24*4        # 1 天 label
            self.pred_len  = 24*4        # 1 天预测
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train':0, 'val':1, 'test':2}[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.turbine_id = turbine_id
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.drop_stop = drop_stop

        # ---------- 读取主数据 ----------
        self.__read_data__()

        # ---------- (可选) 读取功率曲线 ----------
        if load_cp_curve:
            self.cp_curve = self.__read_cp_curve__(cp_curve_path)

    # -------------------------------------------------------------
    def __read_data__(self):
        file = os.path.join(self.root_path, self.data_path)

        # 1) 读 Excel 主表
        df_raw = pd.read_excel(file, sheet_name='DATA_10MIN')

        # 2) 过滤风机
        df_raw = df_raw[df_raw['TURBINE NUMBER'].astype(str).str.strip() == self.turbine_id]

        # 3) 处理时间列（Paris → aware → 可转 UTC）
        paris = tz.gettz('Europe/Paris')
        df_raw['DATE'] = pd.to_datetime(df_raw['DATE ']).dt.tz_localize('UTC')
        # 如果希望统一为 UTC，可加 .dt.tz_convert('UTC')
        df_raw = df_raw.set_index('DATE').sort_index()

        # 4) （可选）剔除停机区间
        if self.drop_stop:
            stop_sheet = pd.read_excel(file, sheet_name='LIST_STOPS')
            stop_sheet = stop_sheet[stop_sheet['TURBINE NUMBER'].astype(str).str.strip() == self.turbine_id]
            # 将开始结束时间转为 tz‑aware
            stop_sheet['START'] = pd.to_datetime(stop_sheet['START OF STOP']).dt.tz_localize(paris)
            stop_sheet['END']   = pd.to_datetime(stop_sheet['END OF STOP']).dt.tz_localize(paris)
            mask = pd.Series(True, index=df_raw.index)
            for _, row in stop_sheet.iterrows():
                mask[row['START']:row['END']] = False
            df_raw = df_raw[mask]

        # 5) 特征选择
        if self.features in ['M', 'MS']:
            # 去掉非数值或无需参与模型的列
            drop_cols = ['ID', 'PROJECT NAME', 'MANUFACTURER',
                         'TURBINE MODEL', 'TURBINE NUMBER']
            cols_data = df_raw.columns.difference(drop_cols)
            df_data = df_raw[cols_data]
        else:            # 'S'
            df_data = df_raw[[self.target]]

        # 6) 划分训练/验证/测试（时间连续）
        n = len(df_data)
        train_end = int(n * 0.7)
        val_end   = int(n * 0.85)
        borders = [(0, train_end),
                   (train_end - self.seq_len, val_end),
                   (val_end - self.seq_len, n)]
        border1, border2 = borders[self.set_type]

        # 7) 标准化
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.iloc[borders[0][0]:borders[0][1]])
            data = self.scaler.transform(df_data)
        else:
            data = df_data.values

        # 8) 时间特征
        if self.timeenc == 0:
            data_stamp = time_features(df_raw.index, freq=self.freq)
        else:
            data_stamp = time_features(df_raw.index, freq=self.freq)  # 也可以换成其它实现
        # 9) 最终切片
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

    # -------------------------------------------------------------
    def __read_cp_curve__(self, cp_curve_path):
        """
        简单示例：读取 CP‑CT.dat，返回 pandas.DataFrame
        文件格式（假设以空格分隔，首行为表头）:
            WS  Power  Cp  Ct
        """
        f = os.path.join(self.root_path, cp_curve_path)
        curve = pd.read_csv(f, sep=r'\s+', comment='#')
        return curve

    # ------------------ Dataset API ------------------------------
    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]
        seq_y      = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (seq_x.astype(np.float32),
                seq_y.astype(np.float32),
                seq_x_mark.astype(np.float32),
                seq_y_mark.astype(np.float32))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



# ----------------- 时间特征 -----------------
def time_features(dates, freq='10min'):
    """
    返回 shape=(len(dates), 6)：
    month / day / weekday / hour / minute(10min 桶) / year_sin_cos(可选)
    """
    df = pd.DataFrame(index=dates)
    df['month']    = df.index.month
    df['day']      = df.index.day
    df['weekday']  = df.index.weekday
    df['hour']     = df.index.hour
    df['minute']   = (df.index.minute // 10)          # 10‑min 桶
    # 可选的周期性特征（示例）:
    # df['year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365.25)
    # df['year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365.25)
    return df.values.astype(np.float32)


class SaintHilaireDataset_1(Dataset):
    """
    长序列预测通用 Dataset
    - seq_len     : 编码器输入长度
    - label_len   : 解码器已知部分长度（如果模型需要，如 Informer）
    - pred_len    : 预测步长
    - features    : 使用哪些输入特征列；'M' = 多变量, 'S' = 单变量
    - target      : 预测目标列名
    """
    def __init__(self,
                 root_path,
                 data_path='SaintHilaire_SCADA.xlsx',
                 flag='train',               # 'train' | 'val' | 'test'
                 size=None,                  # [seq_len, label_len, pred_len]
                 turbine_id='T2',
                 features='S',               # 'S' | 'M' | 'MS'
                 target='ACTIVE POWER [Watt]',
                 scale=True,
                 timeenc=0,                  # 0: 简单手工; 1: 调用 time_features()
                 freq='10min',
                 drop_stop=False,            # 是否剔除停机区间
                 cp_curve_path='CP-CT.dat',  # 功率曲线文件（可选）
                 load_cp_curve=False):
        
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        self.features_mode, self.target_col = features, target

        cols_keep = [
            "DATE",                     # 时间索引
            "WIND_SPEED",
            "WIND_DIRECTION",
            "EXTERNAL_TEMPERATURE",
            "LOW_SPEED_SHAFT",
            "HIGH_SPEED_SHAFT",
            "PITCH_ANGLE",
            "GENERATE_TORQUE",
            "TARGETED_TORQUE",
            "ACTIVE_POWER",             # 预测目标
        ]

        input_cols = [
            "WIND_SPEED",
            "WIND_DIRECTION",
            "EXTERNAL_TEMPERATURE",
            "LOW_SPEED_SHAFT",
            "HIGH_SPEED_SHAFT",
            "PITCH_ANGLE",
            "GENERATE_TORQUE",
            "TARGETED_TORQUE",
        ]

        # ---------- 1. 读 & 清洗 ----------
        df = (
            pd.read_csv(csv_path, encoding="utf-8-sig", usecols=cols_keep)   # ← 只留需要的列
            .rename(columns=str.strip)
            .drop_duplicates()
        )


        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").set_index("DATE")

        # 把所有字符串列先去空格再转换为 category → one-hot（如果需要）
        cat_cols = df.select_dtypes("object").columns
        if len(cat_cols):
            df[cat_cols] = df[cat_cols].apply(lambda s: s.str.strip())
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # ---------- 2. 归一化（只对数值列；不包括时间索引） ----------
        numeric_cols = input_cols + ["ACTIVE_POWER"]
        if scaler is None:                # 只在第一次初始化 train 时建立 scaler
            scaler = StandardScaler()
            scaler.fit(df.iloc[: int(len(df) * train_split)][numeric_cols])
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 保存 scaler 供外部复用
        self.scaler = scaler

        # ---------- 3. 划分数据段 ----------
        n = len(df)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))

        if flag == "train":
            self.data = df.iloc[: train_end]
            self.start = 0
        elif flag == "val":
            self.data = df.iloc[train_end:val_end]
            self.start = train_end
        else:  # "test"
            self.data = df.iloc[val_end:]
            self.start = val_end

        if features == "S":               # 单变量
            self.cols_x = [target]
        else:                             # 多变量
            self.cols_x = list(df.columns)

        # ---------- 4. 转成 numpy ----------
        self.data_x = self.data[self.cols_x].values.astype(np.float32)
        self.data_y = self.data[target].values.astype(np.float32).reshape(-1, 1)

    # ---- 核心：滑动窗口 ----
    def __len__(self):
        # 每一次取 seq_len + pred_len 的窗口；label_len 用于 decoder 已知部分
        return len(self.data_x) - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, idx):
        seq_start = idx
        seq_end = seq_start + self.seq_len
        label_end = seq_end + self.pred_len

        # encoder 输入
        seq_x = self.data_x[seq_start:seq_end]

        # decoder 输入（已知 label_len + pred_len，但模型常用前 label_len 作为已知）
        seq_y = self.data_y[seq_end - self.label_len : label_end]

        # 真实待预测 y （只取最后 pred_len）
        target_y = self.data_y[seq_end : label_end]

        return (
            torch.from_numpy(seq_x),        # [seq_len, dim]
            torch.from_numpy(seq_y),        # [label_len + pred_len, 1]
            torch.from_numpy(target_y),     # [pred_len, 1]
        )




import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

# def time_features_stamp(dates: pd.Series):
#     """
#     从 pandas 时间序列中提取时间特征：月、日、星期、小时
#     返回 numpy array，shape=(len(dates), 4)
#     """
#     df_stamp = pd.DataFrame({
#         'date': dates
#     })
#     df_stamp['month'] = df_stamp['date'].dt.month
#     df_stamp['day'] = df_stamp['date'].dt.day
#     df_stamp['weekday'] = df_stamp['date'].dt.weekday
#     df_stamp['hour'] = df_stamp['date'].dt.hour
#     return df_stamp[['month','day','weekday','hour']].values
def time_features_stamp(dates: pd.Series):
    df_stamp = pd.DataFrame({'date': dates})
    # 编码小时
    hour = df_stamp['date'].dt.hour
    df_stamp['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_stamp['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    # 编码月份
    month = df_stamp['date'].dt.month
    df_stamp['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df_stamp['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    return df_stamp[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']].values

class SaintHilaireDataset(Dataset):
    """
    只用 8 个数值特征 + 时间特征 → 预测 ACTIVE_POWER
    预处理流程：
      1. 整个文件读入 → **删除含 NaN 行**
      2. 在训练段拟合 StandardScaler
      3. transform 整个数据
      4. 再切出 train/val/test
    返回 (seq_x, seq_y, seq_x_mark, seq_y_mark)
    """

    input_cols: List[str] = [
        "WIND_SPEED",         # 风速
        "WIND_DIRECTION",   # 风向
        "EXTERNAL_TEMPERATURE", # 外部温度
        "LOW_SPEED_SHAFT",    # 低速轴
        "HIGH_SPEED_SHAFT",   # 高速轴
        "PITCH_ANGLE",      # 桨距角
        "GENERATE_TORQUE",    # 发电扭矩
        "TARGETED_TORQUE",    # 目标扭矩
    ]

    # 预测目标列
    target_col: str = "ACTIVE_POWER"

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: list,
        flag: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.1,
        scale: bool = True,
    ):
        super().__init__()
        assert flag in ["train", "val", "test"], "flag 必须是 'train' | 'val' | 'test'"
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        # 窗口长度配置
        self.seq_len, self.label_len, self.pred_len = size
        self.scale = scale
        self.scaler = StandardScaler()

        # 读取与预处理
        self._read_data_(root_path, data_path, train_split, val_split)

    def _read_data_(self, root_path: str, data_path: str, train_split: float, val_split: float):
        """读取 csv 并完成缺失值剔除、时间排序、缩放与切片"""
        full_path = os.path.join(root_path, data_path)
        keep_cols = ["DATE", *self.input_cols, self.target_col]

        # 1️⃣ 读取并初步处理 ---------------------------------------------------
        df_raw = (
            pd.read_csv(full_path, usecols=keep_cols, encoding="utf-8-sig")
              .rename(columns=str.strip)         # 去除列名空格
              .drop_duplicates()                 # 去重
        )

        # 转换日期列
        df_raw["DATE"] = pd.to_datetime(df_raw["DATE"])
        df_raw = df_raw.sort_values("DATE").reset_index(drop=True)

        # **删除包含 NaN 的行**  —— 解决下游 NaN 问题
        df_raw = df_raw.dropna().reset_index(drop=True)

        # 2️⃣ 计算 train/val/test 边界 ----------------------------------------
        n = len(df_raw)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        borders1 = [0, train_end, val_end]
        borders2 = [train_end, val_end, n]
        b1, b2 = borders1[self.set_type], borders2[self.set_type]

        # 3️⃣ 标准化（可选） ---------------------------------------------------
        num_df = df_raw[self.input_cols + [self.target_col]]

        if self.scale:
            # 仅用训练段统计 μ、σ，防止信息泄露
            train_slice = num_df.iloc[borders1[0]:borders2[0]]
            self.scaler.fit(train_slice.values)
            data_all = self.scaler.transform(num_df.values)
        else:
            data_all = num_df.values

        # 4️⃣ 切片当前 flag 段 -------------------------------------------------
        slice_data = data_all[b1:b2]
        self.data_x = slice_data[:, : len(self.input_cols)].astype(np.float32)
        self.data_y = slice_data[:, -1:].astype(np.float32)  # 形状 (N,1)

        # 5️⃣ 构造时间特征 -----------------------------------------------------
        dates = df_raw["DATE"].iloc[b1:b2]
        self.data_stamp = time_features_stamp(dates)  # 用户提供该函数

    # ----------------------------------------------------------------------
    # Dataset 接口实现
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.data_x) - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, idx: int):
        s = idx
        e = s + self.seq_len
        r_start = e - self.label_len
        r_end = r_start + self.label_len + self.pred_len

        seq_x = self.data_x[s:e]                    # 形状 [seq_len, 8]
        seq_y = self.data_y[r_start:r_end]          # 形状 [label_len+pred_len, 1]
        seq_x_mark = self.data_stamp[s:e]           # 时间特征 [seq_len, ?]
        seq_y_mark = self.data_stamp[r_start:r_end] # 时间特征 [label_len+pred_len, ?]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(seq_x_mark),
            torch.from_numpy(seq_y_mark),
        )

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def inverse_transform(self, data: np.ndarray):
        """将预测输出从标准化域映射回原始单位（只针对 target）"""
        dummy = np.zeros((*data.shape[:-1], len(self.input_cols) + 1))
        dummy[..., -1:] = data
        inv = self.scaler.inverse_transform(dummy.reshape(-1, dummy.shape[-1]))[:, -1]
        return inv.reshape(data.shape)





class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
