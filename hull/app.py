
import os
import sys
# --- 环境硬化：在导入任何 heavy numeric 库之前设置线程限制并使用 spawn 启动方法
# 这能显著降低 OpenMP/BLAS 与 Qt 主线程交互时出现的 native crash 风险
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
try:
    import multiprocessing as _mp
    _mp.set_start_method('spawn', force=True)
except Exception:
    # 如果已经在某处设置或不支持，则忽略
    pass
import time
import tempfile
import traceback
import json
import pickle

from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shutil
import importlib.util
import threading

from PySide6 import QtWidgets, QtGui, QtCore
import faulthandler
faulthandler.enable()

import pyqtgraph as pg

import torch
# Ensure parent directory (project root) is on sys.path so imports like
# `train_adjusted_sharpe` work when running this file from the `hull/` folder.
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import train_adjusted_sharpe as tas
from simulator import SimulationWorker, QuickCheckWorker, LGBRunWorker
import job_queue


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Hull Tactical — 训练器 GUI')
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # job poll offsets: track how many bytes of progress.jsonl we've read per job
        self._job_poll_offsets = {}

        # 创建主水平分割：左（参数） | 中（选项卡） | 右（运行历史）
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)
        # 使用垂直布局把 splitter 放上面，bottom_bar 放下面（之前使用水平布局导致 bottom_bar 被放到右侧）
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(self.main_splitter)

        # --- 左侧：参数面板（堆叠分组），放入可滚动容器以适应小窗口 ---
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        # 左对齐整个左侧面板内容（顶部靠左）
        try:
            left_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        except Exception:
            pass
        # 统一左侧输入控件的最小宽度，使其与“运行控制”区域看起来一致
        try:
            left_container.setStyleSheet('QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { min-width: 240px; }')
        except Exception:
            pass

        # 全局 / 数据 分组（在 QToolBox 中使用面板标题，因此内部不显示额外标题）
        global_grp = QtWidgets.QWidget()
        global_layout = QtWidgets.QFormLayout(global_grp)
        # 表单内标签与表单项左对齐
        try:
            global_layout.setFormAlignment(QtCore.Qt.AlignLeft)
            global_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        except Exception:
            pass
        
        self.train_csv_lbl = QtWidgets.QLabel()
        global_layout.addWidget(self.train_csv_lbl)

        # Global options: missing strategy, add is_missing    
        self.missing_strategy_combo = QtWidgets.QComboBox()
        # display Chinese labels but keep english keys as userData when needed
        self.missing_strategy_combo.addItem('无', 'none')
        self.missing_strategy_combo.addItem('向前填充', 'ffill')
        self.missing_strategy_combo.addItem('中位数填充', 'median')
        self.missing_strategy_combo.addItem('置零', 'zero')
        self.missing_strategy_combo.addItem('插值', 'interpolate')
        #self.add_is_missing_cb = QtWidgets.QCheckBox('添加缺失值指示器')
        global_layout.addWidget(QtWidgets.QLabel('缺失填充：') )
        global_layout.addWidget(self.missing_strategy_combo)

        self.model_combo = QtWidgets.QComboBox()
        # 显示中文标签，但保存内部值为实际模型标识（userData）
        self.model_combo.addItem('LSTM（长短期记忆）', 'lstm')
        self.model_combo.addItem('MLP（多层感知器）', 'mlp')
        self.model_combo.addItem('LightGBM（梯度提升）', 'lightgbm')
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        global_layout.addWidget(QtWidgets.QLabel('选择模型：') )
        global_layout.addWidget(self.model_combo)

        self.seq_spin = QtWidgets.QSpinBox()
        self.seq_spin.setRange(2, 1000)
        self.seq_spin.setValue(252)
        global_layout.addWidget(QtWidgets.QLabel('序列长度：'))
        global_layout.addWidget(self.seq_spin)
        
        
        self.lag_checkboxes = []
        lag_widget = QtWidgets.QWidget()
        # 横向分布复选框（水平布局），便于在一行内显示多个选项
        lag_layout = QtWidgets.QHBoxLayout(lag_widget)
        lag_layout.setContentsMargins(4,4,4,4)
        lag_layout.setSpacing(8)
        for v in [1,2,3,5,10,21,63]:
            cb = QtWidgets.QCheckBox(str(v))
            cb.setChecked(False)
            lag_layout.addWidget(cb)
            self.lag_checkboxes.append(cb)
        lag_layout.addStretch(1)
        lag_scroll = QtWidgets.QScrollArea()
        lag_scroll.setWidgetResizable(True)
        lag_scroll.setWidget(lag_widget)
        # 提高高度以容纳水平复选框
        lag_scroll.setMaximumHeight(30)
        lag_scroll.setMinimumWidth(180)
  
        global_layout.addWidget(QtWidgets.QLabel('滞后天数：'))
        global_layout.addWidget(lag_scroll)

        self.rolling_checkboxes = []
        roll_widget = QtWidgets.QWidget()
        # 横向分布滚动窗口复选框
        roll_layout = QtWidgets.QHBoxLayout(roll_widget)
        roll_layout.setContentsMargins(4,4,4,4)
        roll_layout.setSpacing(8)
        for v in [5,10,21,63]:
            cb = QtWidgets.QCheckBox(str(v))
            cb.setChecked(False)
            roll_layout.addWidget(cb)
            self.rolling_checkboxes.append(cb)
        roll_layout.addStretch(1)
        roll_scroll = QtWidgets.QScrollArea()
        roll_scroll.setWidgetResizable(True)
        roll_scroll.setWidget(roll_widget)
        roll_scroll.setMaximumHeight(30)
        roll_scroll.setMinimumWidth(180)
        global_layout.addWidget(QtWidgets.QLabel('滚动窗口：'))
        global_layout.addWidget(roll_scroll)
        # EWMA span
        self.ewma_span = QtWidgets.QSpinBox()
        self.ewma_span.setRange(1, 252)
        self.ewma_span.setValue(21)
        global_layout.addWidget(QtWidgets.QLabel('EWMA数值：'))        
        global_layout.addWidget(self.ewma_span)


        # SHAP
        self.shap_cb = QtWidgets.QCheckBox('SHAP特征：')
        self.shap_sample = QtWidgets.QSpinBox()
        self.shap_sample.setRange(10, 10000)
        self.shap_sample.setValue(200)
        # place SHAP checkbox and sample side-by-side
        shp_row = QtWidgets.QWidget()
        shp_layout = QtWidgets.QHBoxLayout(shp_row)
        shp_layout.setContentsMargins(0,0,0,0)
        shp_layout.addWidget(self.shap_cb)
        shp_layout.addStretch(1)
        shp_layout.addWidget(self.shap_sample)


        # We'll use a QToolBox as an accordion to host the groups
        left_toolbox = QtWidgets.QToolBox()
        left_toolbox.addItem(global_grp, '数据处理')

        # 验证 分组（占位，不显示重复标题）
        val_grp = QtWidgets.QWidget()
        val_layout = QtWidgets.QFormLayout(val_grp)
        try:
            val_layout.setFormAlignment(QtCore.Qt.AlignLeft)
            val_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        except Exception:
            pass
        self.cv_method_combo = QtWidgets.QComboBox()
        # 本地化 CV 方法显示，内部使用标识作为 userData
        self.cv_method_combo.addItem('剔除泄露的K折', 'purged_kfold')
        self.cv_method_combo.addItem('滚动前向', 'walk_forward')
        # 当选择 Walk-Forward 时显示额外参数输入
        self.cv_method_combo.currentIndexChanged.connect(self.on_cv_method_changed)
        val_layout.addWidget(QtWidgets.QLabel('CV 方法：'))
        val_layout.addWidget(self.cv_method_combo)
        # Purged K-Fold 参数
        self.n_splits_spin = QtWidgets.QSpinBox()
        self.n_splits_spin.setRange(2, 50)
        self.n_splits_spin.setValue(5)
        val_layout.addWidget(QtWidgets.QLabel('折数：'))
        val_layout.addWidget(self.n_splits_spin)
        self.embargo_spin = QtWidgets.QSpinBox()
        self.embargo_spin.setRange(0, 365)
        self.embargo_spin.setValue(0)
        val_layout.addWidget(QtWidgets.QLabel('禁封天数：'))
        val_layout.addWidget(self.embargo_spin)
        self.label_horizon_spin = QtWidgets.QSpinBox()
        self.label_horizon_spin.setRange(1, 365)
        self.label_horizon_spin.setValue(1)
        val_layout.addWidget(QtWidgets.QLabel('标签时长：'))
        val_layout.addWidget(self.label_horizon_spin)
        # Walk-Forward 专用参数（默认隐藏，切换到 Walk-Forward 时显示）
        self.wf_train_label = QtWidgets.QLabel('训练窗口：')
        self.wf_train_label.setVisible(False)
        self.wf_train_spin = QtWidgets.QSpinBox()
        self.wf_train_spin.setRange(1, 10000)
        self.wf_train_spin.setValue(252)
        self.wf_train_spin.setVisible(False)
        val_layout.addWidget(self.wf_train_label)
        val_layout.addWidget(self.wf_train_spin)

        self.wf_val_label = QtWidgets.QLabel('验证窗口：')
        self.wf_val_label.setVisible(False)
        self.wf_val_spin = QtWidgets.QSpinBox()
        self.wf_val_spin.setRange(1, 10000)
        self.wf_val_spin.setValue(63)
        self.wf_val_spin.setVisible(False)
        val_layout.addWidget(self.wf_val_label)
        val_layout.addWidget(self.wf_val_spin)

        self.wf_step_label = QtWidgets.QLabel('步长：')
        self.wf_step_label.setVisible(False)
        self.wf_step_spin = QtWidgets.QSpinBox()
        self.wf_step_spin.setRange(1, 10000)
        self.wf_step_spin.setValue(63)
        self.wf_step_spin.setVisible(False)
        val_layout.addWidget(self.wf_step_label)
        val_layout.addWidget(self.wf_step_spin)

        # Embargo / lag inline 警告（红色），供即时显示
        self.embargo_warn_label = QtWidgets.QLabel('')
        self.embargo_warn_label.setStyleSheet('color: #b00020;')
        self.embargo_warn_label.setWordWrap(True)
        self.embargo_warn_label.setVisible(False)
        val_layout.addWidget(self.embargo_warn_label)
        self.verify_cv_btn = QtWidgets.QPushButton('验证CV/分数一致性')
        self.verify_cv_btn.clicked.connect(self.on_verify_cv)
        val_layout.addWidget(self.verify_cv_btn)
        # 隐藏的不定进度条，用于快速检查（快速检查期间显示）
        self.verify_progress = QtWidgets.QProgressBar()
        self.verify_progress.setRange(0, 0)
        self.verify_progress.setVisible(False)
        val_layout.addWidget(self.verify_progress)
        left_toolbox.addItem(val_grp, '序列验证')

        # 模型相关设置 区域
        model_stack_grp = QtWidgets.QWidget()
        model_stack_layout = QtWidgets.QVBoxLayout(model_stack_grp)
        try:
            model_stack_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        except Exception:
            pass
        self.model_stack = QtWidgets.QStackedWidget()

        # NN 面板（复用已有控件）
        nn_panel = QtWidgets.QWidget()
        nn_form = QtWidgets.QFormLayout(nn_panel)
        try:
            nn_form.setFormAlignment(QtCore.Qt.AlignLeft)
            nn_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        except Exception:
            pass
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(6)
        nn_form.addWidget(QtWidgets.QLabel('训练轮数：'))
        nn_form.addWidget(self.epochs_spin)
        self.lr_edit = QtWidgets.QLineEdit('3e-4')
        nn_form.addWidget(QtWidgets.QLabel('学习率：'))
        nn_form.addWidget(self.lr_edit)
        self.lambda_edit = QtWidgets.QLineEdit('1.0')
        nn_form.addWidget(QtWidgets.QLabel('换手正则：'))
        nn_form.addWidget(self.lambda_edit)
        self.l2_edit = QtWidgets.QLineEdit('0.0')
        nn_form.addWidget(QtWidgets.QLabel('L2 正则：'))
        nn_form.addWidget(self.l2_edit)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(32)
        nn_form.addWidget(QtWidgets.QLabel('批大小：'))
        nn_form.addWidget(self.batch_spin)
        # NN 架构
        self.nn_type_combo = QtWidgets.QComboBox()
        # 显示中文标签，内部使用模型标识作为 userData
        self.nn_type_combo.addItem('LSTM（长短期记忆）', 'lstm')
        self.nn_type_combo.addItem('MLP（多层感知器）', 'mlp')
        nn_form.addWidget(QtWidgets.QLabel('NN 类型：'))
        nn_form.addWidget(self.nn_type_combo)
        # MLP 隐藏层 字符串
        self.mlp_hidden_edit = QtWidgets.QLineEdit('128,64')
        nn_form.addWidget(QtWidgets.QLabel('MLP 隐藏层（逗号分隔）：'))
        nn_form.addWidget(self.mlp_hidden_edit)
        self.hidden_dim_spin = QtWidgets.QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(128)
        nn_form.addWidget(QtWidgets.QLabel('hidden_dim：'))
        nn_form.addWidget(self.hidden_dim_spin)
        self.n_layers_spin = QtWidgets.QSpinBox()
        self.n_layers_spin.setRange(1, 10)
        self.n_layers_spin.setValue(2)
        nn_form.addWidget(QtWidgets.QLabel('n_layers：'))
        nn_form.addWidget(self.n_layers_spin)
        self.dropout_spin = QtWidgets.QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.1)
        nn_form.addWidget(QtWidgets.QLabel('dropout：'))
        nn_form.addWidget(self.dropout_spin)
        # 微调 / 高级设置
        self.pretrain_epochs = QtWidgets.QSpinBox()
        self.pretrain_epochs.setRange(0, 1000)
        self.pretrain_epochs.setValue(0)
        nn_form.addWidget(QtWidgets.QLabel('预训练轮数：'))
        nn_form.addWidget(self.pretrain_epochs)
        self.use_double_precision_cb = QtWidgets.QCheckBox('使用双精度')
        nn_form.addWidget(self.use_double_precision_cb)
        self.model_stack.addWidget(nn_panel)

        # LightGBM 面板（占位）
        lgb_panel = QtWidgets.QWidget()
        lgb_form = QtWidgets.QFormLayout(lgb_panel)
        try:
            lgb_form.setFormAlignment(QtCore.Qt.AlignLeft)
            lgb_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        except Exception:
            pass
        self.lgb_learning_rate = QtWidgets.QDoubleSpinBox()
        self.lgb_learning_rate.setRange(1e-6, 1.0)
        self.lgb_learning_rate.setDecimals(6)
        self.lgb_learning_rate.setSingleStep(1e-3)
        self.lgb_learning_rate.setValue(0.01)
        lgb_form.addWidget(QtWidgets.QLabel('学习率：'))
        lgb_form.addWidget(self.lgb_learning_rate)
        self.lgb_num_leaves = QtWidgets.QSpinBox()
        self.lgb_num_leaves.setRange(2, 65536)
        self.lgb_num_leaves.setValue(31)
        lgb_form.addWidget(QtWidgets.QLabel('叶子数：'))
        lgb_form.addWidget(self.lgb_num_leaves)
        self.lgb_min_child_samples = QtWidgets.QSpinBox()
        self.lgb_min_child_samples.setRange(1, 10000)
        self.lgb_min_child_samples.setValue(20)
        lgb_form.addWidget(QtWidgets.QLabel('最小子样本数：'))
        lgb_form.addWidget(self.lgb_min_child_samples)
        self.lgb_feature_fraction = QtWidgets.QDoubleSpinBox()
        self.lgb_feature_fraction.setRange(0.01, 1.0)
        self.lgb_feature_fraction.setSingleStep(0.01)
        self.lgb_feature_fraction.setValue(0.8)
        lgb_form.addWidget(QtWidgets.QLabel('特征子采样比例：'))
        lgb_form.addWidget(self.lgb_feature_fraction)
        self.lgb_bagging_fraction = QtWidgets.QDoubleSpinBox()
        self.lgb_bagging_fraction.setRange(0.01, 1.0)
        self.lgb_bagging_fraction.setSingleStep(0.01)
        self.lgb_bagging_fraction.setValue(0.8)
        lgb_form.addWidget(QtWidgets.QLabel('样本子采样比例：'))
        lgb_form.addWidget(self.lgb_bagging_fraction)
        self.lgb_bagging_freq = QtWidgets.QSpinBox()
        self.lgb_bagging_freq.setRange(0, 1000)
        self.lgb_bagging_freq.setValue(0)
        lgb_form.addWidget(QtWidgets.QLabel('子采样频率：'))
        lgb_form.addWidget(self.lgb_bagging_freq)
        self.lgb_reg_alpha = QtWidgets.QDoubleSpinBox()
        self.lgb_reg_alpha.setRange(0.0, 100.0)
        self.lgb_reg_alpha.setSingleStep(0.1)
        self.lgb_reg_alpha.setValue(0.0)
        lgb_form.addWidget(QtWidgets.QLabel('L1 正则：'))
        lgb_form.addWidget(self.lgb_reg_alpha)
        self.lgb_reg_lambda = QtWidgets.QDoubleSpinBox()
        self.lgb_reg_lambda.setRange(0.0, 100.0)
        self.lgb_reg_lambda.setSingleStep(0.1)
        self.lgb_reg_lambda.setValue(0.0)
        lgb_form.addWidget(QtWidgets.QLabel('L2 正则：'))
        lgb_form.addWidget(self.lgb_reg_lambda)
        self.lgb_num_boost_round = QtWidgets.QSpinBox()
        self.lgb_num_boost_round.setRange(1, 100000)
        self.lgb_num_boost_round.setValue(100)
        lgb_form.addWidget(QtWidgets.QLabel('Boost 轮数：'))
        lgb_form.addWidget(self.lgb_num_boost_round)
        self.lgb_early_stopping = QtWidgets.QSpinBox()
        self.lgb_early_stopping.setRange(0, 10000)
        self.lgb_early_stopping.setValue(20)
        lgb_form.addWidget(QtWidgets.QLabel('早停轮数：'))
        lgb_form.addWidget(self.lgb_early_stopping)
        # Optuna 开关
        self.lgb_use_optuna = QtWidgets.QCheckBox('使用 Optuna')
        lgb_form.addWidget(self.lgb_use_optuna)
        self.lgb_optuna_trials = QtWidgets.QSpinBox()
        self.lgb_optuna_trials.setRange(1, 10000)
        self.lgb_optuna_trials.setValue(50)
        lgb_form.addWidget(QtWidgets.QLabel('Optuna 试验次数：'))
        lgb_form.addWidget(self.lgb_optuna_trials)
        self.model_stack.addWidget(lgb_panel)

        # PPO 面板（占位）
        ppo_panel = QtWidgets.QWidget()
        ppo_form = QtWidgets.QFormLayout(ppo_panel)
        try:
            ppo_form.setFormAlignment(QtCore.Qt.AlignLeft)
            ppo_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        except Exception:
            pass
        self.ppo_window_len = QtWidgets.QSpinBox()
        self.ppo_window_len.setRange(1, 10000)
        self.ppo_window_len.setValue(63)
        ppo_form.addWidget(QtWidgets.QLabel('窗口长度：'))
        ppo_form.addWidget(self.ppo_window_len)
        self.ppo_episode_len = QtWidgets.QSpinBox()
        self.ppo_episode_len.setRange(1, 10000)
        self.ppo_episode_len.setValue(252)
        ppo_form.addWidget(QtWidgets.QLabel('回合长度：'))
        ppo_form.addWidget(self.ppo_episode_len)
        self.ppo_turnover_cost = QtWidgets.QDoubleSpinBox()
        self.ppo_turnover_cost.setRange(0.0, 1.0)
        self.ppo_turnover_cost.setSingleStep(0.0001)
        self.ppo_turnover_cost.setValue(0.0)
        ppo_form.addWidget(QtWidgets.QLabel('换手成本：'))
        ppo_form.addWidget(self.ppo_turnover_cost)
        # PPO 超参数
        self.ppo_lr = QtWidgets.QDoubleSpinBox()
        self.ppo_lr.setRange(1e-8, 1.0)
        self.ppo_lr.setDecimals(8)
        self.ppo_lr.setValue(3e-4)
        ppo_form.addWidget(QtWidgets.QLabel('PPO 学习率：'))
        ppo_form.addWidget(self.ppo_lr)
        self.ppo_n_updates = QtWidgets.QSpinBox()
        self.ppo_n_updates.setRange(1, 100000)
        self.ppo_n_updates.setValue(10)
        ppo_form.addWidget(QtWidgets.QLabel('PPO 更新次数：'))
        ppo_form.addWidget(self.ppo_n_updates)
        self.ppo_steps_per_update = QtWidgets.QSpinBox()
        self.ppo_steps_per_update.setRange(1, 100000)
        self.ppo_steps_per_update.setValue(2048)
        ppo_form.addWidget(QtWidgets.QLabel('每次更新步数：'))
        ppo_form.addWidget(self.ppo_steps_per_update)
        self.ppo_epochs = QtWidgets.QSpinBox()
        self.ppo_epochs.setRange(1, 100)
        self.ppo_epochs.setValue(10)
        ppo_form.addWidget(QtWidgets.QLabel('PPO 轮数：'))
        ppo_form.addWidget(self.ppo_epochs)
        self.ppo_minibatch = QtWidgets.QSpinBox()
        self.ppo_minibatch.setRange(1, 1024)
        self.ppo_minibatch.setValue(64)
        ppo_form.addWidget(QtWidgets.QLabel('PPO 小批量：'))
        ppo_form.addWidget(self.ppo_minibatch)
        self.ppo_clip = QtWidgets.QDoubleSpinBox()
        self.ppo_clip.setRange(0.0, 1.0)
        self.ppo_clip.setSingleStep(0.01)
        self.ppo_clip.setValue(0.2)
        ppo_form.addWidget(QtWidgets.QLabel('PPO 截断值：'))
        ppo_form.addWidget(self.ppo_clip)
        self.ppo_entropy_coef = QtWidgets.QDoubleSpinBox()
        self.ppo_entropy_coef.setRange(0.0, 1.0)
        self.ppo_entropy_coef.setSingleStep(0.001)
        self.ppo_entropy_coef.setValue(0.0)
        ppo_form.addWidget(QtWidgets.QLabel('熵系数：'))
        ppo_form.addWidget(self.ppo_entropy_coef)

        self.model_stack.addWidget(ppo_panel)

        model_stack_layout.addWidget(self.model_stack)
        left_toolbox.addItem(model_stack_grp, '模型设置')

        # 运行按钮与小选项
        run_grp = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_grp)
        try:
            run_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        except Exception:
            pass
        self.sim_btn = QtWidgets.QPushButton('评估运行')
        self.sim_btn.clicked.connect(self.on_run_simulation)
        run_layout.addWidget(self.sim_btn)
        self.save_dir_edit = QtWidgets.QLineEdit('./gui_models')
        run_layout.addWidget(QtWidgets.QLabel('保存目录：'))
        run_layout.addWidget(self.save_dir_edit)
        # last_k（模拟最后 K 行）
        # sim_epochs：每次模拟重训练内部运行的训练轮数
        self.sim_epochs_spin = QtWidgets.QSpinBox()
        self.sim_epochs_spin.setRange(0, 10)
        self.sim_epochs_spin.setValue(2)
        run_layout.addWidget(QtWidgets.QLabel('每日训练轮数：'))
        run_layout.addWidget(self.sim_epochs_spin)

        # 仅干运行复选框（打印命令，不执行）
        self.dry_run_cb = QtWidgets.QCheckBox('仅干运行')
        run_layout.addWidget(self.dry_run_cb)

        self.lastk_spin = QtWidgets.QSpinBox()
        self.lastk_spin.setRange(1, 10000)
        self.lastk_spin.setValue(180)
        run_layout.addWidget(QtWidgets.QLabel('模拟天数：'))
        run_layout.addWidget(self.lastk_spin)
        left_toolbox.addItem(run_grp, '运行控制')

        left_layout.addWidget(left_toolbox)

        left_layout.addStretch(1)
        # 使用 QScrollArea 包装左侧参数面板，启用可缩放内容
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        # 设置合适的最小宽度以避免过窄
        left_scroll.setMinimumWidth(200)
        main_splitter = self.main_splitter
        main_splitter.addWidget(left_scroll)

        # --- Top toolbar (upload, save/load config, presets, help, verify) ---
        toolbar = QtWidgets.QToolBar('主工具栏')
        toolbar.setIconSize(QtCore.QSize(16,16))
        self.addToolBar(toolbar)

        upload_act = QtGui.QAction('上传 CSV', self)
        upload_act.triggered.connect(self.on_select_csv)
        toolbar.addAction(upload_act)

        save_cfg_act = QtGui.QAction('保存配置', self)
        save_cfg_act.triggered.connect(self.on_save_config)
        toolbar.addAction(save_cfg_act)

        load_cfg_act = QtGui.QAction('加载配置', self)
        load_cfg_act.triggered.connect(self.on_load_config)
        toolbar.addAction(load_cfg_act)

        toolbar.addSeparator()
        help_act = QtGui.QAction('帮助', self)
        help_act.triggered.connect(lambda: self.append_log('帮助：打开帮助文档'))
        toolbar.addAction(help_act)

        verify_act = QtGui.QAction('验证', self)
        verify_act.triggered.connect(self.on_verify_cv)
        toolbar.addAction(verify_act)

        # 快速显示诊断图按钮（KPI / 累计收益 / 缺失率）
        show_diag_act = QtGui.QAction('显示诊断图', self)
        show_diag_act.triggered.connect(self.on_show_diagnostics)
        toolbar.addAction(show_diag_act)

        # --- 中间：选项卡（概览 / 训练 / 诊断 / 比较） ---
        center_widget = QtWidgets.QTabWidget()

        # 概览选项卡：快速绘图与 KPI
        overview_tab = QtWidgets.QWidget()
        ov_layout = QtWidgets.QVBoxLayout(overview_tab)
        self.pg_plot = pg.PlotWidget(title='权益 / 仓位')
        self.pg_plot.addLegend()
        self.pg_plot.showGrid(x=True, y=True)
        ov_layout.addWidget(self.pg_plot, 2)
        # 使用 QTextBrowser 作为 WebEngine 的轻量回退，避免 macOS 上的 QtWebEngine 段错误
        self.web = QtWidgets.QTextBrowser()
        ov_layout.addWidget(self.web, 1)
        center_widget.addTab(overview_tab, '概览')

        # 训练选项卡：日志与运行摘要
        training_tab = QtWidgets.QWidget()
        tr_layout = QtWidgets.QVBoxLayout(training_tab)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        tr_layout.addWidget(self.log)
        center_widget.addTab(training_tab, '训练')

        # 诊断选项卡（占位）
        diag_tab = QtWidgets.QWidget()
        diag_layout = QtWidgets.QVBoxLayout(diag_tab)
        diag_layout.addWidget(QtWidgets.QLabel('诊断图表将在此处显示'))
        center_widget.addTab(diag_tab, '诊断')

        # 比较选项卡（占位）
        cmp_tab = QtWidgets.QWidget()
        cmp_layout = QtWidgets.QVBoxLayout(cmp_tab)
        cmp_layout.addWidget(QtWidgets.QLabel('并排比较运行结果'))
        center_widget.addTab(cmp_tab, '比较')

        main_splitter.addWidget(center_widget)

        # --- 右侧：运行历史 / 产物 ---
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.addWidget(QtWidgets.QLabel('运行历史'))
        self.run_table = QtWidgets.QTableWidget(0,5)
        self.run_table.setHorizontalHeaderLabels(['时间戳','模型','调整后夏普','RMSE','路径'])
        self.run_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.run_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.run_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.run_table)
        # run history action buttons
        btn_row = QtWidgets.QWidget()
        br_layout = QtWidgets.QHBoxLayout(btn_row)
        load_btn = QtWidgets.QPushButton('加载所选')
        load_btn.clicked.connect(self.on_load_selected_run)
        br_layout.addWidget(load_btn)
        rerun_btn = QtWidgets.QPushButton('重跑所选')
        rerun_btn.clicked.connect(self.on_rerun_selected)
        br_layout.addWidget(rerun_btn)
        download_btn = QtWidgets.QPushButton('下载选中 run_info')
        download_btn.clicked.connect(self.on_download_selected_run)
        br_layout.addWidget(download_btn)
        compare_btn = QtWidgets.QPushButton('比较所选')
        compare_btn.clicked.connect(self.on_compare_selected)
        br_layout.addWidget(compare_btn)
        right_layout.addWidget(btn_row)
        right_layout.addStretch(1)
        main_splitter.addWidget(right_widget)

        # 设置初始分割尺寸（约 20/55/25）
        # 使用更精确的比率以匹配 20% / 55% / 25%
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 11)
        main_splitter.setStretchFactor(2, 5)

        # 最小宽度保护，避免在窄屏时表单变形
        try:
            left_scroll.setMinimumWidth(280)
        except Exception:
            pass
        try:
            center_widget.setMinimumWidth(640)
        except Exception:
            pass
        try:
            right_widget.setMinimumWidth(320)
        except Exception:
            pass

        # 线程池
        self.pool = QtCore.QThreadPool.globalInstance()

        # Restore window geometry and splitter state if saved
        try:
            settings = QtCore.QSettings('hull', 'trainer_gui')
            geo = settings.value('geometry')
            if geo is not None:
                try:
                    self.restoreGeometry(geo)
                except Exception:
                    pass
            split_state = settings.value('splitter')
            if split_state is not None:
                try:
                    self.main_splitter.restoreState(split_state)
                except Exception:
                    pass
        except Exception:
            pass

        # --- 底部运行控制栏（复制 Run/Stop、autosave、日志操作） ---
        bottom_bar = QtWidgets.QWidget()
        bb_layout = QtWidgets.QHBoxLayout(bottom_bar)
        # 底部 Run/Stop 按钮已合并到左侧控件，移除冗余底部快捷按钮
        self.autosave_cb = QtWidgets.QCheckBox('运行时自动保存配置')
        bb_layout.addWidget(self.autosave_cb)
        bb_layout.addStretch(1)
        save_log_btn = QtWidgets.QPushButton('保存日志')
        save_log_btn.clicked.connect(self.on_save_logs)
        bb_layout.addWidget(save_log_btn)
        copy_log_btn = QtWidgets.QPushButton('复制日志')
        copy_log_btn.clicked.connect(self.on_copy_logs)
        bb_layout.addWidget(copy_log_btn)
        main_layout.addWidget(bottom_bar)
        # 底部高度保护，保证控制栏/日志可用性
        try:
            bottom_bar.setMinimumHeight(140)
        except Exception:
            pass

        # 自动加载 train.csv
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(self.base_dir, 'train.csv')
        if os.path.exists(csv_path):
            self.train_csv = csv_path
            self.train_csv_lbl.setText(csv_path)
            try:
                self.df = pd.read_csv(csv_path).sort_values('date_id').reset_index(drop=True)
                self.append_log(f'已加载 {len(self.df)} 行数据：{csv_path}')
            except Exception as e:
                self.df = None
                self.append_log('加载 train.csv 失败：' + str(e))
        else:
            self.df = None
            self.train_csv = None
            self.append_log('未在当前目录找到 train.csv')

        # 尝试加载 Kaggle 官方评分脚本（如果存在）
        self.kaggle_score_mod = None
        score_path = os.path.join(self.base_dir, 'Hull Competition Sharpe.py')
        if os.path.exists(score_path):
            try:
                spec = importlib.util.spec_from_file_location('kaggle_score', score_path)
                kag = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kag)
                # 期望存在函数 score(solution, submission, row_id_column_name)
                if hasattr(kag, 'score'):
                    self.kaggle_score_mod = kag
                    self.append_log('已加载 Kaggle 评分模块')
            except Exception as e:
                self.append_log('加载 Kaggle 评分模块失败：' + str(e))

        # populate run history from existing run_info files
        try:
            self.refresh_run_history()
        except Exception:
            pass

        # connect run_table double click
        self.run_table.cellDoubleClicked.connect(lambda r,c: self.on_run_table_double_clicked(r,c))

    def append_log(self, msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        self.log.appendPlainText(f'[{ts}] {msg}')

    # ------------------------- Config IO & Presets -------------------------
    def on_save_config(self):
        cfg = self.build_cfg_from_ui()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存配置为 JSON', os.path.join(self.base_dir, 'config.json'), 'JSON 文件 (*.json)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self.append_log(f'配置已保存: {path}')
        except Exception as e:
            self.append_log('保存配置失败：' + str(e))

    def on_load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '加载配置 JSON', self.base_dir, 'JSON 文件 (*.json)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # apply a few known values (partial apply)
            if 'save_dir' in cfg:
                self.save_dir_edit.setText(cfg.get('save_dir'))
            if 'model' in cfg:
                # try to select model in combo
                model = cfg.get('model')
                idx = self.model_combo.findData(model)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            # 尝试恢复预设（preset）到界面
            if 'preset' in cfg:
                try:
                    pres = cfg.get('preset')
                    pidx = self.presets_combo.findData(pres)
                    if pidx >= 0:
                        self.presets_combo.setCurrentIndex(pidx)
                    else:
                        pidx = self.presets_combo.findText(pres)
                        if pidx >= 0:
                            self.presets_combo.setCurrentIndex(pidx)
                except Exception:
                    pass
            self.append_log('配置已加载（部分应用）: ' + path)
        except Exception as e:
            self.append_log('加载配置失败：' + str(e))


    def build_cfg_from_ui(self):
        cfg = {}
        cfg['train_csv'] = self.train_csv
        # include currently selected preset (prefer userData key, fall back to display text)
        try:
            pres_val = self.presets_combo.currentData()
            if not pres_val:
                pres_val = self.presets_combo.currentText()
            cfg['preset'] = pres_val
        except Exception:
            cfg['preset'] = None
        # 获取内部模型标识（userData），若不存在则回退到显示文本
        model_val = self.model_combo.currentData()
        if model_val is None:
            model_val = self.model_combo.currentText()
        cfg['model'] = model_val
        cfg['seq_len'] = int(self.seq_spin.value())
        # NN common
        cfg['epochs'] = int(self.epochs_spin.value())
        try:
            cfg['lr'] = float(self.lr_edit.text())
        except Exception:
            cfg['lr'] = 3e-4
        cfg['lambda_turnover'] = float(self.lambda_edit.text())
        cfg['l2_reg'] = float(self.l2_edit.text())
        cfg['batch_size'] = int(self.batch_spin.value())
        cfg['save_dir'] = self.save_dir_edit.text()
        cfg['sim_epochs'] = int(self.sim_epochs_spin.value())
        cfg['val_ratio'] = 0.1
        cfg['pretrain_epochs'] = 0
        cfg['no_cuda'] = False
        # global/data flags
        # prefer the internal userData key (english code), fall back to display text
        try:
            if hasattr(self, 'missing_strategy_combo'):
                mv = self.missing_strategy_combo.currentData()
                if mv is None:
                    mv = self.missing_strategy_combo.currentText()
                cfg['missing_value_strategy'] = mv
            else:
                cfg['missing_value_strategy'] = 'median'
        except Exception:
            cfg['missing_value_strategy'] = 'median'
        cfg['add_is_missing'] = bool(self.add_is_missing_cb.isChecked()) if hasattr(self, 'add_is_missing_cb') else False
        # validation
        cv_val = self.cv_method_combo.currentData()
        if cv_val is None:
            cv_val = self.cv_method_combo.currentText()
        cfg['cv_method'] = cv_val
        cfg['n_splits'] = int(self.n_splits_spin.value())
        cfg['embargo_days'] = int(self.embargo_spin.value())
        cfg['label_horizon'] = int(self.label_horizon_spin.value())
        # Walk-Forward params
        cv_val = self.cv_method_combo.currentData()
        if cv_val is None:
            cv_val = self.cv_method_combo.currentText()
        if str(cv_val).lower().startswith('walk'):
            cfg['train_window'] = int(self.wf_train_spin.value())
            cfg['val_window'] = int(self.wf_val_spin.value())
            cfg['step'] = int(self.wf_step_spin.value())
        # model-specific options
        m = cfg['model'].lower()
        if m in ('lstm', 'mlp'):
            nn_type_val = self.nn_type_combo.currentData()
            if nn_type_val is None:
                nn_type_val = self.nn_type_combo.currentText()
            cfg['nn_type'] = nn_type_val
            cfg['hidden_dim'] = int(self.hidden_dim_spin.value())
            cfg['n_layers'] = int(self.n_layers_spin.value())
            cfg['dropout'] = float(self.dropout_spin.value())
            cfg['pretrain_epochs'] = int(self.pretrain_epochs.value())
            cfg['use_double_precision'] = bool(self.use_double_precision_cb.isChecked())
            # mlp hidden layers
            if cfg['nn_type'] == 'mlp':
                cfg['mlp_hidden'] = str(self.mlp_hidden_edit.text())
            else:
                # for lstm, ensure hidden_dim/n_layers present
                cfg['hidden_dim'] = int(self.hidden_dim_spin.value())
        elif 'lightgbm' in m or m == 'lgbm':
            cfg['lgb'] = {
                'learning_rate': float(self.lgb_learning_rate.value()),
                'num_leaves': int(self.lgb_num_leaves.value()),
                'min_child_samples': int(self.lgb_min_child_samples.value()),
                'feature_fraction': float(self.lgb_feature_fraction.value()),
                'bagging_fraction': float(self.lgb_bagging_fraction.value()),
                'bagging_freq': int(self.lgb_bagging_freq.value()),
                'reg_alpha': float(self.lgb_reg_alpha.value()),
                'reg_lambda': float(self.lgb_reg_lambda.value()),
                'num_boost_round': int(self.lgb_num_boost_round.value()),
                'early_stopping_rounds': int(self.lgb_early_stopping.value()),
                'use_optuna': bool(self.lgb_use_optuna.isChecked()),
                'optuna_n_trials': int(self.lgb_optuna_trials.value())
            }
        elif m == 'ppo':
            cfg['ppo'] = {
                'window_len': int(self.ppo_window_len.value()),
                'episode_len': int(self.ppo_episode_len.value()),
                'turnover_cost': float(self.ppo_turnover_cost.value()),
                'ppo_lr': float(self.ppo_lr.value()),
                'n_updates': int(self.ppo_n_updates.value()),
                'steps_per_update': int(self.ppo_steps_per_update.value()),
                'ppo_epochs': int(self.ppo_epochs.value()),
                'ppo_minibatch': int(self.ppo_minibatch.value()),
                'ppo_clip': float(self.ppo_clip.value()),
                'entropy_coef': float(self.ppo_entropy_coef.value())
            }

        # 如果用户选择了预设，则把预设对应的默认参数合并到 cfg 中，便于保存配置时携带预设语义
        try:
            pres = cfg.get('preset')
            preset_map = {
                'Conservative': {'lgb': {'learning_rate': 0.01, 'num_leaves': 16}, 'target_features': 150},
                'Balanced': {'lgb': {'learning_rate': 0.01, 'num_leaves': 31}, 'target_features': 250},
                'Aggressive': {'lgb': {'learning_rate': 0.02, 'num_leaves': 64}, 'target_features': 400}
            }
            if pres in preset_map:
                pvals = preset_map[pres]
                # 合并 lgb 子配置
                if 'lgb' in pvals:
                    if 'lgb' not in cfg:
                        cfg['lgb'] = {}
                    for kk, vv in pvals['lgb'].items():
                        # only set if not present to avoid覆盖用户显式修改
                        if kk not in cfg['lgb'] or cfg['lgb'].get(kk) is None:
                            cfg['lgb'][kk] = vv
                # 合并其他顶级字段（例如 target_features）
                if 'target_features' in pvals and ('target_features' not in cfg or cfg.get('target_features') is None):
                    cfg['target_features'] = pvals['target_features']
        except Exception:
            pass

        return cfg

    def on_save_logs(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存日志', os.path.join(self.base_dir, 'run.log'), '文本文件 (*.txt)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.log.toPlainText())
            self.append_log('日志已保存: ' + path)
        except Exception as e:
            self.append_log('保存日志失败：' + str(e))

    def on_copy_logs(self):
        try:
            cb = QtWidgets.QApplication.clipboard()
            cb.setText(self.log.toPlainText())
            self.append_log('日志已复制到剪贴板')
        except Exception as e:
            self.append_log('复制日志失败：' + str(e))

    # ------------------------- Run history / run_info integration -------------------------
    def refresh_run_history(self):
        # scan save_dir for run_info_*.json files and populate self.run_list
        save_dir = os.path.join(self.base_dir, self.save_dir_edit.text()) if not os.path.isabs(self.save_dir_edit.text()) else self.save_dir_edit.text()
        if not os.path.exists(save_dir):
            return
        files = [f for f in os.listdir(save_dir) if f.startswith('run_info_') and f.endswith('.json')]
        rows = sorted(files, reverse=True)
        self.run_table.setRowCount(0)
        for fn in rows:
            p = os.path.join(save_dir, fn)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    rj = json.load(f)
                kpis = rj.get('kpis', {})
                ts = rj.get('timestamp', '')
                model = rj.get('model', '')
                adj = kpis.get('adjusted_sharpe') if isinstance(kpis, dict) else None
                rmse = kpis.get('rmse') if isinstance(kpis, dict) else None
                row = self.run_table.rowCount()
                self.run_table.insertRow(row)
                it_ts = QtWidgets.QTableWidgetItem(str(ts))
                it_ts.setData(QtCore.Qt.UserRole, p)
                self.run_table.setItem(row, 0, it_ts)
                self.run_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(model)))
                self.run_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(adj):.4f}" if adj is not None else ''))
                self.run_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(rmse):.6f}" if rmse is not None else ''))
                self.run_table.setItem(row, 4, QtWidgets.QTableWidgetItem(p))
            except Exception:
                continue

    def on_run_item_double_clicked(self, item):
        path = item.data(QtCore.Qt.UserRole)
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rj = json.load(f)
            self.append_log('运行信息已加载：' + path)
            # show kpis
            k = rj.get('kpis', {})
            self.append_log('关键指标：' + json.dumps(k))
            # show feature importance if present
            fi = rj.get('feature_importance')
            if fi:
                self.show_feature_importance(fi)
        except Exception as e:
            self.append_log('加载 run_info 失败: ' + str(e))

    def on_run_table_double_clicked(self, row, col):
        try:
            item = self.run_table.item(row, 0)
            if item is None:
                return
            path = item.data(QtCore.Qt.UserRole)
            if not path:
                return
            with open(path, 'r', encoding='utf-8') as f:
                rj = json.load(f)
            self.append_log('运行信息已加载：' + path)
            k = rj.get('kpis', {})
            self.append_log('关键指标：' + json.dumps(k))
            fi = rj.get('feature_importance')
            if fi:
                self.show_feature_importance(fi)
        except Exception as e:
            self.append_log('加载 run_info 失败: ' + str(e))

    # ----------------- Run history actions -----------------
    def on_load_selected_run(self):
        sel = self.run_table.selectionModel().selectedRows()
        if not sel:
            self.append_log('未选中任何运行')
            return
        row = sel[0].row()
        self.on_run_table_double_clicked(row, 0)

    def on_rerun_selected(self):
        sel = self.run_table.selectionModel().selectedRows()
        if not sel:
            self.append_log('未选中要重跑的运行')
            return
        row = sel[0].row()
        path_item = self.run_table.item(row, 0)
        path = path_item.data(QtCore.Qt.UserRole) if path_item is not None else None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rj = json.load(f)
            # extract cfg hints from run_info: prefer lgb_cfg or params
            cfg = rj.get('lgb_cfg') or rj.get('params') or {}
            # ensure train_csv points to current csv
            cfg['train_csv'] = self.train_csv
            cfg['model'] = 'lightgbm'
            cfg['save_dir'] = self.save_dir_edit.text() or cfg.get('save_dir', '.')
            self.append_log('从 run_info 启动重跑...')
            lgb_worker = LGBRunWorker(self.df, cfg)
            self.active_lgb_worker = lgb_worker
            # 禁用左侧运行按钮以防重复启动
            self.sim_btn.setEnabled(False)
            lgb_worker.signals.progress.connect(self.append_log)
            lgb_worker.signals.finished.connect(lambda res: (self.append_log('重跑完成'), self.refresh_run_history()))
            lgb_worker.signals.error.connect(lambda e: self.append_log('重跑错误:\n' + e))
            self.pool.start(lgb_worker)
        except Exception as e:
            self.append_log('重跑失败：' + str(e))

    def on_download_selected_run(self):
        sel = self.run_table.selectionModel().selectedRows()
        if not sel:
            self.append_log('未选中任何运行可下载')
            return
        row = sel[0].row()
        path = self.run_table.item(row, 0).data(QtCore.Qt.UserRole)
        if not path or not os.path.exists(path):
            self.append_log('run_info 文件不存在')
            return
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存运行信息为', os.path.basename(path), 'JSON 文件 (*.json)')
        if not dst:
            return
        try:
            shutil.copyfile(path, dst)
            self.append_log('已下载 run_info: ' + dst)
        except Exception as e:
            self.append_log('下载 run_info 失败：' + str(e))

    def on_compare_selected(self):
        sel = self.run_table.selectionModel().selectedRows()
        if not sel or len(sel) < 2:
            self.append_log('请选择至少两个运行以比较')
            return
        runs = []
        for s in sel:
            row = s.row()
            item = self.run_table.item(row, 0)
            p = item.data(QtCore.Qt.UserRole) if item is not None else None
            if not p:
                continue
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    rj = json.load(f)
                runs.append((p, rj))
            except Exception:
                self.append_log('无法读取：' + str(p))
        if not runs:
            self.append_log('未能加载任何 run_info 用于比较')
            return
        # Try to plot cumulative returns + positions if available; otherwise fallback to feature importance comparison
        try:
            has_series = all(('date_ids' in rj and 'positions' in rj and 'fr' in rj and 'rf' in rj) for (_, rj) in runs)
            if has_series:
                # build subplot rows: top cumulative returns, bottom positions
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=['Cumulative Returns','Positions'])
                for p, rj in runs:
                    date_ids = rj.get('date_ids')
                    positions = rj.get('positions')
                    fr = rj.get('fr')
                    rf = rj.get('rf')
                    # ensure lists
                    x = list(date_ids)
                    strat_excess = [ (f - r) for f,r in zip(fr, rf) ]
                    cum_strat = []
                    acc = 1.0
                    for v in strat_excess:
                        acc = acc * (1 + v)
                        cum_strat.append(acc - 1)
                    fig.add_trace(go.Scatter(x=x, y=cum_strat, name=os.path.basename(p)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=x, y=list(positions), name=os.path.basename(p)), row=2, col=1)
                fig.update_layout(height=700, title_text='比较：累计收益与仓位')
                html = fig.to_html(include_plotlyjs='cdn')
                self.web.setHtml(html)
                # try to save HTML and PNG
                try:
                    fd, html_path = tempfile.mkstemp(prefix='compare_', suffix='.html')
                    os.close(fd)
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html)
                    self.append_log('比较 HTML 已保存: ' + html_path)
                    # try PNG via kaleido if available
                    try:
                        fig.write_image(html_path.replace('.html','.png'))
                        self.append_log('比较 PNG 已保存: ' + html_path.replace('.html','.png'))
                    except Exception:
                        self.append_log('生成 PNG 失败（缺少 kaleido 或其他支持），已保存 HTML')
                except Exception as e:
                    self.append_log('保存比较结果失败：' + str(e))
            else:
                # fallback: feature importance side-by-side
                n = len(runs)
                fig = make_subplots(rows=1, cols=n, subplot_titles=[os.path.basename(r[0]) for r in runs])
                for i, (p, rj) in enumerate(runs):
                    fi = rj.get('feature_importance') or []
                    topn = fi[:30]
                    names = [x['feature'] for x in topn]
                    imps = [x['importance'] for x in topn]
                    fig.add_trace(go.Bar(x=imps[::-1], y=names[::-1], orientation='h', name=os.path.basename(p)), row=1, col=i+1)
                fig.update_layout(height=400, title_text='比较：特征重要性')
                self.web.setHtml(fig.to_html(include_plotlyjs='cdn'))
            self.append_log('比较视图已生成')
        except Exception as e:
            self.append_log('生成比较视图失败：' + str(e))

    def on_stop_requested(self):
        # Attempt to cancel active lgb worker if present
        try:
            if hasattr(self, 'active_lgb_worker') and self.active_lgb_worker is not None:
                w = self.active_lgb_worker
                if hasattr(w, 'cancel'):
                    w.cancel()
                    self.append_log('已发送取消请求到运行中的作业')
                else:
                    self.append_log('当前作业不支持取消')
            else:
                self.append_log('当前没有运行中的可取消作业')
        except Exception as e:
            self.append_log('停止请求失败：' + str(e))

    def show_feature_importance(self, fi_list):
        # fi_list: list of {feature, importance}
        try:
            topn = fi_list[:50]
            names = [x['feature'] for x in topn]
            imps = [x['importance'] for x in topn]
            fig = go.Figure([go.Bar(x=imps[::-1], y=names[::-1], orientation='h')])
            fig.update_layout(title='特征重要性（前 {}）'.format(len(topn)), height=400)
            self.web.setHtml(fig.to_html(include_plotlyjs='cdn'))
        except Exception as e:
            self.append_log('绘制特征重要性失败: ' + str(e))

    def on_model_changed(self, idx):
        try:
            # 优先使用 userData（模型内部标识），回退到显示文本
            model_val = self.model_combo.currentData()
            if model_val is None:
                txt = self.model_combo.currentText().lower()
            else:
                txt = str(model_val).lower()
            # map lstm/mlp -> nn panel (index 0), lightgbm -> index 1
            if txt in ('lstm', 'mlp'):
                self.model_stack.setCurrentIndex(0)
                self.append_log(f'已选择模型 {txt}，显示 NN 面板')
            elif txt in ('lightgbm', 'lgbm', 'lgb'):
                self.model_stack.setCurrentIndex(1)
                self.append_log('已选择 LightGBM，显示 LightGBM 面板')
            else:
                # default to NN
                self.model_stack.setCurrentIndex(0)
                self.append_log(f'已选择模型 {txt}，默认显示 NN 面板')
        except Exception:
            # do not raise in UI callback
            traceback.print_exc()

    def on_cv_method_changed(self, idx):
        """Toggle visibility of Walk-Forward specific controls."""
        try:
            cv_val = self.cv_method_combo.currentData()
            if cv_val is None:
                cv_val = self.cv_method_combo.currentText()
            if str(cv_val).lower().startswith('walk'):
                self.wf_train_label.setVisible(True)
                self.wf_train_spin.setVisible(True)
                self.wf_val_label.setVisible(True)
                self.wf_val_spin.setVisible(True)
                self.wf_step_label.setVisible(True)
                self.wf_step_spin.setVisible(True)
            else:
                self.wf_train_label.setVisible(False)
                self.wf_train_spin.setVisible(False)
                self.wf_val_label.setVisible(False)
                self.wf_val_spin.setVisible(False)
                self.wf_step_label.setVisible(False)
                self.wf_step_spin.setVisible(False)
            # hide embargo warning when toggling
            try:
                self.embargo_warn_label.setVisible(False)
            except Exception:
                pass
        except Exception:
            traceback.print_exc()

    def on_select_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择 CSV 文件', os.path.expanduser('~'), 'CSV 文件 (*.csv);;所有文件 (*)')
        if path:
            self.train_csv = path
            self.train_csv_lbl.setText(path)
            self.append_log(f'已选择 CSV：{path}')

    def on_verify_cv(self):
        # 基本静态检查：数据存在、列齐全、禁封 vs 最大滞后
        if getattr(self, 'df', None) is None:
            self.append_log('验证 CV：未加载数据')
            return
        df = self.df
        # 检查必需列
        for col in ['forward_returns', 'risk_free_rate', 'date_id']:
            if col not in df.columns:
                self.append_log(f'验证 CV 失败：缺少列 {col}')
                return
        # 如果存在滞后列则计算最大滞后
        lag_cols = [c for c in df.columns if '_lag_' in c]
        max_lag = 0
        if lag_cols:
            try:
                max_lag = max(int(c.split('_lag_')[-1]) for c in lag_cols)
            except Exception:
                max_lag = 0
        embargo = int(self.embargo_spin.value())
        if embargo < max_lag:
            msg = f'验证 CV 警告：禁封天数 ({embargo}) 小于最大滞后 ({max_lag})'
            self.append_log(msg)
            try:
                self.embargo_warn_label.setText('警告：embargo_days 小于最大滞后，可能存在信息泄露，请调整或确认。')
                self.embargo_warn_label.setVisible(True)
            except Exception:
                pass
            self.append_log('验证 CV：失败')
            return
        else:
            # hide inline warning if previously shown
            try:
                self.embargo_warn_label.setVisible(False)
            except Exception:
                pass
            self.append_log('验证 CV：通过（静态检查）。启动快速检查（小样本）...')

        # 启动快速检查工作线程：运行一个小型训练以验证端到端流程
        try:
            cfg = self.build_cfg_from_ui()
            # quick safety: if user requested heavy Optuna, confirm first
            try:
                lgb_cfg = cfg.get('lgb') or {}
                if lgb_cfg.get('use_optuna'):
                    if not self.ask_user_confirmation('启用 Optuna 搜索','您已启用 Optuna 超参数搜索，可能会运行较长时间。请确认要继续并勾选“我已阅读并理解”以继续。'):
                        self.append_log('已取消：用户未确认 Optuna 搜索')
                        # restore UI if necessary
                        self.verify_progress.setVisible(False)
                        self.verify_cv_btn.setEnabled(True)
                        self.verify_cv_btn.setText('验证 CV / 分数一致性')
                        return
            except Exception:
                pass
            # 提交 QuickCheck 作业到本地作业队列（由独立 worker 执行），以避免在 UI 进程运行 heavy work
            try:
                submit_cfg = cfg.copy()
                submit_cfg.update({
                    'quickcheck': True,
                    'sample_rows': 200,
                    'epochs': int(cfg.get('qc_epochs', 1)) if cfg.get('qc_epochs') is not None else 1,
                    'no_cuda': True,
                })
                job_id = job_queue.submit_job(submit_cfg)
                self.append_log(f'已提交 QuickCheck 作业：{job_id}，等待 worker 处理')
                # disable UI while pending
                self.verify_progress.setVisible(True)
                self.verify_cv_btn.setEnabled(False)
                self.verify_cv_btn.setText('已提交 QuickCheck')
                # start polling for job completion
                self._start_job_poll(job_id, button=self.verify_cv_btn, progress_widget=self.verify_progress, tag='QuickCheck')
            except Exception as e:
                self.append_log('提交 QuickCheck 作业失败：' + str(e))
        except Exception as e:
                self.append_log('启动快速检查失败：' + str(e))

    def ask_user_confirmation(self, title, message):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        cb = QtWidgets.QCheckBox('我已阅读并理解')
        layout.addWidget(cb)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        def on_accept():
            if cb.isChecked():
                dlg.accept()
            else:
                QtWidgets.QMessageBox.warning(self, '需要确认', '请勾选“我已阅读并理解”以继续')
        btns.accepted.connect(on_accept)
        btns.rejected.connect(dlg.reject)
        res = dlg.exec()
        return res == QtWidgets.QDialog.Accepted

    # ----- Job polling helpers (integrate with hull/job_queue.py) -----
    def _start_job_poll(self, job_id: str, button: QtWidgets.QPushButton = None, progress_widget: QtWidgets.QWidget = None, tag: str = 'Job'):
        """Start a polling loop (QTimer singleShot) to monitor job status."""
        # store timer id per job so we can cancel if needed
        def poll_once():
            try:
                st = job_queue.get_job_status(job_id)
                status = st.get('status')
                if status == 'queued' or status == 'processing':
                    # still running — schedule next poll
                    QtCore.QTimer.singleShot(2000, poll_once)
                elif status == 'finished':
                    # read result.json and display
                    paths = job_queue.job_path(job_id)
                    run_dir = paths.get('run_dir')
                    result_file = os.path.join(run_dir, 'result.json')
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            res = json.load(f)
                        rc = res.get('returncode')
                        stdout = res.get('stdout')
                        stderr = res.get('stderr')
                        self.append_log(f"{tag} 完成：job_id={job_id} returncode={rc}")
                        if stdout:
                            self.append_log(f"{tag} stdout: {stdout[:200]}")
                        if stderr:
                            self.append_log(f"{tag} stderr: {stderr[:200]}")
                    except Exception as e:
                        self.append_log(f'{tag} 完成但无法读取结果：{e}')
                    finally:
                        # restore UI
                        if progress_widget is not None:
                            try:
                                progress_widget.setVisible(False)
                            except Exception:
                                pass
                        if button is not None:
                            try:
                                button.setEnabled(True)
                                # restore original text if known
                                button.setText('验证 CV / 分数一致性' if tag == 'QuickCheck' else '运行')
                            except Exception:
                                pass
                else:
                    # missing or unknown
                    self.append_log(f'{tag} 状态异常：{status}')
                    if progress_widget is not None:
                        try:
                            progress_widget.setVisible(False)
                        except Exception:
                            pass
                    if button is not None:
                        try:
                            button.setEnabled(True)
                            button.setText('验证 CV / 分数一致性')
                        except Exception:
                            pass
            except Exception as e:
                self.append_log(f'轮询作业时出错：{e}')
                if progress_widget is not None:
                    try:
                        progress_widget.setVisible(False)
                    except Exception:
                        pass
                if button is not None:
                    try:
                        button.setEnabled(True)
                        button.setText('验证 CV / 分数一致性')
                    except Exception:
                        pass

        # start first poll immediately
        QtCore.QTimer.singleShot(1000, poll_once)

    def load_model_from_pickle(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        model_type = obj.get('model_type', 'lstm')
        cfg = obj.get('model_cfg', {})
        state = obj.get('model_state', {})
        scaler = obj.get('scaler', None)
        feature_cols = obj.get('feature_cols')
        if model_type == 'lstm':
            m = tas.SimpleLSTMPolicy(**cfg)
        else:
            m = tas.SimpleMLP(input_dim=cfg.get('input_dim', len(feature_cols)), hidden_layers=cfg.get('mlp_hidden', [128,64]))
        state_t = {}
        for k, v in state.items():
            try:
                state_t[k] = torch.from_numpy(v)
            except Exception:
                state_t[k] = v
        m.load_state_dict(state_t)
        m.eval()
        return m, scaler, feature_cols

    def update_plots(self, date_ids, positions, fr, rf):
        self.pg_plot.clear()
        if len(positions) == 0:
            return
        # 权益曲线
        strategy_daily = rf * (1 - positions) + positions * fr
        strategy_excess = strategy_daily - rf
        cum_strategy = np.cumprod(1 + strategy_excess) - 1
        p1 = self.pg_plot.plot(date_ids, cum_strategy, pen=pg.mkPen('g', width=2), name='cum_strategy')
        # 仓位
        p2 = self.pg_plot.plot(date_ids, positions, pen=pg.mkPen('b', width=1), name='positions')

        # 构建 Plotly KPI 报告并显示
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date_ids.tolist(), y=cum_strategy.tolist(), name='策略累计'))
        fig.add_trace(go.Scatter(x=date_ids.tolist(), y=np.cumprod(1 + (fr - rf)) - 1, name='市场累计'))
        fig.update_layout(title='权益曲线', height=400)
        html = fig.to_html(include_plotlyjs='cdn')
        self.web.setHtml(html)

    def on_show_diagnostics(self):
        """读取最近的 run_info 与 diagnostic CSV，生成直观图表并在诊断/概览面板显示。"""
        try:
            save_dir = os.path.join(self.base_dir, self.save_dir_edit.text()) if not os.path.isabs(self.save_dir_edit.text()) else self.save_dir_edit.text()
            if not os.path.exists(save_dir):
                self.append_log('显示诊断图失败：保存目录不存在: ' + str(save_dir))
                return
            # 找最近的 run_info 文件
            files = [f for f in os.listdir(save_dir) if f.startswith('run_info_') and f.endswith('.json')]
            if not files:
                self.append_log('显示诊断图：未找到任何 run_info 文件')
                return
            files = sorted(files, reverse=True)
            run_path = os.path.join(save_dir, files[0])
            with open(run_path, 'r', encoding='utf-8') as f:
                rj = json.load(f)

            figs = []
            # 1) KPI 条形图
            kpis = rj.get('kpis') or {}
            if isinstance(kpis, dict) and len(kpis) > 0:
                names = list(kpis.keys())
                vals = [float(kpis.get(n) or 0.0) for n in names]
                f1 = go.Figure([go.Bar(x=names, y=vals, marker_color='steelblue')])
                f1.update_layout(title='关键指标', height=320, margin=dict(t=30))
                figs.append(f1)

            # 2) 如果存在时间序列数据：绘制累计收益与仓位
            if all(k in rj for k in ('date_ids', 'fr', 'rf', 'positions')):
                date_ids = rj.get('date_ids')
                fr = rj.get('fr')
                rf = rj.get('rf')
                positions = rj.get('positions')
                strat_excess = [ (f - r) for f, r in zip(fr, rf) ]
                cum_strat = []
                acc = 1.0
                for v in strat_excess:
                    acc = acc * (1 + v)
                    cum_strat.append(acc - 1)
                market_cum = np.cumprod(1 + (np.array(fr) - np.array(rf))) - 1
                f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=['累计收益','仓位'])
                f2.add_trace(go.Scatter(x=date_ids, y=cum_strat, name='策略累计', line=dict(color='green')), row=1, col=1)
                f2.add_trace(go.Scatter(x=date_ids, y=market_cum.tolist(), name='市场累计', line=dict(color='gray')), row=1, col=1)
                f2.add_trace(go.Scatter(x=date_ids, y=positions, name='仓位', line=dict(color='blue')), row=2, col=1)
                f2.update_layout(height=620, title='系列：累计收益与仓位', margin=dict(t=30))
                figs.append(f2)

            # 3) diagnostic_problem_rows.csv -> 缺失率柱状图（Top N）
            diag_csv = os.path.join(save_dir, 'diagnostic_problem_rows.csv')
            if os.path.exists(diag_csv):
                try:
                    ddf = pd.read_csv(diag_csv)
                    miss = ddf.isna().mean().sort_values(ascending=False)
                    topn = miss.head(40)
                    f3 = go.Figure([go.Bar(x=topn.index.tolist(), y=topn.values.tolist(), marker_color='indianred')])
                    f3.update_layout(title='诊断 CSV：缺失率（前 40）', height=420, margin=dict(t=30))
                    figs.append(f3)
                except Exception:
                    self.append_log('读取 diagnostic_problem_rows.csv 失败（可能已损坏）')

            if not figs:
                self.append_log('诊断：没有可视化的数据（run_info 未包含 KPI/时间序列，且没有 diagnostic CSV）')
                return

            # 把所有 Figure 的 HTML 串联并显示
            html = ''.join([fig.to_html(include_plotlyjs='cdn') for fig in figs])
            self.web.setHtml(html)
            self.append_log('诊断视图已生成: ' + run_path)
        except Exception as e:
            self.append_log('生成诊断视图失败：' + str(e))

    def on_run_simulation(self):
        if self.df is None:
            self.append_log('未加载到数据')
            return
        cfg = self.build_cfg_from_ui()
        # Basic pre-run validation: ensure sequence length is not larger than dataset
        try:
            seq_len = int(cfg.get('seq_len', int(self.seq_spin.value())))
            if seq_len > len(self.df):
                QtWidgets.QMessageBox.warning(self, '序列长度错误', f'序列长度 (seq_len={seq_len}) 大于数据行数 ({len(self.df)})，请减小序列长度或加载更多数据。')
                self.append_log(f'已阻止运行：seq_len ({seq_len}) > 数据行数 ({len(self.df)})')
                return
        except Exception:
            pass
        cfg['save_dir'] = self.save_dir_edit.text()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_cols = [c for c in self.df.columns if c not in {'date_id','forward_returns','risk_free_rate','market_forward_excess_returns'}]
        m = str(cfg.get('model', '')).lower()
        if 'lightgbm' in m or 'lgb' in m:
            # Run LightGBM via subprocess runner (wrapped in LGBRunWorker)
            self.append_log('启动 LightGBM 训练 (子进程隔离)...')
            # if Optuna enabled, confirm with user
            try:
                lgb_cfg = cfg.get('lgb') or {}
                if lgb_cfg.get('use_optuna'):
                    if not self.ask_user_confirmation('启用 Optuna 搜索','您已启用 Optuna 超参数搜索，可能会运行较长时间。请确认要继续并勾选“我已阅读并理解”以继续。'):
                        self.append_log('已取消：用户未确认 Optuna 搜索')
                        return
            except Exception:
                pass
            lgb_worker = LGBRunWorker(self.df, cfg)
            # keep reference to allow cancellation
            self.active_lgb_worker = lgb_worker
            # UI state: 禁用左侧运行按钮以防重复启动
            self.sim_btn.setEnabled(False)

            lgb_worker.signals.progress.connect(self.append_log)

            def lgb_finished(res):
                try:
                    kpis = res.get('kpis') if isinstance(res, dict) else None
                    self.append_log('LightGBM 训练完成；KPIs: ' + str(kpis))
                    # show run_info if available
                    run_info = res.get('run_info') if isinstance(res, dict) else None
                    if run_info and isinstance(run_info, str):
                        try:
                            with open(run_info, 'r', encoding='utf-8') as f:
                                rj = json.load(f)
                                self.append_log('运行信息：' + json.dumps(rj.get('kpis', {})))
                        except Exception:
                            pass
                    # refresh run history to show newly saved run_info
                    try:
                        self.refresh_run_history()
                    except Exception:
                        pass
                finally:
                    self.append_log('LightGBM 运行结束')
                    # clear active worker and restore UI
                    try:
                        self.active_lgb_worker = None
                    except Exception:
                        pass
                    # restore UI: 启用左侧运行按钮
                    self.sim_btn.setEnabled(True)

            def lgb_error(e):
                self.append_log('LightGBM 运行错误：\n' + e)
                try:
                    self.active_lgb_worker = None
                except Exception:
                    pass
                # restore UI after error
                self.sim_btn.setEnabled(True)

            lgb_worker.signals.finished.connect(lgb_finished)
            lgb_worker.signals.error.connect(lambda e: self.append_log('错误：\n' + e))
            self.pool.start(lgb_worker)
        else:
            # if PPO mode, confirm heavy operation
            try:
                if m == 'ppo':
                    if not self.ask_user_confirmation('PPO 训练','您选择了 PPO 训练，此操作可能非常耗时且占用大量资源。请确认要继续并勾选“我已阅读并理解”以继续。'):
                        self.append_log('已取消：用户未确认 PPO 训练')
                        return
            except Exception:
                pass
            sim = SimulationWorker(self.df, feature_cols, cfg, seq_len=int(self.seq_spin.value()), model_type=cfg['model'], last_k=int(self.lastk_spin.value()), device=device)
            sim.signals.progress.connect(self.append_log)
            sim.signals.finished.connect(self.on_simulation_finished)
            sim.signals.error.connect(lambda e: self.append_log('错误：\n' + e))
            self.pool.start(sim)
            self.append_log('模拟已启动')

    def on_simulation_finished(self, out):
        self.append_log('模拟完成；调整后夏普=%.6f' % out.get('adj_sharpe', float('nan')))
        self.update_plots(out['date_ids'], out['positions'], out['fr'], out['rf'])
        # 通过 Plotly 显示 KPI 表
        stats = out.get('stats', {})
        kpis = [{'metric': k, 'value': float(v)} for k, v in stats.items()]
        # 如果存在 Kaggle 评分模块，计算官方分数
        try:
            if self.kaggle_score_mod is not None and len(out.get('date_ids', []))>1:
                sol = pd.DataFrame({'date_id': out['date_ids'], 'forward_returns': out['fr'], 'risk_free_rate': out['rf']})
                sub = pd.DataFrame({'date_id': out['date_ids'], 'prediction': out['positions']})
                try:
                    kag_score = float(self.kaggle_score_mod.score(sol.copy(), sub.copy(), row_id_column_name='date_id'))
                    kpis.append({'metric': 'kaggle_adj_sharpe', 'value': kag_score})
                    self.append_log(f'Kaggle 官方调整后夏普：{kag_score:.6f}')
                except Exception as e:
                    self.append_log('Kaggle 评分失败：' + str(e))
        except Exception:
            pass
        # 把常见指标名翻译成中文便于展示
        name_map = {
            'adjusted_sharpe': '调整后夏普',
            'sharpe': '夏普',
            'strategy_vol': '策略波动率(%)',
            'market_vol': '市场波动率(%)',
            'turnover_mean': '平均换手',
            'cumulative_strategy_excess': '策略累计超额收益',
            'cumulative_market_excess': '市场累计超额收益',
            'kaggle_adj_sharpe': 'Kaggle 官方调整后夏普'
        }
        display_metrics = [ (name_map.get(item['metric'], item['metric']), item['value']) for item in kpis ]
        if len(display_metrics)==0:
            display_metrics = [('message','无 KPI')]
        fig = go.Figure(data=[go.Table(header=dict(values=['指标','数值']), cells=dict(values=[[r[0] for r in display_metrics],[r[1] for r in display_metrics]]))])
        self.web.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def closeEvent(self, event: QtGui.QCloseEvent):
        # persist window geometry and splitter state
        try:
            settings = QtCore.QSettings('hull', 'trainer_gui')
            settings.setValue('geometry', self.saveGeometry())
            try:
                settings.setValue('splitter', self.main_splitter.saveState())
            except Exception:
                pass
        except Exception:
            pass

        # attempt to cancel running worker(s)
        try:
            if hasattr(self, 'active_lgb_worker') and self.active_lgb_worker is not None:
                w = self.active_lgb_worker
                if hasattr(w, 'cancel'):
                    w.cancel()
            # SimulationWorker may be running via threadpool without stored ref; best-effort only
        except Exception:
            pass

        return super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # 确保 Qt WebEngine 已初始化
    QtWebEngineImported = True
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
