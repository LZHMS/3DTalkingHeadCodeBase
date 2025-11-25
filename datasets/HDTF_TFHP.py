import os
import pickle
import lmdb
import io
import torchaudio
import torch
import numpy as np
from base import Datum, DatasetBase, DATASET_REGISTRY, DataManager, DatasetWrapper
import logging
logger: logging.Logger

@DATASET_REGISTRY.register()
class HDTF_TFHP(DatasetBase):
    """
    Parameter:
        Pose: N = 6, 3DMM pose parameters (rotation in axis-angle) including global pose and jaw pose
        Shape: N = 100, 3DMM shape parameters
        Exp: N = 50, 3DMM expression parameters
    """

    def __init__(self, cfg):
        # data config and path
        root = os.path.abspath(os.path.expanduser(cfg.ROOT))
        self.dataset_dir = os.path.join(root, cfg.NAME)
        lmdb_path = self.dataset_dir
        split_path = [os.path.join(self.dataset_dir, cfg.HDTF_TFHP.TRAIN),
                           os.path.join(self.dataset_dir, cfg.HDTF_TFHP.VAL),
                           os.path.join(self.dataset_dir, cfg.HDTF_TFHP.TEST)]
        coef_stats_path = os.path.join(self.dataset_dir, cfg.HDTF_TFHP.COEF_STATS)
        if coef_stats_path is not None:
            coef_stats = dict(np.load(coef_stats_path))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            logger.warning('Warning: No stats file found. Coef will not be normalized.')
        
        # calculate the number of audio samples per frame
        self.audio_unit = cfg.HDTF_TFHP.AUDIO_SR / cfg.HDTF_TFHP.COEF_FPS

        # total number of motions and audio samples
        self.n_motions = cfg.HDTF_TFHP.MOTIONS
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)

        # Load lmdb env and get the clip len
        lmdb_env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        with lmdb_env.begin(write=False) as txn:
            self.clip_len = pickle.loads(txn.get('metadata'.encode()))['seg_len']
            self.audio_clip_len = round(self.audio_unit * self.clip_len)

        # Read split file
        subjects_dict = {"train": [], "val": [], "test": []}
        for split, fpath in zip(subjects_dict, split_path):
            with open(fpath) as f:
                for line in f:
                    subjects_dict[split].append(line.strip())

        data_dict = {"train": [], "val": [], "test": []}
        for split in ["train", "val", "test"]:
            for subject in subjects_dict[split]:
                # Read audio and coef
                with lmdb_env.begin(write=False) as txn:
                    meta_key = f'{subject}/metadata'.encode()
                    metadata = pickle.loads(txn.get(meta_key))
                    seq_len = metadata['n_frames']

                # Crop the audio and coef
                if cfg.HDTF_TFHP.CROP == 'random':
                    start_frame = np.random.randint(0, seq_len - self.coef_total_len + 1)
                elif cfg.HDTF_TFHP.CROP == 'begin':
                    start_frame = 0
                elif cfg.HDTF_TFHP.CROP == 'end':
                    start_frame = seq_len - self.coef_total_len
                else:
                    raise ValueError(f'Unknown crop strategy: {cfg.HDTF_TFHP.CROP}')
                
                coef_dict = {'shape': [], 'exp': [], 'pose': []}
                audio = []
                start_clip = start_frame // self.clip_len
                end_clip = (start_frame + self.coef_total_len - 1) // self.clip_len + 1
                with lmdb_env.begin(write=False) as txn:
                    for clip_idx in range(start_clip, end_clip):
                        key = f'{subject}/{clip_idx:03d}'.encode()
                        start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                        end_idx = min(start_frame + self.coef_total_len - clip_idx * self.clip_len, self.clip_len)

                        # load the coefficients
                        entry = pickle.loads(txn.get(key))
                        for coef_key in ['shape', 'exp', 'pose']:
                            coef_dict[coef_key].append(entry['coef'][coef_key][start_idx:end_idx])

                        audio_data = entry['audio']
                        audio_clip, audio_sr = torchaudio.load(io.BytesIO(audio_data))
                        assert audio_sr == cfg.HDTF_TFHP.AUDIO_SR, f'Invalid sampling rate: {audio_sr}'
                        audio_clip = audio_clip.squeeze()
                        audio.append(audio_clip[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)])

                coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in ['shape', 'exp', 'pose']}
                assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'
                audio = torch.cat(audio, dim=0)
                assert audio.shape[0] == self.coef_total_len * self.audio_unit, f'Invalid audio length: {audio.shape[0]}'
                audio_mean, audio_std = audio.mean(), audio.std()
                audio = (audio - audio_mean) / (audio_std + 1e-5)

                keys = ['shape', 'exp', 'pose']
                # normalize coef if applicable
                if self.coef_stats is not None:
                    coef_dict = {k: (coef_dict[k] - self.coef_stats[f'{k}_mean']) / (self.coef_stats[f'{k}_std'] + 1e-9)
                                for k in keys}
                # Extract two consecutive audio/coef clips
                audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
                coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},
                            {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]

                data_dict[split].append(Datum(name=subject, audio=audio_pair, coefficients=coef_pair))

        super().__init__(train=data_dict['train'], val=data_dict['val'], test=data_dict['test'])


class HDTF_TFHPDM(DataManager):
  
    def __init__(self,
                cfg,
                dataset_wrapper=None,
                infinite_train=False):
        super().__init__(cfg, dataset_wrapper, infinite_train)

    def data_analysis(self):
        """
        分析HDTF_TFHP数据集中exp和pose参数的统计特性和分布差异
        用于后续流匹配模型的建模
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        import seaborn as sns
        
        logger.info("Starting data analysis for exp and pose parameters...")
        
        # 收集所有数据
        all_exp = []
        all_pose = []
        all_exp_diff = []  # 帧间差分
        all_pose_diff = []
        
        # 遍历训练集收集数据
        for split_name, split_data in [('train', self.dataset.train), 
                                        ('val', self.dataset.val), 
                                        ('test', self.dataset.test)]:
            logger.info(f"Processing {split_name} split with {len(split_data)} samples...")
            
            for item in split_data:
                for clip_id in range(2):  # 每个样本有两个clip
                    exp = item.coefficients[clip_id]['exp'].numpy()
                    pose = item.coefficients[clip_id]['pose'][:, :-2].numpy()
                    
                    all_exp.append(exp)
                    all_pose.append(pose)
                    
                    # 计算帧间差分（temporal dynamics）
                    if exp.shape[0] > 1:
                        all_exp_diff.append(np.diff(exp, axis=0))
                        all_pose_diff.append(np.diff(pose, axis=0))
        
        # 合并所有数据
        all_exp = np.concatenate(all_exp, axis=0)  # (total_frames, exp_dim)
        all_pose = np.concatenate(all_pose, axis=0)  # (total_frames, pose_dim)
        all_exp_diff = np.concatenate(all_exp_diff, axis=0)
        all_pose_diff = np.concatenate(all_pose_diff, axis=0)
        
        logger.info(f"Total frames collected: {all_exp.shape[0]}")
        logger.info(f"Exp dimension: {all_exp.shape[1]}, Pose dimension: {all_pose.shape[1]}")
        
        # 诊断数据质量
        logger.info("\n" + "="*80)
        logger.info("DATA QUALITY DIAGNOSIS")
        logger.info("="*80)
        logger.info(f"EXP - Contains NaN: {np.isnan(all_exp).any()}, Contains Inf: {np.isinf(all_exp).any()}")
        logger.info(f"POSE - Contains NaN: {np.isnan(all_pose).any()}, Contains Inf: {np.isinf(all_pose).any()}")
        logger.info(f"EXP - Unique values per dim: min={min([len(np.unique(all_exp[:, i])) for i in range(all_exp.shape[1])])}, max={max([len(np.unique(all_exp[:, i])) for i in range(all_exp.shape[1])])}")
        logger.info(f"POSE - Unique values per dim: min={min([len(np.unique(all_pose[:, i])) for i in range(all_pose.shape[1])])}, max={max([len(np.unique(all_pose[:, i])) for i in range(all_pose.shape[1])])}")
        
        # 检查常量维度
        exp_constant_dims = [i for i in range(all_exp.shape[1]) if np.std(all_exp[:, i]) < 1e-10]
        pose_constant_dims = [i for i in range(all_pose.shape[1]) if np.std(all_pose[:, i]) < 1e-10]
        
        if exp_constant_dims:
            logger.warning(f"EXP has {len(exp_constant_dims)} constant dimensions: {exp_constant_dims}")
        if pose_constant_dims:
            logger.warning(f"POSE has {len(pose_constant_dims)} constant dimensions: {pose_constant_dims}")
        
        # 1. 基础统计分析
        logger.info("\n" + "="*80)
        logger.info("1. BASIC STATISTICS ANALYSIS")
        logger.info("="*80)
        
        exp_mean = np.mean(all_exp, axis=0)
        exp_std = np.std(all_exp, axis=0)
        exp_min = np.min(all_exp, axis=0)
        exp_max = np.max(all_exp, axis=0)
        
        pose_mean = np.mean(all_pose, axis=0)
        pose_std = np.std(all_pose, axis=0)
        pose_min = np.min(all_pose, axis=0)
        pose_max = np.max(all_pose, axis=0)
        
        logger.info(f"\nEXP Statistics:")
        logger.info(f"  Mean range: [{exp_mean.min():.4f}, {exp_mean.max():.4f}]")
        logger.info(f"  Std range: [{exp_std.min():.4f}, {exp_std.max():.4f}]")
        logger.info(f"  Global range: [{exp_min.min():.4f}, {exp_max.max():.4f}]")
        logger.info(f"  Avg std across dims: {exp_std.mean():.4f}")
        
        logger.info(f"\nPOSE Statistics:")
        logger.info(f"  Mean range: [{pose_mean.min():.4f}, {pose_mean.max():.4f}]")
        logger.info(f"  Std range: [{pose_std.min():.4f}, {pose_std.max():.4f}]")
        logger.info(f"  Global range: [{pose_min.min():.4f}, {pose_max.max():.4f}]")
        logger.info(f"  Avg std across dims: {pose_std.mean():.4f}")
        
        # 2. 波动性分析（Temporal Dynamics）
        logger.info("\n" + "="*80)
        logger.info("2. TEMPORAL DYNAMICS ANALYSIS")
        logger.info("="*80)
        
        exp_diff_mean = np.mean(np.abs(all_exp_diff), axis=0)
        exp_diff_std = np.std(all_exp_diff, axis=0)
        pose_diff_mean = np.mean(np.abs(all_pose_diff), axis=0)
        pose_diff_std = np.std(all_pose_diff, axis=0)
        
        logger.info(f"\nEXP Frame-to-frame Changes:")
        logger.info(f"  Mean absolute change: {exp_diff_mean.mean():.6f}")
        logger.info(f"  Std of changes: {exp_diff_std.mean():.6f}")
        logger.info(f"  Max dimension change: {exp_diff_mean.max():.6f}")
        
        logger.info(f"\nPOSE Frame-to-frame Changes:")
        logger.info(f"  Mean absolute change: {pose_diff_mean.mean():.6f}")
        logger.info(f"  Std of changes: {pose_diff_std.mean():.6f}")
        logger.info(f"  Max dimension change: {pose_diff_mean.max():.6f}")
        
        logger.info(f"\nTemporal Volatility Ratio (Pose/Exp): {pose_diff_mean.mean() / exp_diff_mean.mean():.4f}")
        
        # 3. 分布形状分析
        logger.info("\n" + "="*80)
        logger.info("3. DISTRIBUTION SHAPE ANALYSIS")
        logger.info("="*80)
        
        # 对每个维度计算偏度和峰度（排除常量维度）
        exp_skewness = stats.skew(all_exp, axis=0, nan_policy='propagate')
        exp_kurtosis = stats.kurtosis(all_exp, axis=0, nan_policy='propagate')
        pose_skewness = stats.skew(all_pose, axis=0, nan_policy='propagate')
        pose_kurtosis = stats.kurtosis(all_pose, axis=0, nan_policy='propagate')
        
        # 计算有效值（非NaN）
        exp_skew_valid = exp_skewness[~np.isnan(exp_skewness)]
        exp_kurt_valid = exp_kurtosis[~np.isnan(exp_kurtosis)]
        pose_skew_valid = pose_skewness[~np.isnan(pose_skewness)]
        pose_kurt_valid = pose_kurtosis[~np.isnan(pose_kurtosis)]
        
        logger.info(f"\nEXP Distribution Shape:")
        if len(exp_skew_valid) > 0:
            logger.info(f"  Avg Skewness: {exp_skew_valid.mean():.4f} (range: [{exp_skew_valid.min():.4f}, {exp_skew_valid.max():.4f}])")
            logger.info(f"  Valid dimensions: {len(exp_skew_valid)}/{len(exp_skewness)}")
        else:
            logger.warning(f"  Skewness: All NaN (likely constant values)")
        
        if len(exp_kurt_valid) > 0:
            logger.info(f"  Avg Kurtosis: {exp_kurt_valid.mean():.4f} (range: [{exp_kurt_valid.min():.4f}, {exp_kurt_valid.max():.4f}])")
        else:
            logger.warning(f"  Kurtosis: All NaN (likely constant values)")
        
        logger.info(f"\nPOSE Distribution Shape:")
        if len(pose_skew_valid) > 0:
            logger.info(f"  Avg Skewness: {pose_skew_valid.mean():.4f} (range: [{pose_skew_valid.min():.4f}, {pose_skew_valid.max():.4f}])")
            logger.info(f"  Valid dimensions: {len(pose_skew_valid)}/{len(pose_skewness)}")
        else:
            logger.warning(f"  Skewness: All NaN (likely constant values)")
        
        if len(pose_kurt_valid) > 0:
            logger.info(f"  Avg Kurtosis: {pose_kurt_valid.mean():.4f} (range: [{pose_kurt_valid.min():.4f}, {pose_kurt_valid.max():.4f}])")
        else:
            logger.warning(f"  Kurtosis: All NaN (likely constant values)")
        
        # 4. 维度间相关性分析
        logger.info("\n" + "="*80)
        logger.info("4. CORRELATION ANALYSIS")
        logger.info("="*80)
        
        # 计算exp和pose维度内的相关性
        exp_corr = np.corrcoef(all_exp.T)
        pose_corr = np.corrcoef(all_pose.T)
        
        # 计算平均相关性（排除对角线）
        exp_corr_mean = (np.sum(np.abs(exp_corr)) - np.trace(np.abs(exp_corr))) / (exp_corr.shape[0] * (exp_corr.shape[0] - 1))
        pose_corr_mean = (np.sum(np.abs(pose_corr)) - np.trace(np.abs(pose_corr))) / (pose_corr.shape[0] * (pose_corr.shape[0] - 1))
        
        logger.info(f"\nIntra-dimension Correlation:")
        logger.info(f"  EXP avg abs correlation: {exp_corr_mean:.4f}")
        logger.info(f"  POSE avg abs correlation: {pose_corr_mean:.4f}")
        
        # 5. 数据分布差异总结
        logger.info("\n" + "="*80)
        logger.info("5. KEY DIFFERENCES SUMMARY FOR FLOW MATCHING")
        logger.info("="*80)
        
        variance_ratio = pose_std.mean() / exp_std.mean()
        temporal_ratio = pose_diff_mean.mean() / exp_diff_mean.mean()
        
        logger.info(f"\n关键发现：")
        logger.info(f"  1. 方差比 (Pose/Exp): {variance_ratio:.4f}")
        logger.info(f"     -> Pose参数的空间变化范围是Exp的 {variance_ratio:.2f} 倍")
        
        logger.info(f"\n  2. 时序波动比 (Pose/Exp): {temporal_ratio:.4f}")
        logger.info(f"     -> Pose参数的时序变化速度是Exp的 {temporal_ratio:.2f} 倍")
        
        logger.info(f"\n  3. 相关性差异:")
        logger.info(f"     -> Exp维度间相关性: {exp_corr_mean:.4f}")
        logger.info(f"     -> Pose维度间相关性: {pose_corr_mean:.4f}")
        
        logger.info(f"\n  4. 分布形状:")
        if len(exp_kurt_valid) > 0 and len(pose_kurt_valid) > 0:
            logger.info(f"     -> Exp更接近高斯分布 (kurtosis: {exp_kurt_valid.mean():.2f})")
            logger.info(f"     -> Pose偏离高斯分布更多 (kurtosis: {pose_kurt_valid.mean():.2f})")
        else:
            logger.warning(f"     -> 无法计算分布形状 (存在常量维度)")
        
        # 6. 可视化分析
        logger.info("\n" + "="*80)
        logger.info("6. GENERATING VISUALIZATION")
        logger.info("="*80)
        
        output_dir = os.path.join(self.dataset.dataset_dir, 'analysis_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 6.1 标准差对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Std comparison
        axes[0, 0].bar(range(len(exp_std)), exp_std, alpha=0.7, label='EXP')
        axes[0, 0].set_title('EXP Standard Deviation per Dimension')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Std')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(range(len(pose_std)), pose_std, alpha=0.7, color='orange', label='POSE')
        axes[0, 1].set_title('POSE Standard Deviation per Dimension')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Temporal change comparison
        axes[1, 0].bar(range(len(exp_diff_mean)), exp_diff_mean, alpha=0.7, label='EXP')
        axes[1, 0].set_title('EXP Temporal Changes (Frame-to-Frame)')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Mean Absolute Change')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(range(len(pose_diff_mean)), pose_diff_mean, alpha=0.7, color='orange', label='POSE')
        axes[1, 1].set_title('POSE Temporal Changes (Frame-to-Frame)')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Mean Absolute Change')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exp_pose_statistics.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {os.path.join(output_dir, 'exp_pose_statistics.png')}")
        plt.close()
        
        # 6.2 分布直方图（选择几个代表性维度）
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        sample_dims = [0, all_exp.shape[1]//2, all_exp.shape[1]-1]
        for i, dim in enumerate(sample_dims):
            axes[0, i].hist(all_exp[:, dim], bins=50, alpha=0.7, density=True)
            axes[0, i].set_title(f'EXP Dim {dim} Distribution')
            axes[0, i].set_xlabel('Value')
            axes[0, i].set_ylabel('Density')
            axes[0, i].grid(True, alpha=0.3)
        
        sample_dims_pose = [0, all_pose.shape[1]//2, all_pose.shape[1]-1]
        for i, dim in enumerate(sample_dims_pose):
            axes[1, i].hist(all_pose[:, dim], bins=50, alpha=0.7, color='orange', density=True)
            axes[1, i].set_title(f'POSE Dim {dim} Distribution')
            axes[1, i].set_xlabel('Value')
            axes[1, i].set_ylabel('Density')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_histograms.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {os.path.join(output_dir, 'distribution_histograms.png')}")
        plt.close()
        
        # 6.3 相关性矩阵热图
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        sns.heatmap(exp_corr, cmap='coolwarm', center=0, ax=axes[0], 
                   cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        axes[0].set_title('EXP Inter-dimension Correlation Matrix')
        
        sns.heatmap(pose_corr, cmap='coolwarm', center=0, ax=axes[1], 
                   cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        axes[1].set_title('POSE Inter-dimension Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrices.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {os.path.join(output_dir, 'correlation_matrices.png')}")
        plt.close()
        
        # 6.4 时序变化可视化（计算所有样本的平均时序演变）
        logger.info("Computing average temporal evolution across all samples...")
        
        # 收集所有序列用于平均
        all_exp_sequences = []
        all_pose_sequences = []
        
        for split_name, split_data in [('train', self.dataset.train), 
                                        ('val', self.dataset.val), 
                                        ('test', self.dataset.test)]:
            for item in split_data:
                for clip_id in range(2):
                    exp_seq = item.coefficients[clip_id]['exp'].numpy()
                    pose_seq = item.coefficients[clip_id]['pose'][:, :-2].numpy()
                    all_exp_sequences.append(exp_seq)
                    all_pose_sequences.append(pose_seq)
        
        # 将所有序列堆叠并计算平均值和标准差
        # 假设所有序列长度相同
        all_exp_sequences = np.stack(all_exp_sequences, axis=0)  # (N_samples, T, D_exp)
        all_pose_sequences = np.stack(all_pose_sequences, axis=0)  # (N_samples, T, D_pose)
        
        exp_mean_trajectory = np.mean(all_exp_sequences, axis=0)  # (T, D_exp)
        exp_std_trajectory = np.std(all_exp_sequences, axis=0)    # (T, D_exp)
        pose_mean_trajectory = np.mean(all_pose_sequences, axis=0)  # (T, D_pose)
        pose_std_trajectory = np.std(all_pose_sequences, axis=0)    # (T, D_pose)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # EXP时序演变（前5个维度）
        n_exp_dims = min(5, exp_mean_trajectory.shape[1])
        for dim in range(n_exp_dims):
            mean_curve = exp_mean_trajectory[:, dim]
            std_curve = exp_std_trajectory[:, dim]
            frames = np.arange(len(mean_curve))
            
            axes[0].plot(frames, mean_curve, label=f'Dim {dim}', linewidth=2)
            # axes[0].fill_between(frames, 
            #                      mean_curve - std_curve, 
            #                      mean_curve + std_curve, 
            #                      alpha=0.2)
        
        axes[0].set_title('Average EXP Temporal Evolution (First 5 Dims) across All Samples')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Value (Mean ± Std)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # POSE时序演变（前4个维度）
        n_pose_dims = min(4, pose_mean_trajectory.shape[1])
        for dim in range(n_pose_dims):
            mean_curve = pose_mean_trajectory[:, dim]
            std_curve = pose_std_trajectory[:, dim]
            frames = np.arange(len(mean_curve))
            
            axes[1].plot(frames, mean_curve, label=f'Dim {dim}', linewidth=2)
            # axes[1].fill_between(frames, 
            #                      mean_curve - std_curve, 
            #                      mean_curve + std_curve, 
            #                      alpha=0.2)
        
        axes[1].set_title('Average POSE Temporal Evolution (First 4 Dims) across All Samples')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Value (Mean ± Std)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_evolution.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {os.path.join(output_dir, 'temporal_evolution.png')}")
        plt.close()
        
        # 重合程度，在px发生情况下 p||q 分布重合的期望值
        # 交集分布
        # 7. 保存统计数据
        analysis_stats = {
            'exp_mean': exp_mean,
            'exp_std': exp_std,
            'exp_min': exp_min,
            'exp_max': exp_max,
            'pose_mean': pose_mean,
            'pose_std': pose_std,
            'pose_min': pose_min,
            'pose_max': pose_max,
            'exp_diff_mean': exp_diff_mean,
            'exp_diff_std': exp_diff_std,
            'pose_diff_mean': pose_diff_mean,
            'pose_diff_std': pose_diff_std,
            'exp_skewness': exp_skewness,
            'exp_kurtosis': exp_kurtosis,
            'pose_skewness': pose_skewness,
            'pose_kurtosis': pose_kurtosis,
            'exp_corr': exp_corr,
            'pose_corr': pose_corr,
            'variance_ratio': variance_ratio,
            'temporal_ratio': temporal_ratio,
        }
        
        np.savez(os.path.join(output_dir, 'analysis_statistics.npz'), **analysis_stats)
        logger.info(f"Saved: {os.path.join(output_dir, 'analysis_statistics.npz')}")
        
        logger.info("\n" + "="*80)
        logger.info("DATA ANALYSIS COMPLETED!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*80 + "\n")
        
        return analysis_stats
        


class StyledTalkWrapper(DatasetWrapper):

    def __init__(self, cfg, data_source, is_train=False):
        super().__init__(cfg, data_source, is_train)
        self.rot_repr = cfg.MODEL.HEAD.ROT_REPR

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {"index": idx, "motion_coef": []}
        for clip_id in range(2):
            motion_coef = torch.cat([item.coefficients[clip_id][k] for k in ['exp', 'pose']], dim=-1)

            if self.rot_repr == 'aa':
                # Remove mouth rotation around y, z axis
                motion_coef = motion_coef[:, :-2]
            output["motion_coef"].append(motion_coef)
        output.update(item.to_dict())
        for k in ["audio", "coefficients"]:
            output.pop(k)
            
        return output

class HDTF_TFHPWrapper(DatasetWrapper):

    def __init__(self, cfg, data_source, is_train=False):
        super().__init__(cfg, data_source, is_train)
        self.rot_repr = cfg.MODEL.HEAD.ROT_REPR
        self.no_head_pose = cfg.MODEL.HEAD.NO_HEAD_POSE

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {"index": idx, "motion_coef": []}
        for clip_id in range(2):
            if self.rot_repr == 'aa':
                pose_coef = item.coefficients[clip_id]['pose'] if not self.no_head_pose else item.coefficients[clip_id]['pose'][..., -3:]
                # Remove mouth rotation round y, z axis
                pose_coef = pose_coef[..., :-2]
            else:
                raise ValueError(f'Unknown rotation representation {self.rot_repr}!')
            
            output["motion_coef"].append(torch.cat([item.coefficients[clip_id]['exp'], pose_coef], dim=-1))

        # Use the shape coefficients from the first frame of the first clip as the condition
        if item.coefficients[0]['shape'].ndim == 2:  # (N, 100)
            output["shape_coef"] = item.coefficients[0]['shape'].clone()
        else:  # (N, L, 100)
            output["shape_coef"] = item.coefficients[0]['shape'][:, 0].clone()

        output.update(item.to_dict())
        output.pop("coefficients")
            
        return output