# -*- coding: utf-8 -*-
"""
Custom Data Loader for TimeMIL
Loads CSV files from Custom_Dataset_HJW_Parse folder
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
import os

class CustomDatasetLoader(Dataset):
    def __init__(self, data_dir="Custom_Dataset_20_ds", split='train', seed=42, test_size=0.2, max_samples=500):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.test_size = test_size
        self.max_samples = max_samples  # 최대 사용할 샘플 수
        
        # 시계열 길이 설정
        self.seq_len = 250  # 더 짧게 설정 (250 이 기본)
        
        # 데이터 로드
        self._load_data()
        
        print(f"CustomDataset loaded - {split}: {len(self.FeatList)} samples")
        print(f"Feature dimensions: {self.feat_in}")
        print(f"Sequence length: {self.max_len}")
        print(f"Number of classes: {self.num_class}")
    
    def _load_data(self):
        """CSV 파일들에서 데이터 로드"""
        print(f"Loading data from {self.data_dir}...")

        # CSV 파일들 찾기
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "dataset_*.csv")))
        print(f"Found {len(csv_files)} CSV files")

        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        # 최대 500개만 사용
        if len(csv_files) > self.max_samples:
            csv_files = csv_files[:self.max_samples]
            print(f"Using only first {self.max_samples} files out of {len(csv_files)}")

        # 데이터와 레이블 로드
        X_list = []
        y_list = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # 마지막 column을 label로 사용
                label = int(df.iloc[0, -1])
                y_list.append(label)

                # label을 제외한 모든 열을 feature로 사용
                features = df.iloc[:, :-1].values
                X_list.append(features)

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

        # numpy 배열로 변환
        X = np.array(X_list)  # (N_samples, timesteps, features)
        y = np.array(y_list)

        print(f"Loaded data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label classes: {np.unique(y)}")

        # train/test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )

        if self.split == 'train':
            self.FeatList = X_train
            self.label = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes=2).float()
        elif self.split == 'valid':
            # validation용 분할이 별도로 없는 경우 test 사용
            self.FeatList = X_test
            self.label = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes=2).float()
        else:  # 'test'
            self.FeatList = X_test
            self.label = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes=2).float()

        # 속성 설정
        self.feat_in = self.FeatList[0].shape[1]  # 특성 수
        self.max_len = self.FeatList[0].shape[0]  # 시계열 길이
        self.num_class = self.label.shape[-1]     # 클래스 수

    
    def __getitem__(self, idx):
        """데이터 샘플 반환"""
        # (timesteps, features) 형태로 로드
        feats = torch.from_numpy(self.FeatList[idx]).float()  # (timesteps, features)
        
        # 시계열 길이 조정
        if feats.shape[0] < self.seq_len:
            # 패딩
            padding_size = self.seq_len - feats.shape[0]
            feats = F.pad(feats, pad=(0, 0, padding_size, 0))
        elif feats.shape[0] > self.seq_len:
            # 자르기
            feats = feats[:self.seq_len, :]
        
        # TimeMIL 모델이 (timesteps, features) 형태를 기대하므로 그대로 반환
        # 모델 내부에서 transpose 처리됨
        
        label = self.label[idx].float()
        
        return feats, label
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.label)
    
    def get_properties(self):
        """데이터셋 속성 반환"""
        return self.max_len, self.num_class, self.feat_in

# 사용 예시
if __name__ == "__main__":
    # 데이터 로더 테스트
    print("Testing CustomDatasetLoader...")
    
    # 훈련 데이터 로드
    train_dataset = CustomDatasetLoader(split='train')
    print(f"Train dataset size: {len(train_dataset)}")
    
    # 테스트 데이터 로드
    test_dataset = CustomDatasetLoader(split='test')
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 샘플 데이터 확인
    sample_feats, sample_label = train_dataset[0]
    print(f"Sample features shape: {sample_feats.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Sample features: {sample_feats[:, :5]}")  # 처음 5개 시점 