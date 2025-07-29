import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def downsample_dataset():
    """
    Custom_Dataset_HJW_Parse의 데이터를 20step마다 샘플링하여 
    Custom_Dataset_20_ds에 저장하고 timestep을 0,1,2,3...으로 재인덱싱
    """
    
    # 입력/출력 디렉토리 설정
    input_dir = "Custom_Dataset_HJW_Parse"
    output_dir = "Custom_Dataset_20_ds"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일들 찾기
    csv_files = sorted(glob.glob(f"{input_dir}/dataset_*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    # 각 파일 처리
    for csv_file in tqdm(csv_files, desc="Processing datasets"):
        try:
            # CSV 파일 읽기
            df = pd.read_csv(csv_file)
            
            # 20step마다 샘플링 (0, 20, 40, 60, ...)
            downsampled_df = df.iloc[::20].copy()
            
            # 첫 번째 컬럼을 time으로 대체 (0, 1, 2, 3...)
            downsampled_df.iloc[:, 0] = range(len(downsampled_df))
            
            # 파일명 추출
            filename = os.path.basename(csv_file)
            
            # 출력 파일 경로
            output_file = os.path.join(output_dir, filename)
            
            # CSV로 저장
            downsampled_df.to_csv(output_file, index=False)
            
            print(f"Processed {filename}: {len(df)} -> {len(downsampled_df)} samples")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"\n✅ Downsampling completed!")
    print(f"Original data: {input_dir}")
    print(f"Downsampled data: {output_dir}")
    print(f"Sampling rate: every 20th step")
    print(f"Timesteps reindexed: 0, 1, 2, 3, ...")

if __name__ == "__main__":
    downsample_dataset() 