import numpy as np
import pandas as pd
import os
import glob

# 출력 디렉토리 생성
output_dir = "Custom_Dataset_HJW_Parse"
os.makedirs(output_dir, exist_ok=True)

# 선택할 차원들 (0-based indexing)
state_selected_dims = [0, 1, 4, 6]  # 1, 2, 5, 7번째 (0부터 시작하므로 -1)
state_ddot_selected_dims = [0, 1, 4]  # 1, 2, 5번째 (0부터 시작하므로 -1)

print("데이터 추출 시작...")
print(f"State 선택 차원: {[i+1 for i in state_selected_dims]} (0-based: {state_selected_dims})")
print(f"State_ddot 선택 차원: {[i+1 for i in state_ddot_selected_dims]} (0-based: {state_ddot_selected_dims})")

# 데이터 폴더들 찾기
data_folders = sorted(glob.glob("Custom_Dataset_HJW/data_*"))
print(f"총 {len(data_folders)}개 데이터 폴더 발견")

# 각 데이터셋 처리
for folder_path in data_folders:
    dataset_id = folder_path.split('_')[-1]  # data_0 -> 0
    print(f"처리 중: {dataset_id}")
    
    try:
        # 데이터 로드
        state = np.load(os.path.join(folder_path, "state.npy"))
        state_ddot = np.load(os.path.join(folder_path, "state_ddot.npy"))
        
        # oracle_response 레이블 로드
        oracle_file = os.path.join(folder_path, "oracle_response.txt")
        try:
            with open(oracle_file, 'r') as f:
                oracle_response = int(f.read().strip())
            print(f"  레이블 로드: {oracle_response}")
        except Exception as e:
            print(f"  레이블 로드 실패: {e}, 기본값 0 사용")
            oracle_response = 0
        
        # 선택된 차원들 추출
        state_selected = state[:, state_selected_dims]  # (5000, 4)
        state_ddot_selected = state_ddot[:, state_ddot_selected_dims]  # (5000, 3)
        
        # 컬럼명 생성
        state_cols = [f"state_{i+1}" for i in state_selected_dims]
        state_ddot_cols = [f"state_ddot_{i+1}" for i in state_ddot_selected_dims]
        
        # 시간 인덱스 생성
        time_steps = np.arange(5000)
        
        # DataFrame 생성
        df = pd.DataFrame()
        df['time_step'] = time_steps
        
        # State 데이터 추가
        for i, col in enumerate(state_cols):
            df[col] = state_selected[:, i]
        
        # State_ddot 데이터 추가
        for i, col in enumerate(state_ddot_cols):
            df[col] = state_ddot_selected[:, i]
        
        # 레이블 추가 (모든 행에 동일한 레이블)
        df['label'] = oracle_response
        
        # CSV 저장
        output_file = os.path.join(output_dir, f"dataset_{dataset_id}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"  저장됨: {output_file}")
        print(f"  형태: {df.shape}")
        print(f"  컬럼: {list(df.columns)}")
        print(f"  레이블: {oracle_response}")
        
    except Exception as e:
        print(f"  오류 발생: {e}")
        continue

print(f"\n추출 완료!")
print(f"총 {len(glob.glob(os.path.join(output_dir, '*.csv')))}개 CSV 파일 생성")
print(f"출력 디렉토리: {output_dir}")

# 샘플 데이터 확인
sample_file = os.path.join(output_dir, "dataset_0.csv")
if os.path.exists(sample_file):
    sample_df = pd.read_csv(sample_file)
    print(f"\n샘플 데이터 (dataset_0.csv):")
    print(f"형태: {sample_df.shape}")
    print(f"컬럼: {list(sample_df.columns)}")
    print(f"레이블 분포: {sample_df['label'].value_counts()}")
    print("\n처음 5행:")
    print(sample_df.head())
    
    print("\n기술 통계:")
    print(sample_df.describe())

# 전체 데이터셋 요약 생성
print("\n전체 데이터셋 요약 생성 중...")
summary_data = []

for csv_file in sorted(glob.glob(os.path.join(output_dir, "*.csv"))):
    dataset_id = csv_file.split('_')[-1].replace('.csv', '')
    df = pd.read_csv(csv_file)
    
    summary = {
        'dataset_id': dataset_id,
        'rows': len(df),
        'columns': len(df.columns),
        'label': df['label'].iloc[0],  # 모든 행이 동일한 레이블
        'state_1_mean': df['state_1'].mean(),
        'state_1_std': df['state_1'].std(),
        'state_2_mean': df['state_2'].mean(),
        'state_2_std': df['state_2'].std(),
        'state_5_mean': df['state_5'].mean(),
        'state_5_std': df['state_5'].std(),
        'state_7_mean': df['state_7'].mean(),
        'state_7_std': df['state_7'].std(),
        'state_ddot_1_mean': df['state_ddot_1'].mean(),
        'state_ddot_1_std': df['state_ddot_1'].std(),
        'state_ddot_2_mean': df['state_ddot_2'].mean(),
        'state_ddot_2_std': df['state_ddot_2'].std(),
        'state_ddot_5_mean': df['state_ddot_5'].mean(),
        'state_ddot_5_std': df['state_ddot_5'].std(),
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)
print(f"요약 파일 저장: {output_dir}/dataset_summary.csv")

# 레이블 분포 확인
print("\n전체 데이터셋 레이블 분포:")
label_counts = summary_df['label'].value_counts().sort_index()
print(label_counts)
print(f"총 데이터셋 수: {len(summary_df)}")

print("\n모든 작업 완료!") 