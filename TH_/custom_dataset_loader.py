import pickle
import numpy as np
import pandas as pd

# pickle 파일 로드
print("Loading datasets_4000.pkl...")
with open('datasets_4000.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n데이터 타입: {type(data)}")

if isinstance(data, dict):
    print(f"키 목록: {list(data.keys())}")
    
    # 딕셔너리를 DataFrame으로 변환
    df_data = {}
    for key, value in data.items():
        print(f"\n키: {key}")
        print(f"  타입: {type(value)}")
        
        if isinstance(value, np.ndarray):
            print(f"  형태: {value.shape}")
            print(f"  데이터 타입: {value.dtype}")
            print(f"  최소값: {value.min():.4f}")
            print(f"  최대값: {value.max():.4f}")
            print(f"  평균값: {value.mean():.4f}")
            
            # 배열 정보를 딕셔너리에 저장
            df_data[f"{key}_shape"] = [str(value.shape)]
            df_data[f"{key}_dtype"] = [value.dtype]
            df_data[f"{key}_min"] = [value.min()]
            df_data[f"{key}_max"] = [value.max()]
            df_data[f"{key}_mean"] = [value.mean()]
            
        elif isinstance(value, list):
            print(f"  길이: {len(value)}")
            if len(value) > 0:
                print(f"  첫 번째 요소 타입: {type(value[0])}")
                if isinstance(value[0], np.ndarray):
                    print(f"  첫 번째 요소 형태: {value[0].shape}")
                    
            df_data[f"{key}_length"] = [len(value)]
            df_data[f"{key}_type"] = [type(value[0]).__name__ if len(value) > 0 else "empty"]
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(df_data)
    df.to_csv('dataset_analysis.csv', index=False)
    print(f"\n분석 결과가 'dataset_analysis.csv'에 저장되었습니다.")

elif isinstance(data, np.ndarray):
    print(f"배열 형태: {data.shape}")
    print(f"데이터 타입: {data.dtype}")
    print(f"최소값: {data.min():.4f}")
    print(f"최대값: {data.max():.4f}")
    print(f"평균값: {data.mean():.4f}")
    
    # 배열 정보를 DataFrame으로 저장
    df = pd.DataFrame({
        'shape': [str(data.shape)],
        'dtype': [data.dtype],
        'min': [data.min()],
        'max': [data.max()],
        'mean': [data.mean()]
    })
    df.to_csv('dataset_analysis.csv', index=False)
    print(f"\n분석 결과가 'dataset_analysis.csv'에 저장되었습니다.")

elif isinstance(data, list):
    print(f"리스트 길이: {len(data)}")
    if len(data) > 0:
        print(f"첫 번째 요소 타입: {type(data[0])}")
        if isinstance(data[0], np.ndarray):
            print(f"첫 번째 요소 형태: {data[0].shape}")
    
    # 리스트 정보를 DataFrame으로 저장
    df = pd.DataFrame({
        'length': [len(data)],
        'first_element_type': [type(data[0]).__name__ if len(data) > 0 else "empty"]
    })
    df.to_csv('dataset_analysis.csv', index=False)
    print(f"\n분석 결과가 'dataset_analysis.csv'에 저장되었습니다.")

print("\n데이터 분석 완료!")
