# 확인해주세요
import pandas as pd

# 파일 하나만 정확히 읽기
df = pd.read_parquet("/home/aivlab/kkb_capstone/datasets/pickup/data/chunk-000/file-002.parquet")
print(f"에피소드 종류: {df['episode_index'].unique()}")
print(f"전체 프레임: {len(df)}")