import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
data_dir = "Custom_Dataset_HJW/data_0"

print("데이터 로딩 중...")
state = np.load(os.path.join(data_dir, "state.npy"))
state_ddot = np.load(os.path.join(data_dir, "state_ddot.npy"))
state_ddot = np.load(os.path.join(data_dir, "state_ddot.npy"))
t = np.load(os.path.join(data_dir, "t.npy"))

print(f"State shape: {state.shape}")
print(f"State_ddot shape: {state_ddot.shape}")
print(f"State_ddot shape: {state_ddot.shape}")
print(f"Time shape: {t.shape}")

# 시계열 플롯 생성
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('State Time Series Analysis - Data 0', fontsize=16, fontweight='bold')

# 1. State (10차원)
ax1 = axes[0]
for i in range(state.shape[1]):
    ax1.plot(t, state[:, i], label=f'State {i+1}', linewidth=1.5, alpha=0.8)
ax1.set_title('State Variables (10 dimensions)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('State Value')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(t.min(), t.max())

# 2. State_ddot (5차원) - 1차 미분
ax2 = axes[1]
for i in range(state_ddot.shape[1]):
    ax2.plot(t, state_ddot[:, i], label=f'State_ddot {i+1}', linewidth=1.5, alpha=0.8)
ax2.set_title('State 1st Derivative (5 dimensions)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('State 1st Derivative')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(t.min(), t.max())

# 3. State_ddot (5차원) - 2차 미분
ax3 = axes[2]
for i in range(state_ddot.shape[1]):
    ax3.plot(t, state_ddot[:, i], label=f'State_ddot {i+1}', linewidth=1.5, alpha=0.8)
ax3.set_title('State 2nd Derivative (5 dimensions)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('State 2nd Derivative')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(t.min(), t.max())

plt.tight_layout()
plt.savefig('state_timeseries_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 통계 정보 출력
print("\n=== State Statistics ===")
print(f"State range: [{state.min():.4f}, {state.max():.4f}]")
print(f"State mean: {state.mean():.4f}")
print(f"State std: {state.std():.4f}")

print(f"\nState_ddot range: [{state_ddot.min():.4f}, {state_ddot.max():.4f}]")
print(f"State_ddot mean: {state_ddot.mean():.4f}")
print(f"State_ddot std: {state_ddot.std():.4f}")

print(f"\nState_ddot range: [{state_ddot.min():.4f}, {state_ddot.max():.4f}]")
print(f"State_ddot mean: {state_ddot.mean():.4f}")
print(f"State_ddot std: {state_ddot.std():.4f}")

# 각 차원별 상세 분석
print("\n=== Dimension-wise Analysis ===")
for i in range(state.shape[1]):
    print(f"State {i+1}: mean={state[:, i].mean():.4f}, std={state[:, i].std():.4f}, range=[{state[:, i].min():.4f}, {state[:, i].max():.4f}]")

for i in range(state_ddot.shape[1]):
    print(f"State_ddot {i+1}: mean={state_ddot[:, i].mean():.4f}, std={state_ddot[:, i].std():.4f}, range=[{state_ddot[:, i].min():.4f}, {state_ddot[:, i].max():.4f}]")

for i in range(state_ddot.shape[1]):
    print(f"State_ddot {i+1}: mean={state_ddot[:, i].mean():.4f}, std={state_ddot[:, i].std():.4f}, range=[{state_ddot[:, i].min():.4f}, {state_ddot[:, i].max():.4f}]") 