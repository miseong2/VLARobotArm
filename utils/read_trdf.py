import tensorflow as tf
import tensorflow_datasets as tfds

# json 파일과 tfrecord 파일이 모여있는 폴더 경로를 지정하세요.
# 예: dataset_path = './so101_dataset'
dataset_path = '/home/aivlab/kkb_capstone/datasets/tensorflow_datasets/pickup' 

# TFDS 빌더를 사용하여 디렉토리에서 데이터셋을 로드합니다.
builder = tfds.builder_from_directory(dataset_path)
dataset = builder.as_dataset(split='train')

# 첫 번째 에피소드만 가져옵니다.
for episode in dataset.take(1):
    print("========== 첫 번째 에피소드 정보 ==========")
    
    # 에피소드 내의 각 스텝(단계)을 순회합니다.
    # 첫 번째 스텝의 값만 확인하기 위해 반복문을 한 번만 돕니다.
    for i, step in enumerate(episode['steps']):
        print(f"\n[Step {i}]")
        
        # 1. 텍스트 지시어 (Language Instruction)
        # 바이트 형태이므로 utf-8로 디코딩합니다.
        instruction = step['language_instruction'].numpy().decode('utf-8')
        print(f"지시어(Language Instruction): '{instruction}'")
        
        # 2. 로봇의 상태 및 행동 (Action & State)
        print(f"행동(Action, 7차원): {step['action'].numpy()}")
        print(f"상태(State, 6차원): {step['observation']['state'].numpy()}")
        
        # 3. 보상 및 조건 (Reward, Discount, Flags)
        print(f"보상(Reward): {step['reward'].numpy()}")
        print(f"할인율(Discount): {step['discount'].numpy()}")
        print(f"첫 스텝 여부(is_first): {step['is_first'].numpy()}")
        print(f"마지막 스텝 여부(is_last): {step['is_last'].numpy()}")
        print(f"종료 여부(is_terminal): {step['is_terminal'].numpy()}")
        
        # 4. 이미지 데이터 (Observation Images)
        # 이미지 픽셀 값은 너무 방대하므로 배열의 형태(Shape)만 출력합니다.
        print(f"메인 카메라 이미지 형태(Image Primary): {step['observation']['image_primary'].shape}")
        print(f"손목 카메라 이미지 형태(Image Wrist): {step['observation']['image_wrist'].shape}")
        
        # 전체 스텝을 다 보려면 아래 break를 지우세요. 
        # (첫 번째 스텝만 확인하고 멈춥니다)
        break