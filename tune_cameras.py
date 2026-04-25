import genesis as gs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------
# 1. 상태 저장용 전역 딕셔너리
# ---------------------------------------------------------
state = {
    'active_cam': 'top', # 시작 시 Top 카메라 모드
    
    # Top Camera (Spherical)
    't_zoom': 0.6, 't_azi': 45.0, 't_elev': 45.0,
    't_fx': 0.25, 't_fy': 0.0, 't_fz': 0.0,
    
    # Wrist Camera (Local Offset)
    'w_x': 0.05, 'w_y': 0.0, 'w_z': 0.08,
    'w_roll': 0.0, 'w_pitch': 45.0, 'w_yaw': 0.0,
}

# ---------------------------------------------------------
# 2. 키보드 입력 처리 함수
# ---------------------------------------------------------
def process_key_input(key):
    step_pos = 0.005 
    step_deg = 2.0   
    
    if key == ord('m'):
        state['active_cam'] = 'top' if state['active_cam'] == 'wrist' else 'wrist'
        return

    if state['active_cam'] == 'wrist':
        if key == ord('w'): state['w_x'] += step_pos
        elif key == ord('s'): state['w_x'] -= step_pos
        elif key == ord('a'): state['w_y'] -= step_pos
        elif key == ord('d'): state['w_y'] += step_pos
        elif key == ord('q'): state['w_z'] -= step_pos
        elif key == ord('e'): state['w_z'] += step_pos
        elif key == ord('i'): state['w_pitch'] += step_deg
        elif key == ord('k'): state['w_pitch'] -= step_deg
        elif key == ord('j'): state['w_yaw'] += step_deg
        elif key == ord('l'): state['w_yaw'] -= step_deg
        elif key == ord('u'): state['w_roll'] -= step_deg
        elif key == ord('o'): state['w_roll'] += step_deg

    else: # Top Camera
        if key == ord('w'): state['t_zoom'] -= step_pos * 5  
        elif key == ord('s'): state['t_zoom'] += step_pos * 5 
        elif key == ord('a'): state['t_azi'] -= step_deg      
        elif key == ord('d'): state['t_azi'] += step_deg      
        elif key == ord('q'): state['t_elev'] -= step_deg     
        elif key == ord('e'): state['t_elev'] += step_deg     
        elif key == ord('i'): state['t_fx'] += step_pos
        elif key == ord('k'): state['t_fx'] -= step_pos
        elif key == ord('j'): state['t_fy'] += step_pos
        elif key == ord('l'): state['t_fy'] -= step_pos

# ---------------------------------------------------------
# 3. 화면 텍스트 오버레이 (HUD)
# ---------------------------------------------------------
def draw_hud(img, title):
    # 타이틀 배경
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    # 선택된 카메라면 노란색, 아니면 흰색
    color = (0, 255, 255) if title.lower() == state['active_cam'] else (255, 255, 255)
    
    if title == "GLOBAL OBSERVER":
        color = (200, 200, 200) # 관찰자 시점은 고정색상
        
    cv2.putText(img, f"[{title}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def get_camera_poses():
    r = state['t_zoom']
    azi = np.deg2rad(state['t_azi'])
    elev = np.deg2rad(max(1, state['t_elev'])) 
    
    top_look = [state['t_fx'], state['t_fy'], state['t_fz']]
    top_pos = [
        top_look[0] + r * np.cos(elev) * np.cos(azi),
        top_look[1] + r * np.cos(elev) * np.sin(azi),
        top_look[2] + r * np.sin(elev)
    ]
    
    wr_off = [state['w_x'], state['w_y'], state['w_z']]
    wr_rot = [state['w_roll'], state['w_pitch'], state['w_yaw']]
    
    return top_pos, top_look, wr_off, wr_rot

def main():
    urdf_path = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    
    gs.init(seed=42, precision="32", logging_level="warning")
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(ambient_light=(0.9, 0.9, 0.9)),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(dt=0.01),
    )

    plane = scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.96, 0.94, 0.87), roughness=0.3))
    robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True))
    cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.3, 0.0, 0.02)), surface=gs.surfaces.Default(color=(0.8, 0.1, 0.1)))

    # [추가됨] 가상의 마커 엔티티 (Top 카메라의 위치를 표시할 빨간색 떠다니는 구체)
    # kinematic=True로 설정하여 중력의 영향을 받지 않고 위치만 강제 이동시킬 수 있도록 함
    cam_marker = scene.add_entity(
        gs.morphs.Sphere(radius=0.025, pos=(0,0,0)), 
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
    )

    # 카메라 3대 셋업
    # 1. 고정된 3인칭 관찰자 카메라 (전체 뷰)
    cam_global = scene.add_camera(res=(480, 640), pos=(-0.5, -0.8, 0.8), lookat=(0.2, 0.0, 0.1), fov=60, GUI=False)
    # 2. 제어할 Top 카메라
    cam_top = scene.add_camera(res=(480, 640), fov=55, GUI=False)
    # 3. 제어할 Wrist 카메라
    cam_wrist = scene.add_camera(res=(480, 640), fov=65, GUI=False)
    
    scene.build()
    wrist_link = robot.get_link("wrist_link")

    cv2.namedWindow("Camera Tuning Pro", cv2.WINDOW_NORMAL)
    # 3개의 화면을 나란히 띄우기 위해 창을 더 넓게 설정
    cv2.resizeWindow("Camera Tuning Pro", 1800, 480) 

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key != 255:
            process_key_input(key)

        top_pos, top_look, wr_off, wr_rot = get_camera_poses()
        
        # 1. Top Camera 위치 갱신 및 마커 위치 동기화
        try: 
            cam_top.set_pose(pos=np.array(top_pos), lookat=np.array(top_look))
            # [추가됨] 마커의 위치를 Top 카메라의 글로벌 포지션으로 강제 갱신
            cam_marker.set_qpos(np.array(top_pos)) 
        except: pass
            
        # 2. Wrist Camera 갱신
        offset_T = np.eye(4)
        offset_T[:3, 3] = np.array(wr_off)
        offset_T[:3, :3] = R.from_euler('xyz', wr_rot, degrees=True).as_matrix()
        try: cam_wrist.attach(wrist_link, offset_T)
        except: pass

        scene.step()

        # 3. 렌더링 (Global, Top, Wrist 3개 동시)
        rgb_global = cam_global.render(rgb=True)
        rgb_top = cam_top.render(rgb=True)
        rgb_wrist = cam_wrist.render(rgb=True)
        
        rgb_global = rgb_global[0] if isinstance(rgb_global, tuple) else rgb_global
        rgb_top = rgb_top[0] if isinstance(rgb_top, tuple) else rgb_top
        rgb_wrist = rgb_wrist[0] if isinstance(rgb_wrist, tuple) else rgb_wrist
        
        if hasattr(rgb_top, 'cpu'):
            rgb_global = rgb_global.cpu().numpy()
            rgb_top = rgb_top.cpu().numpy()
            rgb_wrist = rgb_wrist.cpu().numpy()
            
        bgr_global = cv2.cvtColor((rgb_global * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr_top = cv2.cvtColor((rgb_top * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr_wrist = cv2.cvtColor((rgb_wrist * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 각 화면에 이름표 붙이기
        draw_hud(bgr_global, "GLOBAL OBSERVER")
        draw_hud(bgr_top, "TOP")
        draw_hud(bgr_wrist, "WRIST")
        
        # 3개의 이미지를 가로로 결합
        combined = np.hstack((bgr_global, bgr_top, bgr_wrist))
        cv2.imshow("Camera Tuning Pro", combined)

    print("\n✨ [최종 카메라 파라미터]")
    print(f"top_pos = {top_pos}\ntop_look = {top_look}")
    print(f"wr_offset_pos = {wr_off}\nwr_rot_euler = {wr_rot}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()