import cv2
# [QUAN TRỌNG] Ngăn OpenCV tạo luồng con gây xung đột với PyTorch
cv2.setNumThreads(0)

import numpy as np
import os
import time
from ultralytics import YOLO
from config import CameraConfig
from visualizer import Visualizer
from processor import FrameProcessor

# --- CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG) ---
# 1. Điền đường dẫn Model
MODEL_ITEM_PATH = r"D:\AI_CK\final\minh\models\best_ck.pt"
MODEL_SLOT_PATH = r"D:\AI_CK\final\minh\models\best.pt"

# 2. Điền đường dẫn 4 Video tương ứng cho 4 Cam
# (Nếu bạn chỉ có 1 video, có thể điền giống nhau cho cả 4 dòng để test tải hệ thống)
VIDEO_PATHS = {
    "cam_1": r"D:\AI_CK\final\minh\test_data\video_test\Video test đúng.avi",
    "cam_2": r"D:\AI_CK\final\minh\test_data\video_test\Video test đúng.avi", 
    "cam_3": r"D:\AI_CK\final\minh\test_data\video_test\Video test đúng.avi",
    "cam_4": r"D:\AI_CK\final\minh\test_data\video_test\Video test đúng.avi"
}

PROC_W, PROC_H = 640, 480 
DASHBOARD_WIDTH = 350 

# --- CLASS ĐỌC VIDEO (CÓ LẶP LẠI) ---
class VideoLooper:
    def __init__(self, video_path, cam_name):
        self.path = video_path
        self.cam_name = cam_name
        if not os.path.exists(video_path):
            print(f"❌ Không tìm thấy file video cho {cam_name}: {video_path}")
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(video_path)
            
    def read(self):
        if self.cap is None or not self.cap.isOpened():
            # Trả về ảnh đen nếu video lỗi
            return np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
            
        ret, frame = self.cap.read()
        if not ret:
            # Hết video -> Quay lại từ đầu (Loop)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame

    def release(self):
        if self.cap: self.cap.release()

# --- CLASS QUẢN LÝ QUY TRÌNH (FULL LOGIC) ---
class SystemFlowManager:
    def __init__(self):
        self.timer_start = None
        self.final_verdict = None 
        self.state = "IDLE" 
        # IDLE -> RUNNING -> COUNTDOWN -> SHOW_RESULT

    def update(self, configs, cam4_detected):
        # 1. NẾU CAM 4 THẤY KHAY -> ĐANG LÀM VIỆC
        if cam4_detected:
            self.state = "RUNNING"
            self.timer_start = None
            self.final_verdict = None
            return None

        # 2. NẾU CAM 4 KHÔNG THẤY KHAY (Mất tín hiệu)
        else:
            if self.state == "RUNNING":
                self.state = "COUNTDOWN"
                self.timer_start = time.time()
                print("🏁 Cam 4 mất tín hiệu -> Bắt đầu đếm ngược 10s...")
            
            elif self.state == "IDLE": return None

            # --- XỬ LÝ ĐẾM NGƯỢC ---
            if self.state == "COUNTDOWN":
                elapsed = time.time() - self.timer_start
                remaining = 10.0 - elapsed 
                
                if remaining <= 0:
                    self.state = "SHOW_RESULT"
                    
                    # Quét toàn bộ checklist của 4 Cam
                    checklist_ok = True
                    for cfg in configs:
                        stats = cfg.get_item_counts()
                        for item_info in stats.values():
                            if not item_info['done']: 
                                checklist_ok = False
                    
                    self.final_verdict = "PASS" if checklist_ok else "FAIL"
                    self.timer_start = time.time() # Reset timer để show result
                    return "FINISHED"
                return remaining

            # --- XỬ LÝ HIỂN THỊ KẾT QUẢ ---
            elif self.state == "SHOW_RESULT":
                elapsed = time.time() - self.timer_start
                if elapsed > 5.0: # Show 5s
                    print("🔄 Reset Hệ Thống")
                    return "RESET_NOW"
                return "SHOWING"

        return None

def main():
    if not os.path.exists(MODEL_ITEM_PATH): 
        print("❌ Lỗi: Không tìm thấy model!")
        return

    print(f"🚀 TEST SIMULATION: 4 CAMERAS")

    # Load AI
    print("⏳ Đang load model...")
    model_items = YOLO(MODEL_ITEM_PATH)
    model_slots = YOLO(MODEL_SLOT_PATH)

    cam_names = ["cam_1", "cam_2", "cam_3", "cam_4"]

    # Khởi tạo 4 luồng Video
    streams = []
    for name in cam_names:
        streams.append(VideoLooper(VIDEO_PATHS[name], name))

    # Khởi tạo Logic
    configs = [CameraConfig(name) for name in cam_names]
    processors = [FrameProcessor(cfg) for cfg in configs]
    visualizer = Visualizer()
    flow_manager = SystemFlowManager()

    # Canvas Setup
    total_w = (PROC_W * 2) + DASHBOARD_WIDTH
    total_h = PROC_H * 2
    main_canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    try:
        while True:
            # 1. Đọc và Resize 4 Frame
            batch_frames = []
            
            for i, stream in enumerate(streams):
                frame = stream.read()
                # Resize ngay lập tức để đồng bộ kích thước
                resized = cv2.resize(frame, (PROC_W, PROC_H))
                batch_frames.append(resized)

            # 2. AI Predict (Batch 4 ảnh cùng lúc)
            # stream=False vì đây là list ảnh rời rạc trong vòng lặp thủ công
            res_slots = model_slots.predict(batch_frames, conf=0.5, verbose=False)
            res_items = model_items.predict(batch_frames, conf=0.45, verbose=False)

            # 3. Process Logic & Drawing
            cam4_detected = False

            for i in range(4):
                # Tính toán vị trí vẽ trên canvas lớn
                dx, dy = (i % 2) * PROC_W, (i // 2) * PROC_H
                roi = main_canvas[dy:dy+PROC_H, dx:dx+PROC_W]
                
                # Copy ảnh webcam vào vùng ROI
                np.copyto(roi, batch_frames[i])

                # Xử lý Logic
                detected = processors[i].process(res_slots[i], res_items[i])

                # Lưu trạng thái Cam 4 để điều phối quy trình
                if i == 3: cam4_detected = detected

                # --- VẼ VISUALIZATION ---
                # Vẽ Slot
                for slot in configs[i].slots.values():
                    visualizer.draw_slot_obb(roi, slot)
                
                # Vẽ Item
                if res_items[i].boxes:
                    for b, c, cl in zip(res_items[i].boxes.xyxy.cpu().numpy(), res_items[i].boxes.conf.cpu().numpy(), res_items[i].boxes.cls.cpu().numpy()):
                        visualizer.draw_item_box(roi, b, res_items[i].names[int(cl)], c)
                
                # Vẽ Info bar
                visualizer.draw_camera_info(roi, configs[i])

            # 4. LOGIC QUẢN LÝ LUỒNG (System Flow)
            status = flow_manager.update(configs, cam4_detected)

            # Xử lý lệnh Reset
            if status == "RESET_NOW":
                for cfg in configs: cfg.force_reset()
                flow_manager.state = "IDLE"

            # --- GIAO DIỆN TỔNG ---
            blink = int(time.time() * 4) % 2 == 0

            # A. Đếm ngược (Vẽ lên góc Cam 4)
            if flow_manager.state == "COUNTDOWN" and isinstance(status, float):
                # Tọa độ Cam 4 là (PROC_W, PROC_H)
                start_x, start_y = PROC_W, PROC_H
                cv2.putText(main_canvas, f"CHECK: {status:.1f}s", (start_x + 50, start_y + 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

            # B. Hiển thị Kết quả PASS/FAIL
            elif flow_manager.state == "SHOW_RESULT":
                if flow_manager.final_verdict == "PASS" and blink:
                    cv2.putText(main_canvas, "OKE - DONE", (total_w//2 - 200, total_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)
                    cv2.rectangle(main_canvas, (0,0), (total_w, total_h), (0,255,0), 20)
                
                elif flow_manager.final_verdict == "FAIL" and blink:
                    cv2.putText(main_canvas, "WRONG / MISSING", (total_w//2 - 350, total_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
                    cv2.rectangle(main_canvas, (0,0), (total_w, total_h), (0,0,255), 20)

            # 5. Dashboard bên phải
            dashboard_roi = main_canvas[:, -DASHBOARD_WIDTH:]
            dashboard_roi[:] = (20, 20, 20) # Reset nền đen
            
            # Nhấp nháy đỏ Dashboard nếu Fail
            if flow_manager.state == "SHOW_RESULT" and flow_manager.final_verdict == "FAIL" and blink:
                dashboard_roi[:] = (0, 0, 100)

            visualizer.draw_dashboard_on_roi(dashboard_roi, configs)
            visualizer.draw_fps(main_canvas)

            # Show
            cv2.imshow("Full System Simulation (4 Cams)", main_canvas)
            
            # Điều khiển
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord('p'): # Pause
                cv2.waitKey(-1)

    finally:
        for s in streams: s.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()