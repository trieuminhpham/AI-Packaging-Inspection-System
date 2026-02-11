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

# --- CẤU HÌNH TEST ---
# 1. Điền đường dẫn file video của bạn vào đây
VIDEO_PATH = r"D:\AI_CK\final\minh\test_data\video_test\Video test đúng.avi" 

# 2. Điền đường dẫn Model
MODEL_ITEM_PATH = r"D:\AI_CK\final\minh\models\best_ck.pt"
MODEL_SLOT_PATH = r"D:\AI_CK\final\minh\models\best.pt"

# 3. Bạn muốn giả lập video này là Camera mấy? (cam_1, cam_2, cam_3, hoặc cam_4)
# Chọn 'cam_4' nếu muốn test logic đếm ngược và chốt kết quả.
TEST_CAM_NAME = "cam_1" 

PROC_W, PROC_H = 640, 480 
DASHBOARD_WIDTH = 350 

# --- CLASS ĐỌC VIDEO (CÓ LẶP LẠI) ---
class VideoLooper:
    def __init__(self, video_path):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            exit()
            
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            # Nếu hết video -> Quay lại từ đầu (Loop)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()

# --- CLASS QUẢN LÝ QUY TRÌNH (GIẢN LƯỢC CHO TEST) ---
class SystemFlowManagerTest:
    def __init__(self):
        self.timer_start = None
        self.final_verdict = None
        self.state = "IDLE" 

    def update(self, config, cam_detected):
        # Logic chỉ tập trung vào Camera đang test
        if cam_detected:
            self.state = "RUNNING"
            self.timer_start = None
            self.final_verdict = None
            return None
        else:
            if self.state == "RUNNING":
                self.state = "COUNTDOWN"
                self.timer_start = time.time()
                print("🏁 Mất tín hiệu khay -> Đếm ngược 5s (Test Mode)...")
            
            elif self.state == "IDLE": return None

            if self.state == "COUNTDOWN":
                elapsed = time.time() - self.timer_start
                remaining = 5.0 - elapsed # Test để 5s cho nhanh
                
                if remaining <= 0:
                    self.state = "SHOW_RESULT"
                    
                    # Chỉ check đúng camera đang test
                    stats = config.get_item_counts()
                    checklist_ok = True
                    for item_info in stats.values():
                        if not item_info['done']: checklist_ok = False
                    
                    self.final_verdict = "PASS" if checklist_ok else "FAIL"
                    self.timer_start = time.time()
                    return "FINISHED"
                return remaining

            elif self.state == "SHOW_RESULT":
                elapsed = time.time() - self.timer_start
                if elapsed > 3.0: # Show 3s thôi
                    print("🔄 Reset Test")
                    return "RESET_NOW"
                return "SHOWING"
        return None

def main():
    if not os.path.exists(MODEL_ITEM_PATH): 
        print("❌ Không tìm thấy model!")
        return

    print(f"🎥 CHẾ ĐỘ TEST VIDEO: {TEST_CAM_NAME}")
    print(f"📂 File: {VIDEO_PATH}")

    # Load Models
    print("⏳ Đang load model...")
    model_items = YOLO(MODEL_ITEM_PATH)
    model_slots = YOLO(MODEL_SLOT_PATH)

    # Khởi tạo Video
    video_stream = VideoLooper(VIDEO_PATH)

    # Khởi tạo Config chỉ cho 1 Camera
    cam_config = CameraConfig(TEST_CAM_NAME)
    processor = FrameProcessor(cam_config)
    visualizer = Visualizer()
    flow_manager = SystemFlowManagerTest()

    # Canvas Setup
    total_w = PROC_W + DASHBOARD_WIDTH
    total_h = PROC_H
    
    try:
        while True:
            # 1. Đọc Frame
            frame = video_stream.read()
            if frame is None: break

            # Resize chuẩn 640x480
            resized = cv2.resize(frame, (PROC_W, PROC_H))
            
            # Tạo batch frame (YOLO expect list)
            batch_frames = [resized]

            # 2. AI Predict
            # Stream=True giúp chạy mượt hơn với video file, nhưng ở đây ta dùng list nên để stream=False
            res_slots = model_slots.predict(batch_frames, conf=0.5, verbose=False)[0]
            res_items = model_items.predict(batch_frames, conf=0.45, verbose=False)[0]

            # 3. Process Logic
            detected = processor.process(res_slots, res_items)

            # 4. Vẽ lên ảnh
            display_frame = resized.copy()
            
            # Vẽ Slot
            for slot in cam_config.slots.values():
                visualizer.draw_slot_obb(display_frame, slot)
            
            # Vẽ Item
            if res_items.boxes:
                for b, c, cl in zip(res_items.boxes.xyxy.cpu().numpy(), res_items.boxes.conf.cpu().numpy(), res_items.boxes.cls.cpu().numpy()):
                    visualizer.draw_item_box(display_frame, b, res_items.names[int(cl)], c)
            
            visualizer.draw_camera_info(display_frame, cam_config)

            # 5. Logic Flow (Chỉ kích hoạt nếu đang test Cam 4 hoặc muốn test giả lập)
            status = None
            if TEST_CAM_NAME == "cam_4":
                status = flow_manager.update(cam_config, detected)
                if status == "RESET_NOW":
                    cam_config.force_reset()
                    flow_manager.state = "IDLE"

            # --- GIAO DIỆN ---
            final_canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            final_canvas[0:PROC_H, 0:PROC_W] = display_frame

            # Dashboard bên phải
            dashboard_roi = final_canvas[:, -DASHBOARD_WIDTH:]
            dashboard_roi[:] = (20, 20, 20) # Màu nền xám đậm
            
            # Hiệu ứng Blink
            blink = int(time.time() * 5) % 2 == 0

            # Hiển thị kết quả Test
            if TEST_CAM_NAME == "cam_4":
                if flow_manager.state == "COUNTDOWN" and isinstance(status, float):
                    cv2.putText(final_canvas, f"{status:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
                
                elif flow_manager.state == "SHOW_RESULT":
                    color = (0, 255, 0) if flow_manager.final_verdict == "PASS" else (0, 0, 255)
                    msg = "PASS" if flow_manager.final_verdict == "PASS" else "FAIL"
                    if blink:
                        cv2.rectangle(final_canvas, (0,0), (total_w, total_h), color, 10)
                        cv2.putText(final_canvas, msg, (PROC_W//2 - 100, PROC_H//2), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

            # Vẽ Dashboard (Cần đưa vào list để tái sử dụng hàm cũ)
            visualizer.draw_dashboard_on_roi(dashboard_roi, [cam_config])

            # Show
            cv2.imshow(f"Test Mode - {TEST_CAM_NAME}", final_canvas)
            
            # Điều khiển tốc độ: Video file chạy rất nhanh, cần waitKey lâu hơn chút (30ms ~ 30fps)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord('p'): # Phím P để tạm dừng soi lỗi
                cv2.waitKey(-1)

    finally:
        video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()