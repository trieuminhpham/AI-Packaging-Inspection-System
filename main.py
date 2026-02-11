import cv2
# [QUAN TRỌNG] Ngăn OpenCV tạo luồng con gây xung đột với PyTorch
cv2.setNumThreads(0)

import numpy as np
import os
import time
from threading import Thread, Lock
from ultralytics import YOLO
from config import CameraConfig
from visualizer import Visualizer
from processor import FrameProcessor

# --- CẤU HÌNH ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
MODEL_ITEM_PATH = r"/media/edabk/IOTK68/Working_space_phuc/quangminh/CKI/best (1).pt"
MODEL_SLOT_PATH = r"/media/edabk/IOTK68/Working_space_phuc/quangminh/CKI/best.pt"

RTSP_URLS = [
    "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main", # CAM 1
    "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main", # CAM 2
    "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main", # CAM 3
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main", # CAM 4
]

PROC_W, PROC_H = 640, 480 
DASHBOARD_WIDTH = 350 

# --- CLASS CAMERA AN TOÀN ---
class SafeCameraStream:
    def __init__(self, rtsp_url, cam_id):
        self.url = rtsp_url
        self.cam_id = cam_id
        self.frame = None
        self.stopped = False
        self.lock = Lock()
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened(): print(f"❌ Lỗi: {cam_id}")
        else:
            ret, frame = self.cap.read()
            if ret: self.frame = frame

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            if ret:
                with self.lock: self.frame = frame
            else:
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

    def read(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- CLASS QUẢN LÝ QUY TRÌNH (LOGIC CAM 4 TRIGGER) ---
class SystemFlowManager:
    def __init__(self):
        self.timer_start = None
        self.final_verdict = None # PASS / FAIL
        self.state = "IDLE" 
        # IDLE: Chờ khay xuất hiện ở Cam 4
        # RUNNING: Cam 4 đang thấy khay
        # COUNTDOWN: Cam 4 vừa mất khay -> Đếm 10s
        # SHOW_RESULT: Hiện kết quả OK/WRONG

    def update(self, configs, cam4_detected):
        """
        Logic dựa trên tín hiệu của Cam 4
        """
        # 1. NẾU CAM 4 THẤY KHAY -> ĐANG LÀM VIỆC
        if cam4_detected:
            self.state = "RUNNING"
            self.timer_start = None
            self.final_verdict = None
            return None

        # 2. NẾU CAM 4 KHÔNG THẤY KHAY (Mất tín hiệu)
        else:
            # Nếu trước đó đang chạy (RUNNING) mà giờ mất -> Chuyển sang ĐẾM NGƯỢC
            if self.state == "RUNNING":
                self.state = "COUNTDOWN"
                self.timer_start = time.time()
                print("🏁 Cam 4 mất tín hiệu -> Bắt đầu đếm ngược 10s...")
            
            # Nếu đang ở trạng thái IDLE (chưa chạy bao giờ) -> Kệ nó
            elif self.state == "IDLE":
                return None

            # --- XỬ LÝ ĐẾM NGƯỢC ---
            if self.state == "COUNTDOWN":
                elapsed = time.time() - self.timer_start
                remaining = 10.0 - elapsed
                
                if remaining <= 0:
                    # Hết 10s -> CHỐT KẾT QUẢ CHECKLIST TOÀN BỘ
                    self.state = "SHOW_RESULT"
                    
                    # Quét toàn bộ checklist của 4 Cam
                    checklist_ok = True
                    for cfg in configs:
                        stats = cfg.get_item_counts()
                        for item_info in stats.values():
                            # Kiểm tra từng item xem đã đủ chưa
                            if not item_info['done']: 
                                checklist_ok = False
                                # Debug để biết thiếu cái gì
                                # print(f"Thiếu: {cfg.cam_name} - {item_info}")
                    
                    self.final_verdict = "PASS" if checklist_ok else "FAIL"
                    self.timer_start = time.time() # Reset timer để dùng cho việc show result
                    return "FINISHED"
                
                return remaining # Trả về số giây để vẽ

            # --- XỬ LÝ HIỂN THỊ KẾT QUẢ ---
            elif self.state == "SHOW_RESULT":
                elapsed = time.time() - self.timer_start
                # Hiển thị kết quả trong 5 giây rồi RESET
                if elapsed > 5.0:
                    print("🔄 Kết thúc hiển thị -> Reset Hệ Thống")
                    return "RESET_NOW"
                return "SHOWING"

        return None

def main():
    if not os.path.exists(MODEL_ITEM_PATH): return

    print(f"🚀 HỆ THỐNG AN TOÀN (SAFE MODE) - LOGIC CAM 4")

    model_items = YOLO(MODEL_ITEM_PATH)
    model_slots = YOLO(MODEL_SLOT_PATH)

    streams = []
    cam_names = ["cam_1", "cam_2", "cam_3", "cam_4"]
    
    print("⏳ Đang khởi tạo Camera...")
    for i, url in enumerate(RTSP_URLS):
        print(f"   -> Cam {i+1}...")
        s = SafeCameraStream(url, cam_names[i]).start()
        streams.append(s)
        time.sleep(0.5)

    configs = [CameraConfig(name) for name in cam_names]
    processors = [FrameProcessor(cfg) for cfg in configs]
    visualizer = Visualizer()
    flow_manager = SystemFlowManager() # Class quản lý mới

    total_w = (PROC_W * 2) + DASHBOARD_WIDTH
    total_h = PROC_H * 2
    main_canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    try:
        while True:
            # 1. Đọc ảnh
            batch_frames = []
            valid_indices = []
            
            for i, stream in enumerate(streams):
                frame = stream.read()
                if frame is not None:
                    try:
                        resized = cv2.resize(frame, (PROC_W, PROC_H))
                        batch_frames.append(resized)
                        valid_indices.append(i)
                    except: batch_frames.append(np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8))
                else:
                    black = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
                    cv2.putText(black, "NO SIGNAL", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    batch_frames.append(black)

            # 2. AI Predict
            if batch_frames:
                res_slots = model_slots.predict(batch_frames, conf=0.5, verbose=False, stream=False)
                res_items = model_items.predict(batch_frames, conf=0.45, verbose=False, stream=False)

            # 3. Process Logic
            cam4_detected = False # Biến quan trọng để trigger logic

            for i in range(4):
                dx, dy = (i % 2) * PROC_W, (i // 2) * PROC_H
                roi = main_canvas[dy:dy+PROC_H, dx:dx+PROC_W]
                np.copyto(roi, batch_frames[i])

                if i in valid_indices:
                    # detected = True nếu thấy khay
                    detected = processors[i].process(res_slots[i], res_items[i])
                    
                    # Kiểm tra riêng Cam 4
                    if i == 3: cam4_detected = detected 

                    # Vẽ
                    for slot in configs[i].slots.values():
                        visualizer.draw_slot_obb(roi, slot)
                    if res_items[i].boxes:
                        for b, c, cl in zip(res_items[i].boxes.xyxy.cpu().numpy(), res_items[i].boxes.conf.cpu().numpy(), res_items[i].boxes.cls.cpu().numpy()):
                            visualizer.draw_item_box(roi, b, res_items[i].names[int(cl)], c)
                    visualizer.draw_camera_info(roi, configs[i])

            # 4. LOGIC QUẢN LÝ LUỒNG (Dựa trên Cam 4)
            status = flow_manager.update(configs, cam4_detected)
            
            # Xử lý lệnh Reset
            if status == "RESET_NOW":
                for cfg in configs: cfg.force_reset()
                flow_manager.state = "IDLE"
            
            # --- VẼ GIAO DIỆN ---
            blink = int(time.time() * 4) % 2 == 0
            
            # A. Đếm ngược (Khi Cam 4 mất khay)
            if flow_manager.state == "COUNTDOWN" and isinstance(status, float):
                # Vẽ lên Cam 4 (Góc phải dưới)
                cv2.putText(main_canvas, f"FINAL CHECK: {status:.1f}s", (PROC_W+50, PROC_H+100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

            # B. Hiển thị kết quả (Sau 10s)
            elif flow_manager.state == "SHOW_RESULT":
                if flow_manager.final_verdict == "PASS" and blink:
                    cv2.putText(main_canvas, "OKE - DONE", (total_w//2-200, total_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)
                    cv2.rectangle(main_canvas, (0,0), (total_w, total_h), (0,255,0), 20)
                
                elif flow_manager.final_verdict == "FAIL" and blink:
                    cv2.putText(main_canvas, "WRONG / MISSING", (total_w//2-350, total_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
                    cv2.rectangle(main_canvas, (0,0), (total_w, total_h), (0,0,255), 20)

            # 5. Dashboard
            dashboard_roi = main_canvas[:, -DASHBOARD_WIDTH:]
            dashboard_roi[:] = (20, 20, 20)
            # Nhấp nháy đỏ Dashboard nếu Fail
            if flow_manager.state == "SHOW_RESULT" and flow_manager.final_verdict == "FAIL" and blink:
                dashboard_roi[:] = (0, 0, 100)
                
            visualizer.draw_dashboard_on_roi(dashboard_roi, configs)

            visualizer.draw_fps(main_canvas)
            cv2.imshow("Smart Packing System", main_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        for s in streams: s.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()