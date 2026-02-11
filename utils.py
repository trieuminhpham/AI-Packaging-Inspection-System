import numpy as np
import cv2
from itertools import combinations

class GeometryUtils:
    @staticmethod
    def identify_slots_logic(centers):
        """
        Thuật toán định danh Slot (S1-S5) hỗ trợ xoay 4 chiều.
        """
        centers = np.array(centers)
        if len(centers) != 5:
            return None

        # --- BƯỚC 1: TÌM 3 ĐIỂM THẲNG HÀNG (L123) ---
        min_area = float('inf')
        best_g3 = None

        for c in combinations(range(5), 3):
            p1, p2, p3 = centers[list(c)]
            area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if area < min_area:
                min_area = area
                best_g3 = c
        
        if best_g3 is None: return None
        
        idx_g3 = list(best_g3)                
        idx_g2 = [i for i in range(5) if i not in idx_g3] 

        pts_g3 = centers[idx_g3] 
        pts_g2 = centers[idx_g2] 

        # --- BƯỚC 2: TÌM S2 ---
        dists = [sum(np.linalg.norm(pts_g3[i] - pts_g3[j]) for j in range(3) if i != j) for i in range(3)]
        idx_s2 = idx_g3[np.argmin(dists)]
        s2_pos = centers[idx_s2]

        candidates = [i for i in idx_g3 if i != idx_s2]
        c1_idx, c2_idx = candidates[0], candidates[1]
        c1_pos, c2_pos = centers[c1_idx], centers[c2_idx]

        # --- BƯỚC 3: XÁC ĐỊNH HƯỚNG ---
        dx_g3 = np.max(pts_g3[:, 0]) - np.min(pts_g3[:, 0])
        dy_g3 = np.max(pts_g3[:, 1]) - np.min(pts_g3[:, 1])
        
        avg_x_g3 = np.mean(pts_g3[:, 0])
        avg_y_g3 = np.mean(pts_g3[:, 1])
        avg_x_g2 = np.mean(pts_g2[:, 0])
        avg_y_g2 = np.mean(pts_g2[:, 1])

        s1_idx, s3_idx = None, None

        if dy_g3 > dx_g3: # DỌC
            if avg_x_g2 < avg_x_g3: 
                if c1_pos[1] > s2_pos[1]: s1_idx, s3_idx = c1_idx, c2_idx
                else: s1_idx, s3_idx = c2_idx, c1_idx
            else:
                if c1_pos[1] < s2_pos[1]: s1_idx, s3_idx = c1_idx, c2_idx
                else: s1_idx, s3_idx = c2_idx, c1_idx
        else: # NGANG
            if avg_y_g2 < avg_y_g3:
                if c1_pos[0] < s2_pos[0]: s1_idx, s3_idx = c1_idx, c2_idx
                else: s1_idx, s3_idx = c2_idx, c1_idx
            else:
                if c1_pos[0] > s2_pos[0]: s1_idx, s3_idx = c1_idx, c2_idx
                else: s1_idx, s3_idx = c2_idx, c1_idx

        # --- BƯỚC 4: S4, S5 ---
        idx_g2_1 = idx_g2[0]
        idx_g2_2 = idx_g2[1]
        dist_1 = np.linalg.norm(centers[idx_g2_1] - centers[s1_idx])
        dist_2 = np.linalg.norm(centers[idx_g2_2] - centers[s1_idx])

        if dist_1 < dist_2: s4_idx, s5_idx = idx_g2_1, idx_g2_2
        else: s4_idx, s5_idx = idx_g2_2, idx_g2_1

        return {
            1: centers[s1_idx].astype(int),
            2: centers[idx_s2].astype(int),
            3: centers[s3_idx].astype(int),
            4: centers[s4_idx].astype(int),
            5: centers[s5_idx].astype(int)
        }

    @staticmethod
    def calculate_iou_polygon(box_item, poly_slot):
        """Tính diện tích giao nhau giữa Item và Slot OBB"""
        x1, y1, x2, y2 = box_item
        poly_item = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        poly_slot = poly_slot.astype(np.float32)

        # --- [QUAN TRỌNG] TÍNH DIỆN TÍCH ITEM TRƯỚC ---
        # Đây là dòng bị thiếu gây ra lỗi NameError
        item_area = (x2 - x1) * (y2 - y1)
        
        if item_area <= 0: return 0.0, 0.0, 0.0

        # Tính giao nhau
        ret, intersect_pts = cv2.intersectConvexConvex(poly_item, poly_slot)
        intersection_area = cv2.contourArea(intersect_pts) if (ret and intersect_pts is not None) else 0.0
        
        # Tính diện tích Slot
        slot_area = cv2.contourArea(poly_slot)

        # Chọn mẫu số là diện tích nhỏ nhất (giúp bao dung hơn với box bị to)
        denominator = min(item_area, slot_area)
        
        if denominator <= 0: return 0.0, 0.0, 0.0

        ratio = intersection_area / denominator
        return intersection_area, item_area, ratio

    @staticmethod
    def is_item_in_slot(box_item, poly_slot, threshold=0.45): 
        """
        Wrapper check va chạm. 
        Threshold để 0.45 để khắc phục lỗi góc cam nghiêng.
        """
        _, _, ratio = GeometryUtils.calculate_iou_polygon(box_item, poly_slot)
        return ratio >= threshold