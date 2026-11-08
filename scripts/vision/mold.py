# -*- coding: utf-8 -*-
"""霉菌检测核心：分割 -> 双线索（黑霉 / 白霉）-> 斑块过滤 -> 单帧/多帧判定。

两条线索对应两种视角下霉的**两种不同外观**，不是同一件事的两个近似：

``cue="V"``  亮度黑帽（正面 / 未翻腕，看的是侧壁）。``BH = close(V,k) - V``，量化
             "比周围暗多少"。真霉斑 maxBH=164~228，表面暗纹理只有 72~74。最硬的一条。
``cue="W"``  白/灰霉（反面 / 已翻腕）。顶面的霉**不是黑的**：
             loop1 顶面整幅近纯黑像素为 0（BH_V 最大 134），正面却是 vmin=0 的近纯黑。
             顶面的霉是一层灰白覆盖膜，特征是**饱和度掉到中间带** —— 见 ``_white_spots``。

**不要用硬编码小矩形当搜索范围**（2026-11-08 复核推翻）：
``config.TOP_ROI`` 那个框对应的是 1.34 m 旧录像的位姿，在现在的抓取位姿下正好压在
顶棱渐变带上（真霉全落在框外 60~220 px），是现场"反面永远测不出来"的直接原因。
现在白霉线索跑**整块可见奶酪面**，靠物理判据（dS>=45、外环黄占比>=0.80）区分
霉 / 棱线 / 渐变 / 侧壁蹭痕 —— 实测三者分离极宽，见 `_white_spots` 的注释。

已删除：曾经的"顶面掉黄度"(BH_b) 线索。复盘 2252/23:36 共 24 帧反面，它的落点
**全部**在顶面带的上/下边界内 4~7 px，是轮廓转折棱线的边缘响应，从未命中真霉。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import config


# --------------------------------------------------------------------------
# 分割
# --------------------------------------------------------------------------
def segment(bgr):
    """把奶酪从背景里抠出来。返回 dict(inner, solid, bbox, erode, hsv) 或 None。"""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(
        hsv,
        np.array([config.CHEESE_H[0], config.CHEESE_S_MIN, config.CHEESE_V_MIN]),
        np.array([config.CHEESE_H[1], 255, 255]),
    )
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, np.ones((config.CHEESE_CLOSE_K, config.CHEESE_CLOSE_K), np.uint8)
    )
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, np.ones((config.CHEESE_OPEN_K, config.CHEESE_OPEN_K), np.uint8)
    )

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[idx, cv2.CC_STAT_AREA]) < config.CHEESE_MIN_AREA_RATIO * h * w:
        return None
    blob = (lab == idx).astype(np.uint8) * 255

    # 填洞：从外边界 floodFill，没被填到的封闭空腔就是内部孔洞，补回奶酪实体
    pad = cv2.copyMakeBorder(blob, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    mk = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 255)
    solid = cv2.bitwise_or(pad, cv2.bitwise_not(ff))[1:-1, 1:-1]

    x, y, bw, bh = cv2.boundingRect(solid)
    er = max(3, int(round(min(bw, bh) * config.CHEESE_ERODE_FRAC)) | 1)
    inner = cv2.erode(solid, np.ones((er, er), np.uint8))
    return dict(inner=inner, solid=solid, bbox=(x, y, bw, bh), erode=er, hsv=hsv)


# --------------------------------------------------------------------------
# 找菌斑
# --------------------------------------------------------------------------
def _spots_from_mask(mask, cue, score, inner, inner_area, bgr_planes):
    """对一张候选掩码跑几何过滤，返回通过的斑块列表。

    ``score`` 是判决用的响应图（V 路是 BH_V），记进结果的 ``depth`` 字段，
    方便事后看证据。
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    amin = max(config.SPOT_MIN_ABS, config.SPOT_MIN_RATIO * inner_area)
    amax = config.SPOT_MAX_RATIO * inner_area
    dist = cv2.distanceTransform((inner > 0).astype(np.uint8), cv2.DIST_L2, 5)
    need_dist = float(config.SPOT_MIN_DIST)
    V, S = bgr_planes

    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < amin or a > amax:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh_) / max(1.0, min(bw, bh_)) > config.SPOT_MAX_AR:
            continue
        sel = lab == i
        cnts, _ = cv2.findContours(sel.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        per = cv2.arcLength(cnts[0], True)
        circ = 4 * np.pi * a / (per * per) if per > 0 else 0.0
        if circ < config.SPOT_MIN_CIRC:
            continue
        d = float(dist[sel].mean())
        if d < need_dist:
            continue

        vv, ss = V[sel], S[sel]
        depth = int(score[sel].max())
        out.append(dict(
            kind="mold", cue=cue, area=a, w=bw, h=bh_,
            ratio=round(a / float(inner_area), 6), circ=round(float(circ), 3),
            dist=round(d, 1), cx=int(cent[i][0]), cy=int(cent[i][1]),
            vmin=int(vv.min()), vmed=round(float(np.median(vv)), 1),
            smed=round(float(np.median(ss)), 1),
            bh=round(float(score[sel].mean()), 1), depth=depth,
        ))
    return out


def _drop_duplicates(strong, weak, radius=12.0):
    """weak 里和 strong 已经圈到的位置重合的，丢掉（V 证据优先）。"""
    kept = []
    for cand in weak:
        dup = False
        for base in strong:
            if (math.hypot(cand["cx"] - base["cx"], cand["cy"] - base["cy"]) < radius
                    and cand["area"] <= base["area"] * 2.0):
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def roi_mask(shape):
    """用户框定的顶面 ROI，返回 uint8 掩膜（255 = 框内）。

    ROI 由用户给定，**不要用轮廓去自标定**。框是准的，自标定会漂。
    见 ``config.TOP_ROI`` 的注释。
    """
    m = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = config.TOP_ROI
    h, w = shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


def top_face(seg):
    """[已废弃] 按轮廓自标定的顶面带。保留仅供调试画图，**判决不再使用**。

    废弃原因：它会随相机高度和奶酪位置漂移，一旦跑到侧壁上，就把侧壁的
    白色蹭痕当成霉。现在顶面范围由用户框定的 ``config.TOP_ROI`` 决定。
    """
    solid = seg["solid"]
    inner = seg["inner"]
    widths = (solid > 0).sum(axis=1)
    rows = np.flatnonzero(widths)
    if rows.size == 0:
        return np.zeros_like(solid)
    top = int(rows[0])
    bottom = int(rows[-1])
    rim = top + int(np.argmax(widths[top:bottom + 1]))
    band = np.zeros_like(solid)
    band[top:rim + 1] = 255
    return cv2.bitwise_and(band, inner)


def cheese_fully_visible(seg, shape, margin=3):
    """[已废弃] 奶酪是否完整在画面里。判决不再调用，保留供调试。

    旧作用是给白霉线索当"是不是顶面视角"的 gate。现在这个 gate 由固定
    ROI 天然承担：正面查看时奶酪贴画面底边，``TOP_ROI`` 内奶酪面积为 0。
    """
    x, y, w, h = seg["bbox"]
    height, width = shape[:2]
    if w <= 0 or h <= 0:
        return False
    return (x > margin and y > margin
            and x + w < width - margin and y + h < height - margin)


def _white_spots(bgr, seg, inner_area):
    """可见奶酪面上的白/灰霉：**局部饱和度显著凹陷** + **外环被黄色包围**的紧凑斑。

    反面（已翻腕）看到的是顶面，那里的霉是灰白覆盖膜：比奶酪本体黄度低、
    亮度略低。判据用三个条件相乘：

    1. **饱和度相对凹陷**：``dip = median(S, WHITE_BG_K) - S >= WHITE_S_DIP``。
       这里必须用**比霉斑大得多**的中值窗口求背景 —— 旧代码用 21x21，而霉斑
       本身约 50x50，窗口整块落在斑内部，背景就等于斑自己，dip≈0，
       于是反面永远 0 命中（2026-11-08 现场"反面测不出来"的根因）。
       放到 101x101 后，同样的霉斑 dip 从 0~4 升到 86~161。
    2. **外环必须被黄色包围**（``yf``）：真霉是长在黄色表面上的离散浅斑，
       外环 360 度都是黄本体；顶棱那条明暗渐变带一侧贴着背景，实测 yf 只有
       0.46~0.49，而真霉 0.83~1.00。**这一条是"放宽 dS 又不误报"的关键** ——
       没有它，顶棱渐变带会圈走 12~26% 面积、把真霉挤掉。
    3. **绝度过曝/本体排除**：斑内中位 S <= ``WHITE_S_CORE_MAX``，
       V 落在 ``[WHITE_V_MIN, WHITE_V_MAX]``。上限 V 掐掉顶灯直射的镜面白斑
       （实测 V 249~255，那东西 S 也很低，光看 S 会误判）。

    **搜索范围 = 整块奶酪的可见面**（``seg["inner"]``），不再用 ``config.TOP_ROI``
    那个硬编码小框。原因见 ``config.TOP_ROI`` 的注释：那个框对应的是 1.34 m 旧录像
    的位姿，在现在的抓取位姿下正好落在顶棱渐变带上、真霉全在框外，
    是"反面永远测不出来"的直接原因。这里仍**不做轮廓自标定**（用户要求），
    改用物理判据（dS + 外环黄占比）区分"霉"与"棱线/渐变/蹭痕"。
    """
    inner = seg["inner"] > 0
    hsv = seg["hsv"]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    if int(inner.sum()) < 1:
        return []

    k = int(config.WHITE_BG_K) | 1
    bg = cv2.medianBlur(S, k).astype(np.float32)       # 局部背景饱和度
    dip = bg - S.astype(np.float32)                    # 比背景掉了多少
    m = ((dip >= config.WHITE_S_DIP) & inner).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 外环判据用的"黄本体"掩膜
    yellow = ((S >= config.WHITE_RING_S) & (V >= config.WHITE_RING_V)).astype(np.uint8)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < config.WHITE_MIN_AREA or a > config.WHITE_MAX_AREA:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh_) / max(1.0, min(bw, bh_)) > config.WHITE_MAX_AR:
            continue
        sel = lab == i
        s_med = float(np.median(S[sel]))
        if not (config.WHITE_S_LO <= s_med <= config.WHITE_S_CORE_MAX):
            continue
        v_med = float(np.median(V[sel]))
        if not (config.WHITE_V_MIN <= v_med <= config.WHITE_V_MAX):
            continue

        # 外环黄占比：斑块外 5~15 px 那一圈里，有多少是黄奶酪本体
        sel8 = sel.astype(np.uint8)
        d_in = cv2.dilate(sel8, np.ones((11, 11), np.uint8))
        d_out = cv2.dilate(sel8, np.ones((31, 31), np.uint8))
        ring = (d_out > 0) & (d_in == 0)
        if int(ring.sum()) < 1:
            continue
        yf = float(yellow[ring].mean())
        if yf < config.WHITE_RING_YELLOW:
            continue

        cnts, _ = cv2.findContours(sel8 * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        per = cv2.arcLength(cnts[0], True) if cnts else 0.0
        circ = 4 * np.pi * a / (per * per) if per > 0 else 0.0
        d_med = float(np.median(dip[sel]))
        out.append(dict(
            kind="mold", cue="W", area=a, w=bw, h=bh_,
            ratio=round(a / float(inner_area), 6), circ=round(float(circ), 3),
            dist=0.0, cx=int(cent[i][0]), cy=int(cent[i][1]),
            vmin=int(V[sel].min()), vmed=round(v_med, 1), smed=round(s_med, 1),
            bh=round(d_med, 1), depth=int(d_med), yf=round(yf, 3),
        ))
    return out


def find_spots(bgr, seg):
    """在奶酪内部找菌斑。返回 (spots, inner_area)。

    * ``V``：亮度黑帽，全奶酪跑。正面（未翻腕）看到的黑霉靠它 —— 实测真霉斑是
      vmin=0 / vmed=20 / S=255 的近纯黑块，BH_V 能到 220~255。正面看的是侧壁，
      霉在侧壁上，所以这条线索**不受顶面 ROI 限制**。
    * ``W``：白/灰霉，**在整块可见奶酪面上跑**（``seg["inner"]``）—— 见 ``_white_spots``。
      反面/顶面视图的霉不是黑的（近纯黑像素为 0），用黑霉判据必然漏检。
      它靠"局部饱和度凹陷 + 外环被黄本体 360 度包围"区分霉与棱线/渐变/蹭痕，
      不需要（也不能用）一个会随位姿漂移的硬编码 ROI 框。
    """
    inner, hsv = seg["inner"], seg["hsv"]
    inner_area = int((inner > 0).sum())
    if inner_area < 1:
        return [], 0

    V = hsv[:, :, 2]
    S = hsv[:, :, 1]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.BH_KSIZE, config.BH_KSIZE))
    planes = (V, S)

    # 线索 1：亮度黑帽（正面黑霉，原判据）
    bh_v = cv2.bitwise_and(cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, kern), inner)
    mask_v = (bh_v >= config.BH_THR).astype(np.uint8) * 255
    spots_v = _spots_from_mask(mask_v, "V", bh_v, inner, inner_area, planes)
    spots_v = [s for s in spots_v if s["depth"] >= config.SPOT_MIN_DEPTH]

    # 线索 2：白/灰霉，整块可见面（判据自带几何约束，见 _white_spots）
    spots_w = _white_spots(bgr, seg, inner_area)

    spots = spots_v + _drop_duplicates(spots_v, spots_w)
    spots.sort(key=lambda d: -d["area"])
    return spots, inner_area


# --------------------------------------------------------------------------
# 单帧 / 一次查看
# --------------------------------------------------------------------------
def analyze_bgr(bgr, name="frame"):
    """对一帧 BGR 出判定记录。"""
    if bgr is None or bgr.size == 0:
        return dict(frame=name, ok=False, err="empty_image", cheese=False,
                    spots=[], mold=False, max_ratio=0.0, n_mold=0)
    rec = dict(frame=name, ok=True, w=int(bgr.shape[1]), h=int(bgr.shape[0]))
    seg = segment(bgr)
    if seg is None:
        rec.update(cheese=False, spots=[], mold=False, max_ratio=0.0, n_mold=0)
        return rec
    rec["cheese"] = True
    rec["bbox"] = [int(v) for v in seg["bbox"]]
    spots, ia = find_spots(bgr, seg)
    rec["inner_area"] = int(ia)
    rec["spots"] = spots
    rec["n_mold"] = len(spots)
    mx = max([s["ratio"] for s in spots], default=0.0)
    rec["max_ratio"] = mx
    rec["mold"] = bool(mx >= config.FRAME_MIN_RATIO and len(spots) >= 1)
    return rec


def judge_frames(recs):
    """一次查看（多帧）出结论：过半帧命中 -> 这一面有霉菌。"""
    good = [r for r in recs if r.get("cheese")]
    if not good:
        return dict(mold=False, verdict="看不到奶酪", hits=0, n=0, n_frames=len(recs),
                    hit_ratio=0.0, max_ratio=0.0, reason="no_cheese", frames=recs)
    hits = sum(1 for r in good if r["mold"])
    ratio = hits / float(len(good))
    mx = max(r.get("max_ratio", 0.0) for r in good)
    hit = bool(ratio >= config.HIT_FRAME_RATIO)
    return dict(mold=hit, verdict="有霉菌" if hit else "没有霉菌", hits=hits, n=len(good),
                n_frames=len(recs), hit_ratio=round(ratio, 3), max_ratio=round(mx, 6),
                reason="ok", frames=recs)


def annotate(bgr, rec):
    """把结论画到图上，存证据用。绿框=奶酪轮廓，红圈=命中斑。

    ⚠️ 不再画 ``TOP_ROI``：那个框和现场位姿对不上、已不作为搜索范围
    （见 ``config.TOP_ROI`` 注释）。画上去只会误导看图的人以为霉应该在框里。
    """
    if bgr is None:
        return None
    img = bgr.copy()
    bbox = rec.get("bbox")
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
    for s in rec.get("spots", []):
        cv2.circle(img, (s["cx"], s["cy"]), max(6, int(s["w"] / 2)), (0, 0, 255), 2)
    cues = sorted({s.get("cue", "V") for s in rec.get("spots", [])})
    label = "MOLD {} spots [{}]  max {:.4f}%".format(len(rec.get("spots", [])),
                                                     "/".join(cues) or "-",
                                                     rec.get("max_ratio", 0.0) * 100)
    color = (0, 0, 255) if rec.get("mold") else (0, 200, 0)
    cv2.putText(img, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(img, str(rec.get("frame", "")), (24, 88), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return img
