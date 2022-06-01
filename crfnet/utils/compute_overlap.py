import numpy as np


def compute_overlap(
    boxes: np.array,
    query_boxes: np.array
):
    """计算overlap

    Args:
        boxes (np.array): (N, 4)
        query_boxes (np.array): (K, 4)
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float64)
    # box_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
    #             (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw*ih/ua

    return overlaps
