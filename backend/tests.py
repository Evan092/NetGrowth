import torch

from CombinedLoss import filter_and_trim_boxes


def testFilter():
        test_target_boxes = torch.tensor([[[1., 1., 2., 2.],
        [3., 3., 4., 4.]],
        [[2., 2., 3., 3.],
        [4., 4., 5., 5.]]])


        test_pred_boxes = torch.tensor([
            [  # Batch 1
                [  # Grid row 1
                    [[1.0, 1.0, 2.0, 2.0, 0.2], [1.5, 1.5, 2.5, 2.5, 0.65]],  # Grid cell 1
                    [[2.0, 2.0, 3.0, 3.0, 0.7], [2.5, 2.5, 3.5, 3.5, 0.6]]   # Grid cell 2
                ],
                [  # Grid row 2
                    [[3.0, 3.0, 4.0, 4.0, 0.1], [3.5, 3.5, 4.5, 4.5, 0.2]],  # Grid cell 3
                    [[4.0, 4.0, 5.0, 5.0, 0.3], [4.5, 4.5, 5.5, 5.5, 0.8]]   # Grid cell 4
                ]
            ],
            [  # Batch 2 (same structure as above)
                [  # Grid row 1
                    [[5.0, 5.0, 6.0, 6.0, 0.5], [5.5, 5.5, 6.5, 6.5, 0.4]],  # Grid cell 1
                    [[6.0, 6.0, 7.0, 7.0, 0.3], [6.5, 6.5, 7.5, 7.5, 0.2]]   # Grid cell 2
                ],
                [  # Grid row 2
                    [[7.0, 7.0, 8.0, 8.0, 0.9], [7.5, 7.5, 8.5, 8.5, 0.8]],  # Grid cell 3
                    [[8.0, 8.0, 9.0, 9.0, 0.7], [8.5, 8.5, 9.5, 9.5, 0.6]]   # Grid cell 4
                ]
            ]
        ], dtype=torch.float32)


        test_target_boxes, test_pred_boxes, test_confidences_flat = filter_and_trim_boxes(test_pred_boxes, test_target_boxes)