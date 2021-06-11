"""
to guide the inter-view embedding to concentrate on the information
tightly related to links in inter-view graphs
"""


def dissent_loss(res1, res2):
    pass


def rep_gap_loss():
    pass


def auto_weight_focal_loss():
    pass


def classification_loss(res, gts):
    """
    For DDI prediction, we have formulate it as a classification task
    """
    # difference between 2 predictors
    # final predictor results vs ground truths

    pass


def regression_loss():
    """
    For DTA prediction, we have formulate it as a regression task
    """
    # difference between 2 predictors
    # final predictor results vs ground truths

    pass

