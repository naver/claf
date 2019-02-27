
def recall(pycm_obj):
    return {key: pycm_obj.TPR[key] if pycm_obj.TPR[key] != "None" else 0. for key in pycm_obj.TPR}


def precision(pycm_obj):
    return {key: pycm_obj.PPV[key] if pycm_obj.PPV[key] != "None" else 0. for key in pycm_obj.PPV}


def f1(pycm_obj):
    return {key: pycm_obj.F1[key] if pycm_obj.F1[key] != "None" else 0. for key in pycm_obj.F1}


def macro_recall(pycm_obj):
    return sum(recall(pycm_obj).values()) / len(pycm_obj.classes)


def macro_precision(pycm_obj):
    return sum(precision(pycm_obj).values()) / len(pycm_obj.classes)


def macro_f1(pycm_obj):
    return sum(f1(pycm_obj).values()) / len(pycm_obj.classes)
