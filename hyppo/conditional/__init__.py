from .FCIT import FCIT
from .kci import KCI
from .kcipt import kernel_conditional_independence_test


__all__ = [s for s in dir()]  # add imported tests to __all__

COND_INDEP_TESTS = {"fcit": FCIT, "kci": KCI, "kcipt":kernel_conditional_independence_test()}
