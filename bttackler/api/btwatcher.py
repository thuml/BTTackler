import logging
import os

import nni

from bttackler.common.btmessenger import BTMessenger
from bttackler.bridger.btmonitor import BTMonitor
from bttackler.common.utils import set_seed

logger = logging.getLogger(__name__)


class BTWatcher:
    def __init__(self, seed=None):
        set_seed(seed, "manager", logger)
        self.seed = seed

        self.advisor_config = BTMessenger().read_advisor_config()
        self.shared_config = None
        self.monitor_config = None
        self.assessor_config = None
        self.stop_trigger = 0
        self.init_configs()

        self.monitor = BTMonitor(**self.monitor_config) if self.monitor_config is not None else None

        self.raw_mode = False if self.shared_config is not None and self.monitor_config is not None else True  # no inspect/assess maybe tuner

    def init_configs(self):
        if self.advisor_config is None:
            return
        self.shared_config = self.advisor_config["shared"] \
            if "shared" in self.advisor_config else None
        self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
            if "monitor" in self.advisor_config else None
        self.assessor_config = self.advisor_config["assessor"]["classArgs"] \
            if "assessor" in self.advisor_config else None

    def get_raw_dict(self, result_dict):
        if type(result_dict):
            return result_dict
        elif type(result_dict) is int or type(result_dict) is float:
            return {"default": result_dict}
        else:
            return {"default": 0}

    def collect_per_batch(self, *args):
        self.monitor.collect_in_training(*args)

    def collect_after_training(self, *args):
        self.monitor.collect_after_training(*args)
        self.monitor.calculate_after_training()

    def collect_after_validating(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_validating(*args)

    def collect_after_testing(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_testing(*args)

    def init_basic(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_basic(*args)

    def init_cond(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_cond(*args)

    def refresh_before_epoch_start(self):
        self.monitor.refresh_before_epoch_start()

    def report_intermediate_result(self, rd=None, writer=None):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_intermediate_dict()
            BTMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.raw_mode is True:
            d = self.get_raw_dict(rd)
        logger.info(" ".join(["intermediate_result_dict:", str(d)]))
        nni.report_intermediate_result(d)  # assessor _metric
        return d

    def report_final_result(self, rd=None, writer=None):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_final_dict()
            BTMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.assessor_config is not None:
            d3 = BTMessenger().read_assessor_info()
            while d3 is None:
                d3 = BTMessenger().read_assessor_info()
                os.system("sleep 1")
            d.update(d3)
        if self.raw_mode is True:
            d = self.get_raw_dict(rd)
        logger.info(" ".join(["final_result_dict:", str(d)]))
        nni.report_final_result(d)  # tuner symptom
        return d

    def stop_by_diagnosis(self):
        if self.assessor_config is not None:
            early_stop = False
            info_dict = BTMessenger().read_assessor_info()
            if info_dict is None:
                early_stop = False
            elif info_dict["early_stop"] is True:
                early_stop = True
            if early_stop:
                logger.info(" ".join(["assessor_info_dict ", str(info_dict)]))
                self.stop_trigger += 1
            else:
                self.stop_trigger = 0
            return self.stop_trigger> 0
        else:
            return False
