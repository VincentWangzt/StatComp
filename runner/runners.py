from runner.sivi import SIVIRunner
from runner.uivi import UIVIRunner
from runner.rsivi import RSIVIRunner
from runner.aisivi import AISIVIRunner
from runner.dsivi import DSIVIRunner
from runner.ksivi import KSIVIRunner
from runner.kpg import KPGRunner
from runner.nfvi import NFVIRunner
from runner.base_runner import BaseSIVIRunner

Runners: dict[str, type[BaseSIVIRunner]] = {
    "SIVI": SIVIRunner,
    "UIVI": UIVIRunner,
    "RSIVI": RSIVIRunner,
    "AISIVI": AISIVIRunner,
    "DSIVI": DSIVIRunner,
    "KSIVI": KSIVIRunner,
    "KPG": KPGRunner,
    "NFVI": NFVIRunner,
}
