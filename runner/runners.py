from runner.sivi import SIVIRunner
from runner.uivi import UIVIRunner
from runner.rsivi import RSIVIRunner
from runner.aisivi import AISIVIRunner
from runner.base_runner import BaseSIVIRunner

Runners: dict[str, type[BaseSIVIRunner]] = {
    "SIVI": SIVIRunner,
    "UIVI": UIVIRunner,
    "RSIVI": RSIVIRunner,
    "AISIVI": AISIVIRunner,
}
