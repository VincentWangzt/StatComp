from runner.sivi import SIVIRunner
from runner.uivi import UIVIRunner
from runner.ruivi import RUIVIRunner
from runner.aisivi import AISIVIRunner
from runner.base_runner import BaseSIVIRunner

Runners: dict[str, type[BaseSIVIRunner]] = {
    "SIVI": SIVIRunner,
    "UIVI": UIVIRunner,
    "RUIVI": RUIVIRunner,
    "AISIVI": AISIVIRunner,
}
