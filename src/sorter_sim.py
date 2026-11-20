"""
Simulated sorting mechanism.

Per the charter's own risk note, a physical sorting mechanism may not
be feasible in the competition timeline, and a software/simulated
fallback should be ready early rather than as a last resort. This
module is that fallback: it takes a classification result and
"acts" on it the same way a real sorter would (routes it to a bin),
just without the motors.

If Micah gets a physical mechanism (servo/motor + bins) working,
swap PhysicalSorter in for SimulatedSorter below without touching
infer.py's calling code, since both expose the same .sort() method.
"""
import time
from dataclasses import dataclass, field

BIN_MAP = {
    "plastic": "Bin 1 - Plastic",
    "paper": "Bin 2 - Paper",
    "glass": "Bin 3 - Glass",
    "metal": "Bin 4 - Metal",
}


@dataclass
class SimulatedSorter:
    classes: list
    conf_threshold: float = 0.6
    log: list = field(default_factory=list)

    def sort(self, label: str, confidence: float):
        """Route a classified item to its bin, or to 'uncertain' if the
        model wasn't confident enough. Prints a line that stands in for
        a physical actuator firing, and records it for the demo log /
        pitch deck screenshots."""
        if confidence < self.conf_threshold:
            action = "-> UNCERTAIN bin (needs a human look)"
            bin_name = "uncertain"
        else:
            bin_name = BIN_MAP.get(label, label)
            action = f"-> routing to {bin_name}"
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] classified '{label}' ({confidence:.0%}) {action}"
        print(line)
        self.log.append(
            {"time": timestamp, "label": label, "confidence": confidence, "bin": bin_name}
        )
        return bin_name


# --- Stub for a physical mechanism, if Micah gets hardware working ---
# Swap this in for SimulatedSorter in infer.py's `sorter = ...` line.
# Keep the .sort(label, confidence) signature identical so infer.py
# doesn't need to change.
class PhysicalSorter:
    def __init__(self, classes, conf_threshold: float = 0.6, serial_port: str = "/dev/ttyUSB0"):
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.serial_port = serial_port
        # TODO: open serial connection to the microcontroller driving
        # the servo/motor here, e.g. via pyserial.
        raise NotImplementedError(
            "Wire this up once the physical sorting mechanism exists. "
            "Keep the same .sort(label, confidence) -> bin_name interface."
        )

    def sort(self, label: str, confidence: float):
        raise NotImplementedError
