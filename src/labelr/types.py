import enum


class ExportSource(str, enum.Enum):
    hf = "hf"
    ls = "ls"
    ultralytics = "ultralytics"


class ExportDestination(str, enum.Enum):
    hf = "hf"
    ultralytics = "ultralytics"


class TaskType(str, enum.Enum):
    object_detection = "object_detection"
    classification = "classification"
