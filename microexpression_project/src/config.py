"""
Shared constants for micro-expression recognition system.
Ensures consistent emotion label mapping across all modules.
"""

# Canonical emotion label mapping - indices used by CNN / SVM / sklearn classes_
EMOTION_LABELS = {
    "happiness": 0,
    "surprise": 1,
    "disgust": 2,
    "repression": 3,
    "others": 4,
}

# Reverse mapping for predictions
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}

# Canonical optical-flow tensor format used across the project:
# (fx1, fy1, fx2, fy2, strain1, strain2) => 6 channels
FLOW_CHANNELS = 6

# Display names (capitalized for plots/reports)
EMOTION_DISPLAY_NAMES = {
    "happiness": "Happiness",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "repression": "Repression",
    "others": "Others",
}

# Ordered list for consistent plotting (matches numeric labels 0..N-1)
_EMOTION_KEYS_ORDERED = sorted(EMOTION_LABELS.keys(), key=lambda k: EMOTION_LABELS[k])
EMOTION_ORDER = [EMOTION_LABELS[k] for k in _EMOTION_KEYS_ORDERED]
EMOTION_DISPLAY_ORDER = [EMOTION_DISPLAY_NAMES[k] for k in _EMOTION_KEYS_ORDERED]

# CASME-II objective class → training label (includes others + fear collapse)
OBJECTIVE_CLASS_TO_EMOTION = {
    1: "happiness",
    2: "surprise",
    3: "others",
    4: "disgust",
    5: "repression",
    6: "repression",
    7: "others",
}

# Target emotions for this project
TARGET_EMOTIONS = list(EMOTION_LABELS.keys())
NUM_EMOTIONS = len(TARGET_EMOTIONS)
