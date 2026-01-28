"""
Shared constants for micro-expression recognition system.
Ensures consistent emotion label mapping across all modules.
"""

# Canonical emotion label mapping - DO NOT CHANGE ORDER
EMOTION_LABELS = {
    'happiness': 0,
    'surprise': 1,
    'disgust': 2,
    'repression': 3
}

# Reverse mapping for predictions
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}

# Display names (capitalized for plots/reports)
EMOTION_DISPLAY_NAMES = {
    'happiness': 'Happiness',
    'surprise': 'Surprise', 
    'disgust': 'Disgust',
    'repression': 'Repression'
}

# Ordered list for consistent plotting (matches numeric labels)
EMOTION_ORDER = [EMOTION_LABELS[emotion] for emotion in ['happiness', 'surprise', 'disgust', 'repression']]
EMOTION_DISPLAY_ORDER = [EMOTION_DISPLAY_NAMES[emotion] for emotion in ['happiness', 'surprise', 'disgust', 'repression']]

# CASME-II objective class to emotion mapping
OBJECTIVE_CLASS_TO_EMOTION = {
    1: 'happiness',   # happiness
    2: 'surprise',    # surprise  
    3: 'others',      # others (excluded)
    4: 'disgust',     # disgust
    5: 'repression',  # repression
    6: 'fear',        # fear (excluded)
    7: 'others'       # others (excluded)
}

# Target emotions for this project
TARGET_EMOTIONS = list(EMOTION_LABELS.keys())
NUM_EMOTIONS = len(TARGET_EMOTIONS)
