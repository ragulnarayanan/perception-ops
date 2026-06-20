from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import Gauge

# Total API requests
REQUEST_COUNT = Counter(
    "requests_total",
    "Total prediction requests"
)

# Inference latency
INFERENCE_LATENCY = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=(10, 50, 100, 200, 500, 1000, 2000, 5000)
)

# Total detections
DETECTIONS_TOTAL = Counter(
    "detections_total",
    "Total detections produced"
)

# Average confidence
AVG_CONFIDENCE = Gauge(
    "average_confidence",
    "Average confidence score"
)