ENABLE_ALL_FEATURES = False

FEATURE_NAMES = ['latency', 'http_status']

FAULT_TYPES = {'cpu', 'delay', 'disk', 'loss', 'mem', 'socket'}

INVOLVED_SERVICES = [
    'adservice',
    'cartservice',
    'checkoutservice',
    'currencyservice',
    'emailservice',
    'frontendservice',
    'paymentservice',
    'productcatalogservice',
    'recommendationservice',
    'shippingservice',
]

SERVICE2IDX = {service: idx for idx, service in enumerate(INVOLVED_SERVICES)}


# Off-switch defaults for paper-accurate Stage 1 (F1b/F2a/F2b/per-op).
# Every flag's neutral value here reproduces the legacy baseline bit-for-bit.
STAGE1_DEFAULTS = {
    'admit_self_spans':    False,
    'admit_root_spans':    False,
    'feature_selector':    'stderr',
    'fs_delta':            0.1,
    'fs_floor':            0.0,
    'baseline_window':     'global',
    'last_slot_seconds':   300,
    'last_period_seconds': 86400,
}
