# Three-Species Tree-Sum Orientation Audit

Orientation matches paper list: `True`.

## Edge Labeling

| edge label | source | target |
| --- | --- | --- |
| K1 | 2 | 0 |
| K2 | 1 | 2 |
| K3 | 1 | 0 |
| K4 | 2 | 1 |
| K5 | 0 | 1 |
| K6 | 0 | 2 |

## Rooted Tree Edge Sets

### Root A

- Expected: `[['K1', 'K2'], ['K3', 'K4'], ['K1', 'K3']]`
- Actual: `[['K1', 'K2'], ['K1', 'K3'], ['K3', 'K4']]`
- Pass: `True`

### Root B

- Expected: `[['K1', 'K5'], ['K4', 'K6'], ['K4', 'K5']]`
- Actual: `[['K1', 'K5'], ['K4', 'K5'], ['K4', 'K6']]`
- Pass: `True`

### Root C

- Expected: `[['K3', 'K6'], ['K2', 'K5'], ['K2', 'K6']]`
- Actual: `[['K2', 'K5'], ['K2', 'K6'], ['K3', 'K6']]`
- Pass: `True`
