# Saved model inspection

Model: URLFeatureExtractor + XGBClassifier
Version: 0.1.0-xgboost
Created: 2026-10-03T10:57:32.545664+00:00
Threshold: 0.4711279273033142

These are saved global gain importance shares, normalized to 100%.
They are not per-URL contributions, detection rates, or percentiles.
Correlated features can share importance; gain does not show direction.
This reads metadata only and does not verify the model binary matches it.

| Rank | URL feature | Gain importance share |
|---:|---|---:|
| 1 | uses_https | 45.83% |
| 2 | has_ip_hostname | 11.54% |
| 3 | suspicious_word_count | 10.30% |
| 4 | has_at_symbol | 5.86% |
| 5 | hostname_length | 4.13% |
| 6 | hyphen_count | 3.25% |
| 7 | subdomain_count | 2.99% |
| 8 | longest_token_length | 2.57% |
| 9 | dot_count | 2.14% |
| 10 | path_length | 1.74% |
| 11 | digit_ratio | 1.29% |
| 12 | digit_count | 1.27% |
| 13 | special_character_count | 1.17% |
| 14 | tld_length | 1.06% |
| 15 | url_length | 1.06% |
| 16 | query_length | 0.77% |
| 17 | has_explicit_port | 0.71% |
| 18 | character_entropy | 0.69% |
| 19 | underscore_count | 0.57% |
| 20 | query_parameter_count | 0.55% |
| 21 | percent_encoding_count | 0.51% |

Recorded split sizes: {"test": 60551, "train": 457544, "validation": 56621}

## Recorded test metrics

- accuracy: 93.19%
- precision: 93.48%
- recall: 92.74%
- f1: 93.11%
- roc_auc: 98.33%
- false_positive_rate: 6.37%

Recorded metrics have not been rerun by this command.
