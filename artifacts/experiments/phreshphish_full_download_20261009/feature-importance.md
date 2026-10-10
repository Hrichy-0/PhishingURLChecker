# Saved model inspection

Model: Shared URL lexical features + native XGBoost Booster
Version: 0.1.0-native-xgboost-phreshphish
Created: 2026-10-09T21:18:51.375724+00:00
Threshold: 0.4718417227268219

These are saved global gain importance shares, normalized to 100%.
They are not per-URL contributions, detection rates, or percentiles.
Correlated features can share importance; gain does not show direction.
This reads metadata only and does not verify the model binary matches it.

| Rank | URL feature | Gain importance share |
|---:|---|---:|
| 1 | uses_https | 31.24% |
| 2 | has_ip_hostname | 11.25% |
| 3 | suspicious_word_count | 7.70% |
| 4 | hostname_length | 6.93% |
| 5 | subdomain_count | 5.77% |
| 6 | tld_length | 5.15% |
| 7 | has_at_symbol | 4.97% |
| 8 | hyphen_count | 4.16% |
| 9 | longest_token_length | 3.62% |
| 10 | path_length | 2.75% |
| 11 | dot_count | 2.73% |
| 12 | character_entropy | 2.70% |
| 13 | digit_ratio | 2.64% |
| 14 | digit_count | 1.67% |
| 15 | url_length | 1.46% |
| 16 | special_character_count | 1.27% |
| 17 | query_length | 1.09% |
| 18 | underscore_count | 0.90% |
| 19 | query_parameter_count | 0.77% |
| 20 | percent_encoding_count | 0.62% |
| 21 | has_explicit_port | 0.60% |

Recorded split sizes: {"test": 98070, "train": 856536, "validation": 114003}

## Recorded test metrics

- accuracy: 90.61%
- precision: 91.16%
- recall: 87.11%
- f1: 89.09%
- roc_auc: 95.91%
- false_positive_rate: 6.64%

Recorded metrics have not been rerun by this command.
