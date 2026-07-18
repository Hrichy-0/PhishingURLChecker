# PhishGuard model card

## Intended use

PhishGuard estimates phishing risk from an HTTP or HTTPS URL string. It is
designed for education, portfolio demonstration, and an additional browser
warning signal. It does not retrieve or execute content from the URL.

## Data

The model uses the PhiUSIIL Phishing URL Dataset. After removing 425 duplicate
URLs, 235,370 examples remain. Registered domains are kept entirely within one
of the following splits:

| Split | Rows | Percentage |
|---|---:|---:|
| Train | 188,753 | 80.19% |
| Validation | 23,300 | 9.90% |
| Test | 23,317 | 9.91% |

## Model

The saved scikit-learn pipeline contains:

1. A custom transformer producing 22 lexical URL features.
2. Standard scaling fitted only on training data.
3. Class-balanced logistic regression.

The operating threshold is chosen on validation data by maximizing F1 while
meeting a minimum 90% phishing-recall target. The test set is not involved in
training or threshold selection.

## Test results

- Accuracy: 0.9441
- Precision: 0.9576
- Recall: 0.9085
- F1: 0.9324
- ROC-AUC: 0.9849
- Average precision: 0.9859
- True negatives: 13,031
- False positives: 398
- False negatives: 905
- True positives: 8,983

## Known limitations

- Dataset sources can encode collection bias and may not represent future
  phishing campaigns.
- URLs and labels age. Domains can change ownership or behavior.
- The detector cannot see page content, redirects, certificates, DNS, domain
  age, JavaScript behavior, or brand impersonation visible only after loading.
- Attackers can deliberately choose benign-looking URLs.
- The validation-selected threshold represents one security tradeoff and may
  need adjustment for a real organization.
- False negatives and false positives remain possible. The UI must not call a
  low-risk result "safe."
- The score is not calibrated for real-world use. A displayed value near 100%
  is a model output, not a claim of 100% certainty.

## Important validation lesson

An early version achieved over 99% offline accuracy but incorrectly classified
common benign root URLs because browsers add a trailing `/` that much of the
training data omitted. Canonicalizing equivalent root URLs reduced the metric
to a more realistic level and fixed that training-serving skew. This regression
is covered by an automated feature test.

A later sanity check found a second, more important shortcut: the official
Apple Germany URL `https://www.apple.com/de/` received an extremely high
phishing score. Contribution inspection showed that `slash_count` dominated
the logistic-regression decision. In this dataset, legitimate examples are
disproportionately root URLs while phishing examples more often contain paths.
The model therefore learned a data-collection pattern that does not generalize
to normal deep links.

This is a confirmed false positive, not evidence that Apple's page is unsafe.
Domain-disjoint splitting prevents domain leakage, but it cannot remove every
source-collection bias. Version 1 should remain an educational prototype while
version 2 adds an independent sanity set, reviews path-related features, and
uses more representative benign deep-link data.

## Responsible presentation

- Describe the numeric output as a **model risk score**, not certainty.
- Never label a low score as "safe."
- Do not fix individual false positives with a brand whitelist.
- Do not tune the threshold against the same sanity examples used to report
  generalization.
- Keep browser/reputation protections enabled; this model is only one signal.
