# Project journey

This document records what was built, why each decision was made, and what was
learned. It is written as a learning log rather than only a finished-product
description.

## 1. Goal and architecture

The goal was to build a complete machine-learning product, not only a notebook:

1. obtain a phishing URL dataset from Kaggle;
2. inspect, clean, and split it reproducibly;
3. train and evaluate a URL-only model;
4. package the trained pipeline;
5. expose it through a FastAPI service;
6. call the service from a Chrome extension;
7. test every important boundary.

The chosen stack is Python, pandas, scikit-learn, joblib, FastAPI, Pydantic,
pytest, Ruff, and a Manifest V3 extension written in plain HTML/CSS/JavaScript.

## 2. Dataset inspection

The PhiUSIIL Phishing URL Dataset contained 235,795 rows and 56 columns. It had
URL text, domain/page-derived attributes, and a binary `label`. There were no
missing values in the original table.

The first lesson was that “no missing values” does not mean “ready for ML.” We
still needed to understand the label meaning, remove duplicate URLs, select
features that are available when the extension runs, and prevent information
leakage.

For this URL-only first version, the project retains the URL string and target
instead of depending on page-content fields that the live API cannot safely or
quickly collect.

## 3. Cleaning

The cleaning stage:

- validates required columns;
- removes missing or blank URLs and labels;
- removes duplicate URLs;
- normalizes the project target to `1 = phishing`, `0 = legitimate`;
- saves only `url` and `is_phishing`.

Cleaning reduced 235,795 rows to 235,370 by removing 425 duplicate URLs. The
cleaned data contains 100,520 phishing rows (42.71%) and 134,850 legitimate
rows (57.29%). Automated tests protect these rules.

## 4. Leakage-aware splitting

A random row split can put URLs from the same registered domain into both
training and testing. The model could then appear to generalize while partially
remembering a domain.

The project instead groups by registered domain and creates disjoint splits:

| Split | Rows | Share |
|---|---:|---:|
| Train | 188,753 | 80.19% |
| Validation | 23,300 | 9.90% |
| Test | 23,317 | 9.91% |

Training fits model parameters. Validation selects the decision threshold.
Testing is touched only for the final offline estimate.

## 5. Feature engineering and model

The baseline computes 22 explainable lexical features without visiting the
website. Examples include URL/hostname/path length, digit and punctuation
counts, subdomain count, IP-host detection, HTTPS, suspicious words, and
character entropy.

These features feed a scikit-learn pipeline containing standard scaling and
class-balanced logistic regression. Saving the entire pipeline is important:
the API cannot accidentally apply different feature transformations from the
ones used in training.

A dummy classifier establishes a baseline. The real decision threshold is
chosen on validation data by maximizing F1 while requiring at least 90% recall
there.

## 6. Offline results

On the domain-disjoint test split, version 1 produced:

| Metric | Result |
|---|---:|
| Accuracy | 94.41% |
| Precision | 95.76% |
| Recall | 90.85% |
| F1 | 93.24% |
| ROC-AUC | 98.49% |
| False positives | 398 |
| False negatives | 905 |

Precision answers: “Of the URLs flagged as phishing, how many were phishing?”
Recall answers: “Of all phishing URLs, how many did the model catch?” A security
project should report both because missing an attack and blocking a legitimate
site have different costs.

## 7. API and extension

The model is packaged with joblib and its threshold/metrics are stored as JSON
metadata. FastAPI exposes health and prediction endpoints with Pydantic input
validation. It analyzes the URL text only; it never opens the submitted site.

The Chrome extension reads the active HTTP(S) tab, sends its URL to the local
API, and displays the result and lexical reasons. Keeping the extension thin
makes the model logic testable in Python and replaceable without rewriting the
browser interface.

## 8. The most valuable finding: offline success is not enough

Testing the official Apple Germany page exposed a severe false positive:
`https://www.apple.com/de/` received a score near 100%, while Apple's root URL
scored low. Feature-contribution analysis showed that slash count dominated the
decision.

The training collection contains many legitimate root URLs and many phishing
URLs with deeper paths. The model learned that incidental dataset pattern as a
shortcut. A domain-disjoint split prevents one kind of leakage but cannot make
the source distribution representative of the live web.

This is why external sanity tests matter. A high held-out score is evidence
about one dataset, not proof that a model is production-ready.

## 9. Current status and next iteration

Version 1 is a working end-to-end educational prototype. It should not be used
to decide whether a real site is safe.

Version 2 should:

1. rename the displayed value to “model risk score” and add an uncertain state;
2. create a separate, versioned sanity set of benign deep links and safe
   synthetic phishing examples;
3. remove or redesign shortcut-prone path features such as slash count;
4. retrain with more representative, time-separated data;
5. report both the Kaggle test result and the untouched external sanity result;
6. eventually combine lexical ML with reputable browser/domain intelligence.

The Apple URL should be used as a regression check, not copied into training or
added to a whitelist. That tests generalization instead of memorization.

## 10. Skills demonstrated

- reproducible data ingestion and cleaning;
- binary-label normalization and class-distribution analysis;
- group-aware train/validation/test design;
- explainable feature engineering;
- imbalanced classification and threshold selection;
- precision, recall, F1, ROC-AUC, and confusion-matrix interpretation;
- model packaging and API validation;
- Chrome extension integration;
- unit, integration, formatting, and lint checks;
- error analysis, shortcut-learning diagnosis, and responsible ML reporting.
