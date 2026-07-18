# Security policy

PhishingURLChecker is an educational prototype and is not a production security
control. Do not use its score as the sole basis for visiting, blocking, or
trusting a URL.

## Reporting a security issue

Please do not publish exploitable API or extension vulnerabilities in a public
issue. Use the repository owner's private contact or GitHub private
vulnerability reporting once it is enabled.

False positives, false negatives, dataset-bias findings, and non-sensitive
model-quality problems may be reported as normal issues with a reproducible,
non-malicious example.

## Safe testing

Use reserved domains such as `example.com` and `.test`, or documentation IP
ranges such as `192.0.2.0/24`. Do not navigate to live malicious URLs merely to
test this project. The API itself does not fetch submitted URLs.
