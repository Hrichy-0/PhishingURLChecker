# RPA benign site list

`sites.csv` is a fixed list of **197 real legitimate web pages** for the UI-automation
(Playwright) test of the Chrome extension. It is the benign side only: it
measures false-positive rate per category and says nothing about detection.
Freeze it before any scoring; do not add, drop or replace rows after seeing
extension results.

Built 23 October 2026. No model, API, extension or cutoff was changed, and no
phishing URL was visited.

## Columns

| Column | Meaning |
| --- | --- |
| `url` | Exact HTTP(S) address that returned HTTP 200 on 2026-10-23 |
| `category` | `sector/page_type`. Sectors: bank, email, government, education, shopping, news, saas (software, telecom, media and other companies). Page types: homepage, login, docs, deeplink, long_query |
| `source` | Tranco list `467YX` rank, plus Wikidata as the legitimacy evidence |
| `rank_band` | `top1k`, `1k-10k`, `10k-100k`, `100k-1M` |
| `added_date` | 2026-10-23 |
| `verification_note` | Wikidata item, the official-website (P856) value, any redirect, how a deep page was found, and the page's `html lang` |

## How it was built

1. **Exclusion.** `history_domains` from `scripts/evaluate_frozen_url.py` was
   re-run over the current local `data/` and `artifacts/` (84 CSV/Parquet files,
   69 with URL columns). The result was **952,870 previously observed registered
   domains**: the 19 October set of 952,770 plus 100 new ones from the frozen
   pilot's own files. The frozen pilot's 200 candidate domains were also
   excluded, so the two cohorts stay disjoint. Grouping uses the same
   private-PSL `hostname_group` helper.
2. **Ranking.** These counts are domains in Tranco `467YX` (the list saved by
   the frozen benchmark preparation) that are *not* excluded: 357 in the top 1k,
   3,975 in 1k–10k, 55,924 in 10k–100k and 797,798 in 100k–1M. Most of the
   remaining top-1k domains are CDN or DNS infrastructure.
3. **Legitimacy evidence.** Popularity is not proof of legitimacy. A domain was
   kept only if a Wikidata item lists a URL on that registered domain as its
   official website (P856). Candidates came from Wikidata class queries (banks,
   universities, government agencies/ministries, e-commerce, software, news
   media), an exact-URL reverse lookup of every unseen top-10k domain, and a
   hand-checked list of 9 email providers. Most large webmail providers are
   already in the development data. The candidate pool held 10,129 domains.
4. **Selection.** Within each rank band, candidates are ordered by
   `sha256("rpa-sites-v1|" + domain)`, never by a model score, and taken
   round-robin by sector. Each domain contributes **one URL only**.
5. **Page check.** The official P856 page was fetched once. A candidate was
   kept only if it returned HTTP 200 HTML on the same registered domain and the
   final URL path did not contain `error`. **72 candidates failed this check and
   were skipped**: 6 top-1k, 29 in 1k–10k, 16 in 10k–100k and 21 in 100k–1M. No
   scheme or path was invented. Where the official page redirected, the final
   URL is used and the original P856 value is kept in the note.
6. **Page types.** Login, docs, deep-link and long-query pages were taken only
   from links on that official homepage (same registered domain). Each was
   fetched and needed HTTP 200. Bank and email sites try a login link first;
   other sectors rotate through the page types. If no matching link existed,
   the homepage is used. Links whose query looked like it carried a token,
   session, key or e-mail were skipped.
7. **Manual review.** After the crawl, the following were removed or corrected:
   - Removed by judgement before selection: `betika.com` (gambling),
     `musicdel.ir` (music downloads) and `humoruniv.com` (adult humour).
   - Removed after review: an adult image board, a gambling-promotion blog, and
     one URL with a Google linker tracking identifier.
   - Relabelled one certificate authority from government to saas.
   - Excluded `example.com` as a reserved documentation domain, not an
     organisation's site.

## Composition

- Rank bands: top1k 14, 1k–10k 51, 10k–100k 66, 100k–1M 66. The top-1k pool
  ran out before its target of 20.
- Sectors: saas 47, news 37, shopping 34, bank 27, government 23, education 20,
  email 9.
- Page types: homepage 142, deeplink 22, login 17, docs 8, long_query 8.
- Language: 91 pages declare a non-English `html lang`; 29 declare none.
  Five URLs are plain `http://`, as their official sites serve them.

## Limitations

- Wikidata's P856 is community-edited corroboration, not a certificate of
  safety. It refers to the organisation's domain, not to every page on it.
- Page types were assigned automatically from link text in URLs. A few are
  approximate: for example, "login" can be a login-help page, and "docs" can be
  a FAQ.
- Sites change. Re-runs may hit redirects, geo-variants (for example the MTS
  regional redirect), consent walls or outages. Record those as run failures,
  not as model results.
- Domain exclusion covers only locally available history. It does not prove
  independence from campaigns or templates.
- 197 domains give wide uncertainty on low FPRs. Report per-category counts
  with intervals, not just percentages.
