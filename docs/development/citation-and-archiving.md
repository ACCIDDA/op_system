# Citation and archiving

Use the repository's
[`CITATION.cff`](https://github.com/ACCIDDA/op_system/blob/main/CITATION.cff)
to credit the MIT-licensed `op_system` core. The provider is a separately
distributed GPLv3-or-later package with an independent release lifecycle. When
using it, record both distribution versions plus the backend and Flepimop2
versions required to reproduce the calculation.

For published work, cite the exact core version and verified release URL. The
current CFF identifies Joshua Macdonald, Carl Pearson and Timothy Willard,
*op_system: Model specification compiler*, version 0.10.0, released 2026-10-23,
with the
[`v0.10.0` release](https://github.com/ACCIDDA/op_system/releases/tag/v0.10.0).
For a development checkout, also record the commit and local modifications.
Unreleased behavior is not retroactively part of 0.10.0; use the exact tag's CFF,
source and changelog.

Run `just citation` to validate CFF 1.2.0 syntax and alignment with the core
package version, authors, MIT grant, dated changelog and release URL. Maintainers
review attribution rather than deriving authors from commit counts. Update the
version, date and artifact URL with a core release. No preferred paper citation
is currently recorded; add one only after its bibliographic record and
persistent identifier are verified.

## Current identifier audit

On 2026-10-23, the repository had an active Zenodo release webhook, but no
matching public software record or DOI had been verified. Do not reproduce or
publish the credential-bearing webhook URL. A configured hook is not evidence
that a release was deposited or accepted by a community. Do not infer or invent
a DOI before its record resolves.

## Software-paper release archive plan

Use the existing ACCIDDA Zenodo community for a verified core release record.
Community submission and curator acceptance are manual account actions;
community membership does not itself create or own a deposit.

Before a paper release, the maintainer must check existing archive records and
the connected archival account. Reuse an established record/version series if
one exists and avoid duplicate records for the same release. If none exists,
select one maintained archive such as Zenodo and establish its GitHub release
integration or perform a documented manual deposit. See Zenodo's
[repository integration](https://help.zenodo.org/docs/github/enable-repository/)
and
[release archive instructions](https://help.zenodo.org/docs/github/archive-software/github-upload/).

Use this sequence:

1. Confirm core attribution, MIT license, version, release date, changelog and
   citation metadata. Run `just ci`, `just qualify` and
   `just release-validate-core`.
2. Verify the existing release integration without exposing its credential. Do
   not delete and recreate a hook merely to trigger retroactive ingestion.
3. Publish the reviewed core release through the core-only workflow. Retain the
   tag and commit, artifact checksums, environment versions, versioned docs and
   benchmark evidence used for paper claims.
4. Verify that the archival record resolves and contains the exact tagged core
   release. Record its version DOI in release notes and in citation metadata on
   `main` for the next release; do not rewrite an immutable tagged CFF.
5. Submit the published record to the ACCIDDA community and wait for curator
   acceptance.
6. Link the verified archive from the paper and release notes. Add a verified
   preferred paper citation when published.

These instructions do not create a deposit, enable an integration or reserve an
identifier. The GitHub release URL remains the verified core identifier until
an archive record is confirmed.
