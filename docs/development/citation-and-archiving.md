# Citation and archiving

Use the repository's [CITATION.cff](https://github.com/ACCIDDA/op_system/blob/main/CITATION.cff)
to credit the software. The record describes the shared `op_system` and
`flepimop2-op_system` release and retains the authors/order from the two package
metadata files. Core is MIT-licensed; the adapter is GPLv3-or-later licensed. Record both
distribution versions when using the adapter, plus backend and flepimop2 versions
needed to reproduce the calculation.

For published work, cite the exact version used and its release URL. A citation
for version 0.7.0 can identify Joshua Macdonald, Carl Pearson, and Timothy Willard,
*op_system: RHS specification compiler and flepimop2 provider*, version 0.7.0,
released 2026-10-06, with the
[v0.7.0 release](https://github.com/ACCIDDA/op_system/releases/tag/v0.7.0).
For a development checkout, also record the commit and any local modifications.
Do not label unreleased behavior as part of 0.7.0 merely because the package
version has not yet been bumped. Use the CFF in the tag when one is available;
the first CFF here documents the already released 0.7.0 metadata.

Validate syntax with `just citation`, which uses the isolated, pinned
[cffconvert validator](https://github.com/citation-file-format/cffconvert)
for CFF 1.2.0. Maintainers review attribution rather than deriving authors from
commit counts. Update `version`, `date-released`, and the release artifact URL
alongside a shared version bump; citation checks enforce consistency with package
metadata and the dated changelog. No preferred paper citation is currently
recorded. Add `preferred-citation` after a paper's authors, title, venue, year,
and persistent identifier are verified; retain the software version citation
for reproducibility.

## Current identifier audit

On 2026-10-13, the repository, publication workflows, and public GitHub release
notes recorded no archive DOI, `.zenodo.json`, or Zenodo integration instructions.
A DataCite query for the exact repository URL returned no records; public web
searches found none. The Zenodo API request timed out. These observations do not
establish that no archive exists: a maintainer's connected Zenodo account and
unindexed records cannot be verified from this checkout. No DOI is invented in
`CITATION.cff`, and the current publication workflow creates GitHub releases and
package-index artifacts without itself performing an archival deposit.

## Software-paper release archive plan

The maintainer has selected the ACCIDDA Zenodo archive for this release;
[#273](https://github.com/ACCIDDA/op_system/issues/273) tracks the deposit and
verified DOI. Complete the JOSS readiness milestone before freezing the archive
release.

Before a paper release, the release maintainer must check existing archive
records and the repository's connected archival account. Reuse the established
record/version series if one exists; avoid minting a duplicate identifier for
the same release. If none exists, select one maintained archive such as Zenodo
and establish its GitHub release integration or perform a documented manual
deposit. See Zenodo's
[repository integration](https://help.zenodo.org/docs/github/enable-repository/)
and [release archive instructions](https://help.zenodo.org/docs/github/archive-software/github-upload/).

The paper release should then follow this sequence:

1. Confirm attribution, licenses, shared version, release date, changelog, and
   citation metadata. Run `just ci`, `just qualify`, and `just release-validate`.
2. Tag and publish the shared release through the existing workflow. Archive the
   exact tagged source for both distributions, their released artifacts, and
   any benchmark JSON/fixtures used for paper claims. Retain the tag/commit,
   checksums, environment versions, and links to versioned docs.
3. Verify that the archival record resolves and contains that exact release.
   Record its version DOI in the release notes and citation metadata; if the
   archive provides a concept DOI, identify it separately as the version series.
   Use the version DOI for a reproducible software-paper release citation.
4. Link the verified archive from the paper and release notes. Add a verified
   preferred paper citation when published. Check identifiers and CFF syntax
   again before announcing the release.

This plan records work for the release maintainer. Adding this guidance does not
create a deposit, enable an integration, or reserve an identifier. The shared
GitHub release URL remains the verified identifier until an archive is confirmed.
