# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the PQC IoT Retrofit Scanner project.

## ADR Format

We use the format described by Michael Nygard in [Documenting Architecture Decisions](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).

Each ADR should include:

- **Status**: Proposed, Accepted, Deprecated, or Superseded
- **Context**: The technical, political, social, and project local forces at play
- **Decision**: The change that we're proposing or have agreed to implement
- **Consequences**: What becomes easier or more difficult to do because of this change

## Index

- [ADR-0001: Use Dilithium2 as Default Signature Algorithm](0001-dilithium2-default.md)
- [ADR-0002: Memory-Constrained Optimization Strategy](0002-memory-optimization.md)
- [ADR-0003: Side-Channel Protection Implementation](0003-side-channel-protection.md)

## Creating New ADRs

1. Copy the template from `template.md`
2. Number sequentially (e.g., `0004-your-decision.md`)
3. Fill in all sections
4. Update this index
5. Submit for review via pull request