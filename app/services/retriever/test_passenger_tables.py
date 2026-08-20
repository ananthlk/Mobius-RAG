from types import SimpleNamespace

from app.services.retriever.passenger_tables import (
    extract_breadcrumbs,
    is_numeric_only,
    resolve_passenger_tables,
)


def _citation(chunk_id: str, text: str, document_id: str | None = None, page_number: int | None = None):
    return SimpleNamespace(chunk_id=chunk_id, text=text, document_id=document_id, page_number=page_number)


class TestExtractBreadcrumbs:
    def test_single_breadcrumb_matches_sourcing_shape_exactly(self):
        text = (
            "Reimbursement rates by facility are listed below. "
            "[Table: Model 10A Assumptions · →document_tables:3f9c1a2b-1111-4a2b-8c3d-abcdef123456] "
            "See table for full detail."
        )
        crumbs = extract_breadcrumbs(text, "chunk-1")
        assert len(crumbs) == 1
        assert crumbs[0].caption == "Model 10A Assumptions"
        assert crumbs[0].table_id == "3f9c1a2b-1111-4a2b-8c3d-abcdef123456"
        assert crumbs[0].source_chunk_id == "chunk-1"

    def test_multiple_breadcrumbs_in_one_chunk(self):
        text = (
            "[Table: North Region Rates · →document_tables:aaaaaaaa-1111-1111-1111-111111111111] "
            "and also [Table: South Region Rates · →document_tables:bbbbbbbb-2222-2222-2222-222222222222]"
        )
        crumbs = extract_breadcrumbs(text, "chunk-2")
        assert [c.table_id for c in crumbs] == [
            "aaaaaaaa-1111-1111-1111-111111111111",
            "bbbbbbbb-2222-2222-2222-222222222222",
        ]

    def test_no_breadcrumb_returns_empty_not_none(self):
        assert extract_breadcrumbs("Ordinary prose chunk, no tables here.", "chunk-3") == []

    def test_empty_text_does_not_raise(self):
        assert extract_breadcrumbs("", "chunk-4") == []
        assert extract_breadcrumbs(None, "chunk-4") == []

    def test_malformed_breadcrumb_ignored(self):
        # missing the arrow / wrong separator -- Sourcing's shape not honored, so no match
        text = "[Table: Broken · document_tables:cccccccc-3333-3333-3333-333333333333]"
        assert extract_breadcrumbs(text, "chunk-5") == []


class TestResolvePassengerTables:
    def test_single_citation_resolves_to_one_table(self):
        table_id = "aaaaaaaa-1111-1111-1111-111111111111"
        citations = [
            _citation(
                "chunk-1",
                f"Rate detail. [Table: North Region Rates · →document_tables:{table_id}]",
            )
        ]
        fetched = {table_id: {"grid": [["code", "rate"], ["059404", "$266.28"]]}}

        result = resolve_passenger_tables(citations, fetch_table=lambda tid: fetched.get(tid))

        assert len(result) == 1
        assert result[0].table_id == table_id
        assert result[0].caption == "North Region Rates"
        assert result[0].cited_by_chunk_ids == ("chunk-1",)
        assert result[0].payload == fetched[table_id]

    def test_same_table_cited_by_two_chunks_dedups_to_one_entry(self):
        table_id = "aaaaaaaa-1111-1111-1111-111111111111"
        breadcrumb = f"[Table: North Region Rates · →document_tables:{table_id}]"
        citations = [
            _citation("chunk-1", f"First mention. {breadcrumb}"),
            _citation("chunk-2", f"Second mention, same table. {breadcrumb}"),
        ]
        fetch_calls = []

        def fetch_table(tid):
            fetch_calls.append(tid)
            return {"grid": [["x"]]}

        result = resolve_passenger_tables(citations, fetch_table=fetch_table)

        assert len(result) == 1
        assert result[0].cited_by_chunk_ids == ("chunk-1", "chunk-2")
        # fetched once, not once per citing chunk
        assert fetch_calls == [table_id]

    def test_two_distinct_tables_both_attached_in_first_seen_order(self):
        id_a = "aaaaaaaa-1111-1111-1111-111111111111"
        id_b = "bbbbbbbb-2222-2222-2222-222222222222"
        citations = [
            _citation("chunk-1", f"[Table: B first in text · →document_tables:{id_b}]"),
            _citation("chunk-2", f"[Table: A second in text · →document_tables:{id_a}]"),
        ]
        fetched = {id_a: {"grid": ["a"]}, id_b: {"grid": ["b"]}}

        result = resolve_passenger_tables(citations, fetch_table=lambda tid: fetched.get(tid))

        assert [t.table_id for t in result] == [id_b, id_a]

    def test_unresolvable_table_id_dropped_silently_fail_open(self):
        table_id = "aaaaaaaa-1111-1111-1111-111111111111"
        citations = [
            _citation("chunk-1", f"[Table: Ghost · →document_tables:{table_id}]")
        ]
        result = resolve_passenger_tables(citations, fetch_table=lambda tid: None)
        assert result == []

    def test_fetch_table_raising_does_not_propagate(self):
        table_id = "aaaaaaaa-1111-1111-1111-111111111111"
        citations = [
            _citation("chunk-1", f"[Table: Boom · →document_tables:{table_id}]")
        ]

        def fetch_table(tid):
            raise RuntimeError("document_tables unreachable")

        result = resolve_passenger_tables(citations, fetch_table=fetch_table)
        assert result == []

    def test_citation_missing_chunk_id_or_text_skipped_not_raised(self):
        citations = [
            SimpleNamespace(chunk_id=None, text="[Table: X · →document_tables:aaaaaaaa-1111-1111-1111-111111111111]"),
            SimpleNamespace(chunk_id="chunk-2", text=None),
        ]
        result = resolve_passenger_tables(citations, fetch_table=lambda tid: {"grid": []})
        assert result == []

    def test_no_citations_returns_empty(self):
        assert resolve_passenger_tables([], fetch_table=lambda tid: {"grid": []}) == []

    def test_citation_with_no_breadcrumb_contributes_nothing(self):
        citations = [_citation("chunk-1", "Ordinary prose, no table reference at all.")]
        result = resolve_passenger_tables(citations, fetch_table=lambda tid: {"grid": []})
        assert result == []

    def test_no_fallback_fn_means_no_breadcrumb_contributes_nothing_even_with_page(self):
        # fetch_tables_for_page omitted entirely -- path 2 must not activate implicitly
        citations = [_citation("chunk-1", "059404  $266.28", document_id="doc-1", page_number=3)]
        result = resolve_passenger_tables(citations, fetch_table=lambda tid: None)
        assert result == []


class TestIsNumericOnly:
    def test_pure_numeric_row_is_numeric_only(self):
        assert is_numeric_only("059404 $266.28 1,068,571")

    def test_prose_is_not_numeric_only(self):
        assert not is_numeric_only("Patient must be 18 years of age or older.")

    def test_blank_or_none_is_not_numeric_only(self):
        assert not is_numeric_only("")
        assert not is_numeric_only("   ")
        assert not is_numeric_only(None)

    def test_single_letter_disqualifies(self):
        assert not is_numeric_only("N/A")


class TestPageProximityFallback:
    def test_no_breadcrumb_numeric_chunk_falls_back_to_page_join(self):
        citations = [_citation("chunk-1", "059404  $266.28", document_id="doc-1", page_number=3)]
        page_tables = {("doc-1", 3): [{"id": "tbl-a", "caption": "North Region Rates", "grid": []}]}

        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=lambda doc_id, page: page_tables.get((doc_id, page), []),
        )

        assert len(result) == 1
        assert result[0].table_id == "tbl-a"
        assert result[0].matched_via == "page_proximity"
        assert result[0].cited_by_chunk_ids == ("chunk-1",)

    def test_breadcrumb_present_skips_fallback_entirely(self):
        table_id = "aaaaaaaa-1111-1111-1111-111111111111"
        citations = [
            _citation(
                "chunk-1",
                f"[Table: Real Table · →document_tables:{table_id}]",
                document_id="doc-1",
                page_number=3,
            )
        ]
        fallback_calls = []

        def fetch_tables_for_page(doc_id, page):
            fallback_calls.append((doc_id, page))
            return [{"id": "tbl-wrong", "caption": "Should not be reached"}]

        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: {"grid": []},
            fetch_tables_for_page=fetch_tables_for_page,
        )

        assert [t.table_id for t in result] == [table_id]
        assert fallback_calls == []  # fallback never invoked when a breadcrumb resolved

    def test_page_with_two_tables_attaches_both(self):
        citations = [_citation("chunk-1", "12  34", document_id="doc-1", page_number=36)]
        page_tables = {
            ("doc-1", 36): [
                {"id": "tbl-a", "caption": "Table A"},
                {"id": "tbl-b", "caption": "Table B"},
            ]
        }
        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=lambda doc_id, page: page_tables.get((doc_id, page), []),
        )
        assert {t.table_id for t in result} == {"tbl-a", "tbl-b"}

    def test_cross_path_dedup_many_numeric_cells_one_page_one_table_entry(self):
        # The core hazard the go-ahead calls out: many numeric-only chunks
        # off ONE page must collapse to ONE PassengerTable, not one per chunk.
        citations = [
            _citation(f"chunk-{i}", "059404", document_id="doc-1", page_number=3)
            for i in range(20)
        ]
        page_tables = {("doc-1", 3): [{"id": "tbl-a", "caption": "North Region Rates"}]}
        fetch_calls = []

        def fetch_tables_for_page(doc_id, page):
            fetch_calls.append((doc_id, page))
            return page_tables.get((doc_id, page), [])

        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=fetch_tables_for_page,
        )

        assert len(result) == 1
        assert result[0].table_id == "tbl-a"
        assert len(result[0].cited_by_chunk_ids) == 20
        assert len(fetch_calls) == 20  # loader-level batching is the caller's job, not this function's

    def test_dedup_holds_when_same_table_reached_via_both_breadcrumb_and_fallback(self):
        table_id = "aaaaaaaa-1111-1111-1111-111111111111"
        citations = [
            _citation(
                "chunk-1",
                f"[Table: North Region Rates · →document_tables:{table_id}]",
                document_id="doc-1",
                page_number=3,
            ),
            _citation("chunk-2", "059404  $266.28", document_id="doc-1", page_number=3),
        ]
        page_tables = {("doc-1", 3): [{"id": table_id, "caption": "North Region Rates"}]}

        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: {"grid": []},
            fetch_tables_for_page=lambda doc_id, page: page_tables.get((doc_id, page), []),
        )

        assert len(result) == 1
        assert result[0].matched_via == "breadcrumb"  # first path to see it wins the label
        assert result[0].cited_by_chunk_ids == ("chunk-1", "chunk-2")

    def test_no_document_id_or_page_number_skips_fallback_fail_open(self):
        citations = [_citation("chunk-1", "12 34 56", document_id=None, page_number=None)]
        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=lambda doc_id, page: [{"id": "tbl-a"}],
        )
        assert result == []

    def test_fallback_row_missing_id_skipped_not_raised(self):
        citations = [_citation("chunk-1", "12 34", document_id="doc-1", page_number=3)]
        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=lambda doc_id, page: [{"caption": "No id field"}],
        )
        assert result == []

    def test_fetch_tables_for_page_raising_does_not_propagate(self):
        citations = [_citation("chunk-1", "12 34", document_id="doc-1", page_number=3)]

        def fetch_tables_for_page(doc_id, page):
            raise RuntimeError("document_tables unreachable")

        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=fetch_tables_for_page,
        )
        assert result == []

    def test_prose_chunk_with_no_breadcrumb_still_triggers_fallback_not_gated_on_numeric(self):
        # Ananth's wording: fallback fires on "no breadcrumb", numeric-only is
        # the strongest-signal case, not an additional required condition.
        citations = [
            _citation(
                "chunk-1",
                "This page also discusses eligibility in prose form.",
                document_id="doc-1",
                page_number=3,
            )
        ]
        page_tables = {("doc-1", 3): [{"id": "tbl-a", "caption": "North Region Rates"}]}
        result = resolve_passenger_tables(
            citations,
            fetch_table=lambda tid: None,
            fetch_tables_for_page=lambda doc_id, page: page_tables.get((doc_id, page), []),
        )
        assert len(result) == 1
        assert result[0].table_id == "tbl-a"
