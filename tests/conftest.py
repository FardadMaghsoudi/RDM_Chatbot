"""Shared test helpers."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip the slow end-to-end tests unless they are explicitly selected with `-m e2e`."""
    if "e2e" in (config.getoption("markexpr") or ""):
        return
    skip_e2e = pytest.mark.skip(reason="end-to-end test: run with `pytest -m e2e`")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


def write_text_pdf(path, lines):
    """Write a minimal one-page PDF showing `lines` as text in the middle of the page."""
    text_ops = " ".join(
        "({}) Tj T*".format(line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)"))
        for line in lines
    )
    stream = f"BT /F1 12 Tf 14 TL 72 500 Td {text_ops} ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_offset)

    path.write_bytes(bytes(pdf))
    return path


@pytest.fixture
def make_pdf():
    return write_text_pdf
