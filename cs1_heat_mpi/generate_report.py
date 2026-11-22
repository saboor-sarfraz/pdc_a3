#!/usr/bin/env python3
"""Build an editable DOCX report and its scaling graph using Python stdlib only."""

import csv
import pathlib
import struct
import statistics
import zlib
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY_PATH = RESULTS / "heat_summary.csv"
OUTPUT = ROOT.parent / "Assignment3_Report.docx"
EMU_PER_INCH = 914400


def load_rows():
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            "Missing results/heat_summary.csv. Run benchmark.py with --full first."
        )
    with SUMMARY_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("N", "steps", "ranks", "K"):
            row[key] = int(row[key])
        for key in (
            "median_seconds", "serial_baseline_seconds", "speedup_vs_serial",
            "efficiency", "median_comm_post_or_block_seconds",
            "median_comm_wait_seconds", "median_interior_seconds",
        ):
            row[key] = float(row[key])
    return rows


FONT = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000",) * 7,
}


def make_chart(strong_rows, path, x_label="MPI RANKS"):
    width, height = 760, 450
    pixels = bytearray([255] * (width * height * 3))

    def dot(x, y, color, size=1):
        for dy in range(-size + 1, size):
            for dx in range(-size + 1, size):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    offset = (py * width + px) * 3
                    pixels[offset:offset + 3] = bytes(color)

    def line(x0, y0, x1, y1, color, thickness=1):
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx + dy
        while True:
            dot(x0, y0, color, thickness)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def text(x, y, value, color=(30, 30, 30), scale=2):
        cursor = x
        for char in value.upper():
            bitmap = FONT.get(char, FONT[" "])
            for row, bits in enumerate(bitmap):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        dot(cursor + col * scale, y + row * scale, color, scale)
            cursor += 6 * scale

    x_left, x_right, y_top, y_bottom = 84, 720, 42, 368
    values = [row["speedup_vs_serial"] for row in strong_rows]
    ranks = [row["ranks"] for row in strong_rows]
    ymax = max(1.0, max(values), max(ranks)) * 1.15
    for tick in range(6):
        y = y_bottom - round((y_bottom - y_top) * tick / 5)
        line(x_left, y, x_right, y, (225, 230, 235))
        text(12, y - 6, f"{ymax * tick / 5:.1f}", scale=1)
    line(x_left, y_top, x_left, y_bottom, (40, 50, 60), 2)
    line(x_left, y_bottom, x_right, y_bottom, (40, 50, 60), 2)
    x_positions = [
        x_left + round((x_right - x_left) * (rank - min(ranks)) /
                       max(1, max(ranks) - min(ranks)))
        for rank in ranks
    ]
    for rank, x in zip(ranks, x_positions):
        text(x - 2, y_bottom + 12, str(rank), scale=1)
    text(310, 415, x_label, scale=2)
    text(292, 12, "SPEEDUP VS SERIAL", color=(30, 90, 150), scale=2)

    speedup_points = [
        (x, y_bottom - round(value / ymax * (y_bottom - y_top)))
        for x, value in zip(x_positions, values)
    ]
    ideal_points = [
        (x, y_bottom - round(rank / ymax * (y_bottom - y_top)))
        for x, rank in zip(x_positions, ranks)
    ]
    for points, color in ((ideal_points, (150, 150, 150)),
                          (speedup_points, (35, 105, 175))):
        for first, second in zip(points, points[1:]):
            line(*first, *second, color, 2)
        for x, y in points:
            dot(x, y, color, 4)
    text(520, 390, "MEASURED", color=(35, 105, 175), scale=1)
    text(640, 390, "IDEAL", color=(100, 100, 100), scale=1)
    png = _png(width, height, pixels)
    path.write_bytes(png)


def _png(width, height, pixels):
    def chunk(name, data):
        return (
            struct.pack(">I", len(data)) + name + data +
            struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        rows.extend(pixels[y * stride:(y + 1) * stride])
    return (
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(bytes(rows), 9)) +
        chunk(b"IEND", b"")
    )


def xml_text(text):
    return escape(str(text))


def paragraph(text="", style=None, bold=False, italic=False, page_break=False):
    if page_break:
        return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
    style_tag = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    if not text:
        return f"<w:p>{style_tag}</w:p>"
    props = ""
    if bold or italic:
        props = "<w:rPr>"
        if bold:
            props += "<w:b/>"
        if italic:
            props += "<w:i/>"
        props += "</w:rPr>"
    return (
        f"<w:p>{style_tag}<w:r>{props}"
        f'<w:t xml:space="preserve">{xml_text(text)}</w:t></w:r></w:p>'
    )


def table(rows, widths=None):
    if not rows:
        return ""
    columns = len(rows[0])
    if widths is None:
        widths = [1800] * columns
    xml = (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:left w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:right w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D5DDE5"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D5DDE5"/></w:tblBorders>'
        '</w:tblPr><w:tblGrid>'
    )
    xml += "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    xml += "</w:tblGrid>"
    for row_index, row in enumerate(rows):
        xml += "<w:tr>"
        for column, cell in enumerate(row):
            shade = '<w:shd w:fill="DCE6F1"/>' if row_index == 0 else ""
            bold = "<w:b/>" if row_index == 0 else ""
            xml += (
                f"<w:tc><w:tcPr>{shade}"
                f'<w:tcW w:w="{widths[column]}" w:type="dxa"/>'
                f"</w:tcPr><w:p><w:r><w:rPr>{bold}</w:rPr>"
                f'<w:t xml:space="preserve">{xml_text(cell)}</w:t>'
                "</w:r></w:p></w:tc>"
            )
        xml += "</w:tr>"
    return xml + "</w:tbl>"


def image_paragraph(relationship_id, width_px, height_px, drawing_id=1):
    width = int(6.6 * EMU_PER_INCH)
    height = int(width * height_px / width_px)
    return f"""
    <w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
    <wp:inline distT="0" distB="0" distL="0" distR="0">
    <wp:extent cx="{width}" cy="{height}"/><wp:docPr id="{drawing_id}" name="Strong scaling"/>
    <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
    <pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="strong_scaling.png"/>
    <pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{relationship_id}"/>
    <a:stretch><a:fillRect/></a:stretch></pic:blipFill>
    <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>
    </a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
    """


def docx_package(body, image_path, extra_images=None, width_px=760, height_px=450):
    extra_images = extra_images or []
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
      xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
      xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
      <w:body>{body}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/>
      </w:sectPr></w:body></w:document>"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="21"/></w:rPr>
    <w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
    <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:sz w:val="36"/>
    <w:color w:val="17365D"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/>
    <w:rPr><w:sz w:val="24"/><w:color w:val="526579"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="17365D"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="23"/><w:color w:val="365F91"/></w:rPr></w:style>
    <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/></w:style>
    </w:styles>"""
    types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Default Extension="png" ContentType="image/png"/>
    <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
    <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
    </Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
    </Relationships>"""
    relation_rows = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/strong_scaling.png"/>',
    ]
    for index, (_, media_name, _) in enumerate(extra_images, 3):
        relation_rows.append(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
        )
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relation_rows) + "</Relationships>"
    )
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", types)
        package.writestr("_rels/.rels", root_rels)
        package.writestr("word/document.xml", document)
        package.writestr("word/styles.xml", styles)
        package.writestr("word/_rels/document.xml.rels", document_rels)
        package.write(image_path, "word/media/strong_scaling.png")
        for extra_path, media_name, _ in extra_images:
            package.write(extra_path, f"word/media/{media_name}")


def fmt(value):
    return f"{value:.4f}"


def build_report(rows):
    strong = sorted(
        (row for row in rows if row["category"] == "strong"),
        key=lambda row: row["ranks"],
    )
    weak = sorted(
        (row for row in rows if row["category"] == "weak"),
        key=lambda row: row["ranks"],
    )
    variants = [row for row in rows if row["category"] == "variant"]
    convergence = [row for row in rows if row["category"] == "convergence"]
    if not strong or not weak or not variants or not convergence:
        raise SystemExit("Full results are needed; rerun benchmark.py with --full.")

    def variant_time(ranks, layout, exchange):
        return next(
            row["median_seconds"] for row in variants
            if row["ranks"] == ranks and row["layout"] == layout
            and row["exchange"] == exchange
        )

    graph_path = RESULTS / "strong_scaling.png"
    make_chart(strong, graph_path)
    paragraphs = [
        paragraph("Case Study 1: MPI Heat Diffusion", "Title"),
        paragraph("Weather and heat diffusion simulation | editable interim report", "Subtitle"),
        paragraph(
            "This document contains the completed Case Study 1 section and shared "
            "report framing. Case Studies 3–5 and the final cross-case comparison "
            "will be appended after those studies are complete."
        ),
        paragraph("1. Introduction", "Heading1"),
        paragraph(
            "This study models heat diffusion over an N by N temperature grid. "
            "Each time step uses a five-point stencil and two grids; the outer "
            "boundary remains fixed. The update is u_next[i,j] = u[i,j] + alpha "
            "* (north + south + west + east - 4*u[i,j]), with alpha = 0.2. "
            "The implementation uses MPI only, as requested."
        ),
        paragraph("2. Machine and experimental method", "Heading1"),
        table([
            ["Property", "Recorded environment"],
            ["CPU", "Intel Core i7-8550U @ 1.80 GHz"],
            ["Physical / logical CPUs", "4 cores / 8 logical CPUs"],
            ["Memory", "7.7 GiB reported by the OS"],
            ["OS / compiler / MPI", "Ubuntu 24.04; GCC 13.3; Open MPI 4.1.6"],
            ["Build flags", "-O3 -std=c11 for serial and MPI"],
            ["Thread settings", "OMP_NUM_THREADS=1; OMP_PROC_BIND=close (MPI-only code)"],
            ["Scaling processes", "1, 2, and 4 (capped at physical cores)"],
        ], [2500, 6500]),
        paragraph(
            "Each MPI time is measured inside the program between MPI_Barrier calls; "
            "the rank-zero result is the maximum elapsed time over ranks. Output "
            "gathering and file I/O occur after the timed region. The serial baseline "
            "uses the same optimization level. Every reported point is the median "
            "of three independent runs. MPI was launched with "
            "`mpirun --oversubscribe -np p`; rank placement used the Open MPI default."
        ),
        paragraph("3. Implementation and correctness", "Heading1"),
        paragraph(
            "The serial reference uses contiguous row-major arrays and swaps two "
            "grids per step. The 1-D decomposition partitions rows; the 2-D "
            "decomposition uses MPI_Dims_create and MPI_Cart_create. Both exchange "
            "edge halos with either MPI_Sendrecv or nonblocking sends/receives. "
            "The nonblocking path updates cells at least one cell away from a local "
            "subdomain edge before waiting, then updates the edge cells. MPI_Allreduce "
            "computes the global maximum absolute change every K steps."
        ),
        paragraph(
            "Validation compares the complete row-major final grids against the "
            "serial output with absolute tolerance 1e-12. For N=129 and T=40, the "
            "checker passed all combinations of 1, 2, and 4 ranks, row and 2-D "
            "decomposition, and blocking and nonblocking exchange. The observed "
            "maximum absolute error was 0.000e+00."
        ),
        paragraph("4. Strong scaling", "Heading1"),
        paragraph(
            f"Fixed-size run: N={strong[0]['N']}, T={strong[0]['steps']}, "
            "2-D blocks, nonblocking halos, K=50. Speedup is measured against "
            "the serial executable, not MPI with one rank."
        ),
        table([["p", "MPI median (s)", "Serial baseline (s)", "S(p)", "E(p)"]]
              + [[str(row["ranks"]), fmt(row["median_seconds"]),
                  fmt(row["serial_baseline_seconds"]), fmt(row["speedup_vs_serial"]),
                  f"{100 * row['efficiency']:.1f}%"] for row in strong],
              [800, 1700, 1900, 1000, 1000]),
        image_paragraph("rId2", 760, 450),
        paragraph(
            "Figure 1. Measured strong speedup and ideal linear speedup; x-axis "
            "is MPI process count, y-axis is speedup against the serial baseline."
        , italic=True),
    ]
    below = [row["ranks"] for row in strong
             if row["ranks"] > 1 and row["efficiency"] < 0.70]
    if below:
        first_below = min(below)
        threshold_text = (
            f"Efficiency first falls below 70% at p={first_below} in the measured "
            "curve. The model T_p ~= T_comp/p + T_comm attributes the loss to the "
            "non-scaling overhead floor: halo latency, reductions, MPI-kernel "
            "overhead, and memory-bandwidth contention as ranks share the host."
        )
    else:
        threshold_text = (
            "Efficiency does not fall below 70% at the measured process counts. "
            "The measured curve therefore does not identify a below-70% threshold "
            "on this machine. In T_p ~= T_comp/p + T_comm, the residual "
            "T_comm ~= max(0, T_p - T_serial/p) estimates the communication and "
            "synchronization floor; at higher p, halo and collective overhead will "
            "eventually dominate."
        )
    last = max(strong, key=lambda row: row["ranks"])
    residual = max(0.0, last["median_seconds"] -
                   last["serial_baseline_seconds"] / last["ranks"])
    paragraphs += [
        paragraph(threshold_text),
        paragraph(
            f"At p={last['ranks']}, the simple-model residual estimate is "
            f"T_comm ~= {residual:.4f} s (using the serial time as T_comp)."
        ),
        paragraph("5. Weak scaling", "Heading1"),
        paragraph(
            "For weak scaling, N(p)=round(N0*sqrt(p)) with N0 equal to the strong "
            "scaling grid size, so the global cell count grows approximately "
            "linearly with p and the cells assigned to each rank stay nearly "
            "constant."
        ),
        table([["p", "N", "MPI median (s)", "Time / p=1"]]
              + [[str(row["ranks"]), str(row["N"]), fmt(row["median_seconds"]),
                  fmt(row["median_seconds"] / weak[0]["median_seconds"])]
                 for row in weak],
              [800, 1400, 2200, 2000]),
        paragraph(
                  "Ideal weak scaling keeps the elapsed time ratio near 1.0. Increases "
                  "reflect added halo exchanges, synchronization, and MPI overhead. "
            f"At p={weak[-1]['ranks']}, time is "
            f"{weak[-1]['median_seconds'] / weak[0]['median_seconds']:.2f}x "
            "the p=1 time, rather than flat. This is consistent with shared-host "
            "memory-bandwidth/CPU contention and synchronization overhead; this "
            "experiment does not isolate those contributors."
        ),
        paragraph("6. Surface-to-volume analysis", "Heading1"),
        paragraph(
            "For a balanced 1-D row split, each rank computes approximately N^2/p "
            "owned cells and receives about 2N halo values per step (one neighboring "
            "row from each side; edge ranks receive only N). Its receive-halo to "
            "compute ratio is about 2p/N. For a balanced 2-D block, each rank owns "
            "about N^2/p cells and receives about 2(N/sqrt(p))+2(N/sqrt(p)) = "
            "4N/sqrt(p) halo values (fewer at physical edges); the ratio is about "
            "4sqrt(p)/N. The 2-D ratio becomes smaller than the 1-D ratio for "
            "p>4, reducing communication growth as rank count rises."
        ),
        table([["p", "1-D block (s)", "1-D nonblock (s)",
                "2-D block (s)", "2-D nonblock (s)"]]
              + [[str(row["ranks"]),
                  fmt(variant_time(row["ranks"], "row", "blocking")),
                  fmt(variant_time(row["ranks"], "row", "nonblocking")),
                  fmt(variant_time(row["ranks"], "2d", "blocking")),
                  fmt(variant_time(row["ranks"], "2d", "nonblocking"))]
                 for row in sorted(strong, key=lambda item: item["ranks"])],
              [800, 2000, 2000, 2000, 2000]),
        paragraph(
            "This implementation exchanges two halo layers total as one row on "
            "each north/south side and one column on each east/west side, so its "
            "sent-value count is the same order as the receive count. In the 1-D "
            "layout, the column neighbors are MPI_PROC_NULL and only row halos "
            "carry grid values."
        ),
        paragraph(
            f"At p={strong[-1]['ranks']}, measured row-block and 2-D blocking "
            f"times are {fmt(variant_time(strong[-1]['ranks'], 'row', 'blocking'))} "
            f"and {fmt(variant_time(strong[-1]['ranks'], '2d', 'blocking'))} s. "
            "The theoretical 2-D communication advantage begins beyond p=4; "
            "this machine was intentionally limited to four physical cores, so "
            "the asymptotic advantage is not directly tested. At p=4 the halo "
            "counts are theoretically similar, consistent with the modest measured "
            "difference; timings are also affected by local MPI and memory costs."
        ),
        paragraph("7. Blocking versus nonblocking and overlap", "Heading1"),
    ]
    max_ranks = max(row["ranks"] for row in strong)
    block = next((row for row in variants if row["ranks"] == max_ranks
                  and row["layout"] == "2d" and row["exchange"] == "blocking"), None)
    nonblock = next((row for row in variants if row["ranks"] == max_ranks
                     and row["layout"] == "2d" and row["exchange"] == "nonblocking"), None)
    paragraphs.append(table([
        ["2-D, p", "Exchange", "Total median (s)", "Halo/post or wait medians (s)", "Interior update (s)"],
        [str(max_ranks), "Blocking", fmt(block["median_seconds"]),
         fmt(block["median_comm_post_or_block_seconds"]), "n/a"],
        [str(max_ranks), "Nonblocking", fmt(nonblock["median_seconds"]),
         f"post {fmt(nonblock['median_comm_post_or_block_seconds'])}; "
         f"wait {fmt(nonblock['median_comm_wait_seconds'])}",
         fmt(nonblock["median_interior_seconds"])],
    ], [1000, 1300, 1700, 2500, 1800]))
    estimated_saved = max(
        0.0,
        block["median_comm_post_or_block_seconds"] -
        nonblock["median_comm_post_or_block_seconds"] -
        nonblock["median_comm_wait_seconds"],
    )
    estimate_pct = (
        100 * estimated_saved / block["median_comm_post_or_block_seconds"]
        if block["median_comm_post_or_block_seconds"] > 0 else 0.0
    )
    paragraphs += [
        paragraph(
            f"Using the blocking halo-exchange time as a reference, an approximate "
            f"overlap estimate is max(0, blocking exchange - nonblocking post - "
            f"nonblocking wait) = {estimated_saved:.6f} s, or {estimate_pct:.1f}% "
            "of the blocking exchange time. This estimate is diagnostic rather "
            "than an isolated network measurement: MPI_Wait time depends on MPI "
            "progress, the local shared-memory transport, and message size. The "
            "interior update is intentionally overlapped with the outstanding "
            "requests; if the residual wait remains large, communication progress "
            "or a small interior region limits useful overlap."
        ),
        paragraph(
            f"End-to-end at p={max_ranks}, the nonblocking median is "
            f"{100 * (nonblock['median_seconds'] / block['median_seconds'] - 1):+.1f}% "
            "relative to blocking. Thus the measured overlap did not necessarily "
            "translate into a faster full iteration."
        ),
        paragraph(
            "The communication and interior sub-timers in this table are rank-zero "
            "counters; only total elapsed time is reduced as a maximum over ranks. "
            "Treat the overlap percentage as a local diagnostic, not a global "
            "maximum-rank communication measurement."
        ),
        paragraph("8. Convergence reduction cost", "Heading1"),
    ]
    k1 = next(row for row in convergence
              if row["ranks"] == max_ranks and row["K"] == 1)
    k50 = next(row for row in convergence
               if row["ranks"] == max_ranks and row["K"] == 50)
    overhead_pct = 100 * (k1["median_seconds"] / k50["median_seconds"] - 1)
    paragraphs += [
        table([
            ["p", "K", "Allreduces", "Median time (s)"],
            [str(max_ranks), "1", str(k1["steps"]), fmt(k1["median_seconds"])],
            [str(max_ranks), "50", str(k50["steps"] // 50), fmt(k50["median_seconds"])],
        ], [1000, 1200, 2000, 3000]),
        paragraph(
            f"At p={max_ranks}, K=1 took {fmt(k1['median_seconds'])} s and K=50 "
            f"took {fmt(k50['median_seconds'])} s ({overhead_pct:+.1f}% relative "
            "to K=50). K=1 performs a global reduction after every step; K=50 "
            "performs fewer collectives, reducing synchronization and latency cost. "
            "The final convergence metric is the last scheduled check, so the "
            "metric is reported only at that cadence."
        ),
        paragraph("9. Discussion and limitations", "Heading1"),
        paragraph(
            "The tested process counts are limited to the four physical cores; "
            "the machine has eight logical CPUs, but this report does not claim "
            "physical-core speedup beyond four ranks. Results are specific to this "
            "single host and shared-memory MPI transport, so they should not be "
            "generalized to a cluster interconnect. The supplied benchmark script "
            "keeps raw repetitions and medians for reproducibility. The final "
            "assignment report will add the remaining case studies, cross-case "
            "comparison, and team contribution records."
        ),
        paragraph("References", "Heading1"),
        paragraph(
            "[1] Message Passing Interface Forum, MPI: A Message-Passing Interface "
            "Standard, Version 4.1, 2023."
        ),
        paragraph(
            "[2] Open MPI Project, Open MPI 4.1 documentation and implementation "
            "used for the experiments."
        ),
        paragraph(
            "[3] Linux machine information collected with lscpu and /proc/meminfo "
            "on the benchmark host."
        ),
        paragraph(
            "Contribution records: see the accompanying CONTRIBUTIONS.md; team "
            "members must enter actual tasks, hours, and commit identifiers."
        , italic=True),
    ]
    cs2_directory = ROOT.parent / "cs2_kmer_openmp"
    cs2_summary_path = cs2_directory / "results" / "kmer_summary.csv"
    cs2_raw_path = cs2_directory / "results" / "kmer_raw.csv"
    extra_images = []
    if cs2_summary_path.exists() and cs2_raw_path.exists():
        with cs2_summary_path.open(newline="") as stream:
            cs2_rows = list(csv.DictReader(stream))
        with cs2_raw_path.open(newline="") as stream:
            cs2_raw = list(csv.DictReader(stream))
        for row in cs2_rows:
            row["k"] = int(row["k"])
            row["threads"] = int(row["threads"])
            row["median_seconds"] = float(row["median_seconds"])
            row["speedup_vs_serial"] = (
                float(row["speedup_vs_serial"])
                if row["speedup_vs_serial"] else 0.0
            )
            row["efficiency"] = float(row["efficiency"]) if row["efficiency"] else 0.0
            row["throughput_mkmer_s"] = (
                float(row["throughput_mkmer_s"])
                if row["throughput_mkmer_s"] else 0.0
            )
            row["private_scratch_bytes"] = int(row["private_scratch_bytes"])

        def cs2_row(category, k, threads):
            return next(row for row in cs2_rows
                        if row["category"] == category and row["k"] == k
                        and row["threads"] == threads)

        largest_threads = max(
            row["threads"] for row in cs2_rows
            if row["category"] in ("atomic", "private")
        )
        cs2_n = int(next(row["N"] for row in cs2_raw if row["category"] == "serial"))
        first_case = cs2_row("private", 8, 1)
        graph_rows = [
            {
                "ranks": row["threads"],
                "speedup_vs_serial": row["speedup_vs_serial"],
            }
            for row in cs2_rows
            if row["category"] == "private" and row["k"] == 8
        ]
        graph_path_kmer = cs2_directory / "results" / "kmer_speedup.png"
        make_chart(graph_rows, graph_path_kmer, "OMP THREADS")
        extra_images.append((graph_path_kmer, "kmer_speedup.png", 760))
        paragraphs += [
            paragraph(page_break=True),
            paragraph("Case Study 2: OpenMP k-mer Counting", "Title"),
            paragraph("Parallel genomics histogram | measured on the same host", "Subtitle"),
            paragraph(
                f"The experiment counts all L-k+1 windows in a deterministically "
                f"generated L={cs2_n:,} base sequence for k=8 and k=12. DNA bases "
                "are two-bit encoded and the index rolls forward by two bits per "
                "base. The default sequence uses a fixed seed and near-uniform base "
                "frequencies. Timed work includes histogram zeroing, counting, and "
                "private-histogram merging; sequence generation and scratch "
                "allocation are excluded. The build used GCC 13.3 with "
                "-O3 -std=c11 -fopenmp (GNU OpenMP 4.5)."
            ),
            paragraph("Implementation and correctness", "Heading1"),
            paragraph(
                "The serial reference counts each start position once. Version A "
                "intentionally increments shared bins without synchronization; "
                "concurrent read-modify-write operations can overwrite one another, "
                "so counts are lost and C data-race behavior is undefined. Version B "
                "uses an OpenMP atomic update. Version C gives each worker a "
                "zeroed private histogram and merges all bins after counting."
            ),
            paragraph(
                "Contiguous, non-overlapping start-index ranges are assigned to "
                "threads. The DNA sequence itself is shared, so the owner of a "
                "window start reads the following k-1 characters even if those "
                "characters lie beyond its nominal start range. This counts every "
                "boundary-spanning k-mer exactly once. Exact comparison of all "
                "64-bit bins passed for k=8 and k=12, 1/2/4 threads, and lengths "
                "97, 4099, and 100003. The latter lengths do not divide evenly by "
                "the thread counts. The 100-million-base race demonstration is "
                "recorded below and is not included in performance speedup."
            ),
            paragraph("Strong scaling", "Heading1"),
            paragraph(
                f"Fixed input: L={cs2_n:,} bases, OMP_NUM_THREADS in "
                f"1, 2, 4; OMP_PROC_BIND=close; OMP_PLACES=cores. Each point is "
                "the median of three runs. Speedup is against the same -O3 serial "
                "baseline."
            ),
        ]
        scaling_table = [["k", "Version", "p", "Median (s)", "S(p)", "E(p)", "M k-mers/s"]]
        for k in (8, 12):
            scaling_table.append([
                str(k), "Serial", "1",
                fmt(cs2_row("baseline", k, 1)["median_seconds"]), "1.000", "100.0%",
                f"{cs2_row('baseline', k, 1)['throughput_mkmer_s']:.2f}",
            ])
            for category in ("atomic", "private"):
                for threads in sorted({
                    row["threads"] for row in cs2_rows
                    if row["category"] == category and row["k"] == k
                }):
                    item = cs2_row(category, k, threads)
                    scaling_table.append([
                        str(k), category.capitalize(), str(threads),
                        fmt(item["median_seconds"]), f"{item['speedup_vs_serial']:.3f}",
                        f"{100 * item['efficiency']:.1f}%",
                        f"{item['throughput_mkmer_s']:.2f}",
                    ])
        paragraphs.append(table(scaling_table, [550, 1200, 550, 1300, 1000, 1000, 1400]))
        paragraphs += [
            paragraph(
                "The p=1 entries are actual one-thread OpenMP executions; they "
                "are not substituted with the serial time. Small differences from "
                "S(1)=1 arise from distinct implementation paths and timing "
                "variation, and are retained rather than normalized away."
            ),
            image_paragraph("rId3", 760, 450, drawing_id=2),
            paragraph(
                "Figure 2. Version C (private histogram) strong speedup for k=8 "
                "against the serial baseline; ideal linear speedup is shown for reference."
            , italic=True),
            paragraph("Atomic versus private histograms", "Heading1"),
        ]
        compare_table = [[
            "k", "Bins (4^k)", "p", "Atomic (s)", "Private (s)",
            "Private scratch", "Total private histogram memory",
        ]]
        for k in (8, 12):
            atomic = cs2_row("atomic", k, largest_threads)
            private = cs2_row("private", k, largest_threads)
            bins = 4 ** k
            result_hist_bytes = bins * 8
            compare_table.append([
                str(k), f"{bins:,}", str(largest_threads),
                fmt(atomic["median_seconds"]), fmt(private["median_seconds"]),
                f"{private['private_scratch_bytes'] / (1024 ** 2):.1f} MiB",
                f"{(private['private_scratch_bytes'] + result_hist_bytes) / (1024 ** 2):.1f} MiB",
            ])
        atomic_k8 = cs2_row("atomic", 8, largest_threads)
        private_k8 = cs2_row("private", 8, largest_threads)
        atomic_k12 = cs2_row("atomic", 12, largest_threads)
        private_k12 = cs2_row("private", 12, largest_threads)
        winner_k8 = "private" if private_k8["median_seconds"] < atomic_k8["median_seconds"] else "atomic"
        winner_k12 = "private" if private_k12["median_seconds"] < atomic_k12["median_seconds"] else "atomic"
        paragraphs += [
            table(compare_table, [500, 1400, 400, 1000, 1000, 1300, 2000]),
            paragraph(
                f"For k=8, the output histogram is {4**8 * 8 / 1024:.0f} KiB; "
                f"for k=12 it is {4**12 * 8 / (1024**2):.0f} MiB. Version C "
                f"allocates one additional output-sized histogram per thread: "
                f"{first_case['private_scratch_bytes'] / 1024:.0f} KiB scratch "
                f"for k=8 at one thread and "
                f"{cs2_row('private', 12, 1)['private_scratch_bytes'] / (1024**2):.0f} MiB "
                f"for k=12 at one thread. At p={largest_threads}, the private "
                "scratch grows linearly with p, in addition to the merged result "
                "histogram. The k=8 private table is relatively cache-friendly, "
                "while each k=12 private table is far larger than the host's 8 MiB "
                "last-level cache; clearing and merging those dense tables costs "
                "time and memory bandwidth even when most bins are zero. Atomic "
                "updates avoid per-thread dense tables but contend on popular bins "
                "and can cause random cache-line traffic. The measured table, not "
                "a universal rule, determines the crossover for this machine."
            ),
            paragraph(
                f"At p={largest_threads}, {winner_k8} was faster for k=8 "
                f"({min(atomic_k8['median_seconds'], private_k8['median_seconds']):.3f} s "
                f"versus {max(atomic_k8['median_seconds'], private_k8['median_seconds']):.3f} s), "
                f"while {winner_k12} was faster for k=12 "
                f"({min(atomic_k12['median_seconds'], private_k12['median_seconds']):.3f} s "
                f"versus {max(atomic_k12['median_seconds'], private_k12['median_seconds']):.3f} s). "
                "The observed winner is specific to these measurements; dense "
                "zeroing/merging and cache footprint change with k, while atomic "
                "contention changes with both base distribution and thread count."
            ),
            paragraph("Race demonstration", "Heading1"),
        ]
        race_table = [["k", "Threads", "Expected windows", "Observed increments", "Lost increments"]]
        for k in (8, 12):
            race = next(row for row in cs2_raw
                        if row["category"] == "race" and int(row["k"]) == k)
            race_table.append([
                str(k), race["threads"], f"{int(race['expected']):,}",
                f"{int(race['total']):,}",
                f"{int(race['expected']) - int(race['total']):,}",
            ])
        paragraphs += [
            table(race_table, [700, 900, 2000, 2000, 1600]),
            paragraph(
                "The racy result is a single demonstration, not a stable benchmark "
                "or a valid result. Its exact loss can vary by run, compiler, and "
                "thread scheduling; even a run that happens to match is not proof "
                "of correctness because unsynchronized concurrent updates are a data race."
            ),
            paragraph("False sharing", "Heading1"),
        ]
        false_sharing_rows = [[
            "Threads", "Compact counters (s)", "Padded counters (s)",
            "Padding time change", "Counter storage",
        ]]
        for count in sorted({
            row["threads"] for row in cs2_rows
            if row["category"] == "false_sharing_compact"
        }):
            compact = cs2_row("false_sharing_compact", 0, count)
            padded = cs2_row("false_sharing_padded", 0, count)
            change = 100 * (
                1 - padded["median_seconds"] / compact["median_seconds"]
            ) if compact["median_seconds"] else 0.0
            false_sharing_rows.append([
                str(count), fmt(compact["median_seconds"]),
                fmt(padded["median_seconds"]), f"{change:+.1f}%",
                f"{compact['private_scratch_bytes']} B vs "
                f"{padded['private_scratch_bytes']} B",
            ])
        paragraphs += [
            table(false_sharing_rows, [1000, 1800, 1800, 1600, 2200]),
            paragraph(
                "This microbenchmark gives each thread its own volatile counter "
                "(so the counters do not race) and repeatedly updates it. Compact "
                "8-byte counters can occupy the same 64-byte cache line; coherence "
                "then moves the line between cores despite logically independent "
                "counters. The padded representation aligns each counter to a "
                "64-byte line. The measured change above is for this host and "
                "iteration count; padding increases storage and may have no benefit "
                "when only one thread runs."
            ),
            paragraph("Throughput and bottleneck discussion", "Heading1"),
        ]
        fastest = max(
            (row for row in cs2_rows if row["category"] in ("atomic", "private")),
            key=lambda row: row["throughput_mkmer_s"],
        )
        paragraphs += [
            paragraph(
                f"The fastest correct measured configuration processed "
                f"{fastest['throughput_mkmer_s']:.2f} million k-mers/s "
                f"(k={fastest['k']}, {fastest['category']}, "
                f"p={fastest['threads']}). The rolling encoder reads each input "
                "base approximately once across the sequence, about 1 byte per "
                "k-mer asymptotically (plus O(k) initial reads per thread). Each "
                "histogram update also logically reads 8 counter bytes and writes "
                "8 counter bytes, for about 9 logical bytes read and 8 written "
                "per k-mer before cache-line traffic. A cache miss may transfer a "
                "full cache line. For k=8 the dense "
                "histogram is small enough to benefit from cache reuse, although "
                "atomic contention remains possible. For k=12, the 128 MiB "
                "histogram exceeds the LLC, so random update traffic and the "
                "private version's initialization/merge bandwidth dominate. "
                "Scaling flattens when memory bandwidth, atomic/coherence traffic, "
                "and dense merge work outweigh the additional worker compute."
            ),
            paragraph(
                f"From p=2 to p=4, k=8 private throughput changed from "
                f"{cs2_row('private', 8, 2)['throughput_mkmer_s']:.1f} to "
                f"{cs2_row('private', 8, 4)['throughput_mkmer_s']:.1f} million "
                f"k-mers/s ({cs2_row('private', 8, 4)['throughput_mkmer_s'] / cs2_row('private', 8, 2)['throughput_mkmer_s']:.2f}x, "
                "below ideal 2x). For k=12 atomic, the corresponding measured "
                f"gain is {cs2_row('atomic', 12, 4)['throughput_mkmer_s'] / cs2_row('atomic', 12, 2)['throughput_mkmer_s']:.2f}x. "
                "These are the observed flattening points on this four-core host; "
                "the 1-to-2 and 2-to-4 trends differ by k and synchronization strategy."
            ),
            paragraph(
                "The sequence is generated before the timer, so the throughput "
                "measures counting rather than random-number generation. Speedups "
                "are limited to the four physical cores; the host exposes eight "
                "logical CPUs, but no result here is presented as beyond-physical-core "
                "speedup."
            ),
            paragraph(
                "[4] OpenMP Architecture Review Board, OpenMP API Specification, "
                "Version 5.2, 2021."
            ),
        ]
    cs3_directory = ROOT.parent / "cs3_monte_carlo"
    cs3_summary_path = cs3_directory / "results" / "monte_carlo_summary.csv"
    cs3_raw_path = cs3_directory / "results" / "monte_carlo_raw.csv"
    if cs3_summary_path.exists() and cs3_raw_path.exists():
        with cs3_summary_path.open(newline="") as stream:
            mc_rows = list(csv.DictReader(stream))
        with cs3_raw_path.open(newline="") as stream:
            mc_raw = list(csv.DictReader(stream))

        def mc_row(category, threads=None):
            return next(row for row in mc_rows
                        if row["category"] == category and
                        (threads is None or int(row["threads"]) == threads))

        for row in mc_rows:
            for key in ("paths", "threads"):
                row[key] = int(row[key])
            for key in ("median_seconds", "speedup_vs_serial", "efficiency",
                        "estimate", "standard_error", "throughput_mpaths_s",
                        "serial_fraction_amdahl_vs_p1"):
                if row[key] not in ("", None):
                    row[key] = float(row[key])
        for row in mc_raw:
            for key in ("paths", "threads"):
                row[key] = int(row[key])
            for key in ("time", "estimate", "standard_error", "black_scholes"):
                row[key] = float(row[key])

        mc_parallel = sorted(
            (row for row in mc_rows if row["category"] == "parallel"),
            key=lambda row: row["threads"],
        )
        mc_baseline = mc_row("baseline")
        mc_p1 = mc_row("parallel", 1)
        mc_last = mc_parallel[-1]
        mc_same = mc_row("same_seed")
        bs_price = 10.450583572185565
        mc_graph_rows = [
            {"ranks": row["threads"], "speedup_vs_serial": row["speedup_vs_serial"]}
            for row in mc_parallel
        ]
        mc_graph_path = cs3_directory / "results" / "monte_carlo_speedup.png"
        make_chart(mc_graph_rows, mc_graph_path, "OMP THREADS")
        extra_images.append((mc_graph_path, "monte_carlo_speedup.png", 760))
        fixed_time_rows = []
        for count in sorted({
            int(row["threads"]) for row in mc_raw
            if row["category"] == "fixed_time"
        }):
            samples = [
                row for row in mc_raw
                if row["category"] == "fixed_time" and int(row["threads"]) == count
            ]
            median_paths = statistics.median(int(row["paths"]) for row in samples)
            median_se = statistics.median(float(row["standard_error"]) for row in samples)
            median_time = statistics.median(float(row["time"]) for row in samples)
            fixed_time_rows.append({
                "threads": count, "paths": median_paths,
                "standard_error": median_se, "time": median_time,
                "throughput": median_paths / median_time / 1e6,
            })
        ft1 = next(row for row in fixed_time_rows if row["threads"] == 1)
        ft8 = next(row for row in fixed_time_rows if row["threads"] == max(
            row["threads"] for row in fixed_time_rows
        ))
        time_error_ratio = ft1["standard_error"] / ft8["standard_error"]
        speedup_p1 = mc_p1["median_seconds"] / mc_last["median_seconds"]
        amdahl_fraction = (
            (1 / speedup_p1 - 1 / mc_last["threads"]) /
            (1 - 1 / mc_last["threads"])
        ) if mc_last["threads"] > 1 else 0.0
        same_seed_unique_paths = int(mc_same["paths"]) // int(mc_same["threads"])
        same_seed_expected_se_multiplier = int(mc_same["threads"]) ** 0.5
        paragraphs += [
            paragraph(page_break=True),
            paragraph("Case Study 3: Monte Carlo Option Pricing", "Title"),
            paragraph("European call under geometric Brownian motion | OpenMP", "Subtitle"),
            paragraph(
                "This study prices a European call with terminal stock "
                "S_T=S_0 exp((r-sigma^2/2)T + sigma sqrt(T)Z), Z~N(0,1). "
                "The estimate discounts the mean payoff max(S_T-K,0) by exp(-rT). "
                "OpenMP was selected because paths are independent and the available "
                "machine is shared-memory; thread-local RNG streams and Welford "
                "statistics are combined without inter-thread path communication."
            ),
            table([
                ["Parameter", "Value"],
                ["S0", "100"],
                ["Strike K", "100"],
                ["Risk-free rate r", "0.05"],
                ["Volatility sigma", "0.20"],
                ["Maturity T", "1 year"],
                ["Seed / paths", f"20261123 / {mc_baseline['paths']:,}"],
                ["Black-Scholes price", f"{bs_price:.10f}"],
            ], [3000, 6000]),
            paragraph("Implementation, random streams, and reproducibility", "Heading1"),
            paragraph(
                "The serial baseline and parallel solver use the same -O3 C/OpenMP "
                "build. Each worker's xoshiro256** state is initialized using "
                "SplitMix64 from the master seed and a distinct worker seed. Paths "
                "are assigned in contiguous fixed ranges and each thread accumulates "
                "a Welford count, mean, and variance; partial statistics are merged "
                "in thread-index order. This avoids shared rand() state and preserves "
                "the same estimate and standard error for repeated runs with the "
                "same seed and worker count. Results can differ slightly across "
                "worker counts because the stream assignment and floating-point "
                "merge order change."
            ),
            paragraph(
                "Correctness tests reproduced the estimates and standard errors "
                "exactly for repeated runs at 1, 2, 4, and 8 threads. A separate "
                "500,000-path serial estimate was within six measured standard "
                "errors of Black-Scholes. The fixed-work benchmark's final "
                "estimates are compared against the analytic price in the table below."
            ),
            paragraph("Strong scaling and statistical accuracy", "Heading1"),
            paragraph(
                f"Fixed workload: M={mc_baseline['paths']:,} paths; three timed "
                "runs per point. The machine is an Intel Core i7-8550U, four "
                "physical cores/eight logical CPUs, with 7.7 GiB RAM. Process "
                "settings were OMP_PROC_BIND=close and OMP_PLACES=cores. The "
                "eight-thread point uses logical CPUs and is not an eight-physical-core claim."
            ),
        ]
        mc_table = [[
            "p", "Median time (s)", "S(p)", "E(p)", "Estimate",
            "Abs diff", "SE", "M paths/s",
        ]]
        for row in mc_parallel:
            mc_table.append([
                str(row["threads"]), f"{row['median_seconds']:.4f}",
                f"{row['speedup_vs_serial']:.3f}",
                f"{100 * row['efficiency']:.1f}%",
                f"{float(row['estimate']):.8f}",
                f"{abs(float(row['estimate']) - bs_price):.2e}",
                f"{float(row['standard_error']):.3e}",
                f"{float(row['throughput_mpaths_s']):.2f}",
            ])
        paragraphs += [
            table(mc_table, [450, 1300, 700, 850, 1500, 1100, 1300, 1100]),
            image_paragraph("rId4", 760, 450, drawing_id=3),
            paragraph(
                "Figure 3. Fixed-work strong speedup against the serial baseline; "
                "the dashed line is ideal linear speedup."
            , italic=True),
            paragraph(
                f"The measured standard error at fixed time ({ft1['time']:.2f} s "
                f"and {ft8['time']:.2f} s medians) was {ft1['standard_error']:.3e} "
                f"at p=1 and {ft8['standard_error']:.3e} at p={ft8['threads']}. "
                f"That is a {time_error_ratio:.2f}x error reduction for the same "
                "elapsed-time budget, corresponding to approximately "
                f"{time_error_ratio**2:.2f}x as many effective samples because "
                "standard error scales as 1/sqrt(M). Fixed-time throughput was "
                f"{ft1['throughput']:.2f} and {ft8['throughput']:.2f} million paths/s."
            ),
            paragraph("Why the RNG matters", "Heading1"),
        ]
        correlated_estimate = float(mc_same["estimate"])
        correlated_se = float(mc_same["standard_error"])
        paragraphs += [
            table([
                ["Stream setup", "Nominal paths", "Unique paths", "Reported SE", "Actual SE scale"],
                ["Independent streams", f"{mc_same['paths']:,}",
                 f"{mc_same['paths']:,}", f"{float(mc_last['standard_error']):.3e}",
                 f"{float(mc_last['standard_error']):.3e}"],
                ["Same seed per worker", f"{mc_same['paths']:,}",
                 f"{same_seed_unique_paths:,}", f"{correlated_se:.3e}",
                 f"~{correlated_se * same_seed_expected_se_multiplier:.3e}"],
            ], [2100, 1500, 1500, 1500, 1800]),
            paragraph(
                f"In the intentionally flawed same-seed run at p={mc_same['threads']}, "
                f"all workers repeated the same {same_seed_unique_paths:,}-path "
                f"sequence. The nominal {mc_same['paths']:,} samples therefore "
                "contained only one distinct stream. The observed mean was "
                f"{correlated_estimate:.8f}, while the reported SE was "
                f"{correlated_se:.3e}; treating the duplicated paths as independent "
                f"understates the uncertainty by about sqrt(p)={same_seed_expected_se_multiplier:.2f}. "
                "The test compares this reported error with a serial run of the "
                "same number of unique paths. A single shared unsynchronized RNG "
                "would additionally be a data race; adding a lock would serialize "
                "generation. Neither approach is used by the correct solver."
            ),
            paragraph("Statistical and parallel efficiency", "Heading1"),
            paragraph(
                f"For fixed workload, the standard error is computed from the "
                f"sample variance; the analytic price is {bs_price:.8f}. Each row "
                "reports observed accuracy and throughput alongside conventional "
                "speedup. At fixed time, the p=8 result reduces error by the factor "
                "measured above. This is the useful statistical interpretation of "
                "parallel speedup: more independent paths and lower uncertainty per "
                "unit time, not merely a faster evaluation of one random path."
            ),
            paragraph("Amdahl's law and scaling limits", "Heading1"),
            paragraph(
                f"Using the parallel p=1 implementation as the Amdahl baseline, "
                f"p={mc_last['threads']} had speedup {speedup_p1:.3f}; "
                f"f=(1/S-1/p)/(1-1/p) estimates the serial fraction as "
                f"{100 * amdahl_fraction:.2f}%. This estimate is noisy and can be "
                "affected by turbo/thermal changes and run variability. Residual "
                "losses include thread startup, deterministic partial-statistics "
                "merge, RNG setup, transcendental instructions, cache/memory "
                "effects, and hyper-threading contention. The fixed-work speedup "
                "against the separately measured serial baseline is shown in the "
                "table; the Amdahl estimate uses same-binary OpenMP p=1 timing."
            ),
            paragraph(
                f"Although independent paths are theoretically embarrassingly "
                f"parallel, this host measured only {mc_last['speedup_vs_serial']:.2f}x "
                f"speedup at p={mc_last['threads']} (efficiency "
                f"{100 * mc_last['efficiency']:.1f}%), so scaling was not close "
                "to ideal in these runs. The measured result is reported rather "
                "than assuming perfect scaling; the i7-8550U's four physical cores, "
                "eight-thread hyper-threading, frequency behavior, and expensive "
                "log/sqrt/cos/exp work limit throughput."
            ),
            paragraph(
                "The price is reproducible for a fixed seed and worker count, but "
                "not bitwise invariant across worker counts. Different stream "
                "partitioning samples a different finite set of normal variates; "
                "Welford merging also combines floating-point values in a different "
                "grouping. These changes are expected and should remain within "
                "statistical uncertainty."
            ),
            paragraph(
                "[5] F. Black and M. Scholes, “The Pricing of Options and Corporate "
                "Liabilities,” Journal of Political Economy, vol. 81, no. 3, 1973."
            ),
            paragraph(
                "[6] G. Marsaglia and W. W. Tsang, “The Xorshift RNGs,” Journal of "
                "Statistical Software, vol. 8, no. 14, 2003."
            ),
        ]
    body = "".join(paragraphs)
    docx_package(body, graph_path, extra_images)


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = load_rows()
    build_report(rows)
    print(f"Created {OUTPUT}")
    print(f"Created {RESULTS / 'strong_scaling.png'}")


if __name__ == "__main__":
    main()
