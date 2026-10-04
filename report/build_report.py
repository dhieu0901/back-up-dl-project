"""Build the Vietnamese Major Assignment report (Word + PDF) from the Markdown sources in report/.

    python report/build_report.py

1. creates report/reference.docx: pandoc's default styles adjusted to the instructor's template
   (US Letter, margins 1"/1.25", Times New Roman 13 pt, black bold headings, bordered tables,
   page numbers in the footer);
2. concatenates the Markdown parts (cover, table of contents, Part 1, Part 2, references) and
   converts them with pandoc (LaTeX formulas become native Word equations);
3. opens the result in Microsoft Word (COM, if installed) to fill the table of contents and
   export a PDF for checking.
Output: final/Major_Assignment_Report.docx and final/Major_Assignment_Report.pdf
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
OUT_DIR = ROOT / "final"
REFERENCE = REPORT_DIR / "reference.docx"
PARTS = ["00_bia.md", "part1_ly_thuyet.md", "part2_bai_toan_du_lieu_mo_hinh.md", "part2_ket_qua.md",
         "part2_ket_luan.md", "tai_lieu_tham_khao.md"]

FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
PAGE_BREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'
TOC = '''
```{=openxml}
<w:p><w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr><w:r><w:t>MỤC LỤC</w:t></w:r></w:p>
<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>(Mở bằng Word và cập nhật trường để hiện mục lục)</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
```
'''

PARA_STYLES = {  # styleId -> (pPr, rPr) overrides
    "Normal": ('<w:spacing w:before="0" w:after="120" w:line="312" w:lineRule="auto"/><w:jc w:val="both"/>', FONT + '<w:sz w:val="26"/><w:szCs w:val="26"/>'),
    "BodyText": ('<w:spacing w:before="0" w:after="120" w:line="312" w:lineRule="auto"/><w:jc w:val="both"/>', ''),
    "FirstParagraph": ('<w:spacing w:before="0" w:after="120" w:line="312" w:lineRule="auto"/><w:jc w:val="both"/>', ''),
    "Compact": ('<w:spacing w:before="0" w:after="60" w:line="288" w:lineRule="auto"/><w:jc w:val="left"/>', ''),
    "Heading1": ('<w:keepNext/><w:spacing w:before="360" w:after="240"/><w:jc w:val="center"/><w:outlineLvl w:val="0"/>',
                 FONT + '<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="30"/><w:szCs w:val="30"/>'),
    "Heading2": ('<w:keepNext/><w:spacing w:before="240" w:after="120"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/>',
                 FONT + '<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="28"/><w:szCs w:val="28"/>'),
    "Heading3": ('<w:keepNext/><w:spacing w:before="200" w:after="100"/><w:jc w:val="left"/><w:outlineLvl w:val="2"/>',
                 FONT + '<w:b/><w:bCs/><w:i/><w:color w:val="000000"/><w:sz w:val="26"/><w:szCs w:val="26"/>'),
    "ImageCaption": ('<w:spacing w:before="60" w:after="200"/><w:jc w:val="center"/>', FONT + '<w:i/><w:sz w:val="24"/><w:szCs w:val="24"/>'),
    "TableCaption": ('<w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/>', FONT + '<w:i/><w:sz w:val="24"/><w:szCs w:val="24"/>'),
    "CaptionedFigure": ('<w:keepNext/><w:jc w:val="center"/>', ''),
    "Figure": ('<w:keepNext/><w:jc w:val="center"/>', ''),
    "TOCHeading": ('<w:spacing w:before="240" w:after="240"/><w:jc w:val="center"/><w:outlineLvl w:val="9"/>',
                   FONT + '<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="30"/><w:szCs w:val="30"/>'),
}
CUSTOM_STYLES = '''
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Cover"><w:name w:val="Cover"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:before="0" w:after="60" w:line="276" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
<w:rPr>{font}<w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="CoverTitle"><w:name w:val="CoverTitle"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:before="240" w:after="240" w:line="276" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
<w:rPr>{font}<w:b/><w:bCs/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="CoverInfo"><w:name w:val="CoverInfo"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:before="0" w:after="60" w:line="276" w:lineRule="auto"/><w:ind w:left="2880"/><w:jc w:val="left"/></w:pPr>
<w:rPr>{font}<w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>
'''.format(font=FONT)
TABLE_BORDERS = ('<w:tblBorders>' + ''.join(f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                 for side in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')
FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
          '</w:p></w:ftr>')


def set_style(styles, style_id, ppr, rpr):
    m = re.search(rf'<w:style [^>]*w:styleId="{style_id}"[^>]*>.*?</w:style>', styles, flags=re.S)
    if not m:
        return styles
    block = m.group(0)
    for tag, new in (("w:pPr", ppr), ("w:rPr", rpr)):
        if not new:
            continue
        if f"<{tag}>" in block or f"<{tag} " in block:
            block = re.sub(rf"<{tag}>.*?</{tag}>|<{tag}/>", f"<{tag}>{new}</{tag}>", block, count=1, flags=re.S)
        else:
            block = block.replace("</w:style>", f"<{tag}>{new}</{tag}></w:style>")
    return styles.replace(m.group(0), block)


def make_reference():
    tmp = Path(tempfile.mkdtemp())
    default = tmp / "default.docx"
    subprocess.run(["pandoc", "-o", str(default), "--print-default-data-file", "reference.docx"], check=True)
    with zipfile.ZipFile(default) as z:
        files = {n: z.read(n) for n in z.namelist()}
    styles = files["word/styles.xml"].decode("utf-8")
    styles = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
                    f"<w:rPrDefault><w:rPr>{FONT}<w:sz w:val=\"26\"/><w:szCs w:val=\"26\"/><w:lang w:val=\"vi-VN\"/></w:rPr></w:rPrDefault>",
                    styles, flags=re.S)
    for sid, (ppr, rpr) in PARA_STYLES.items():
        styles = set_style(styles, sid, ppr, rpr)
    # bordered tables, header row bold
    styles = re.sub(r'(<w:style [^>]*w:styleId="Table"[^>]*>.*?<w:tblPr>)', r"\1" + TABLE_BORDERS, styles, count=1, flags=re.S)
    styles = styles.replace("</w:styles>", CUSTOM_STYLES + "</w:styles>")
    files["word/styles.xml"] = styles.encode("utf-8")

    doc = files["word/document.xml"].decode("utf-8")
    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooterPage"/>'
            '<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" '
            'w:header="720" w:footer="720" w:gutter="0"/><w:cols w:space="720"/></w:sectPr>')
    doc = re.sub(r"<w:sectPr.*?</w:sectPr>|<w:sectPr[^>]*/>", sect, doc, flags=re.S)
    files["word/document.xml"] = doc.encode("utf-8")
    files["word/footer1.xml"] = FOOTER.encode("utf-8")
    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    if "rIdFooterPage" not in rels:
        rels = rels.replace("</Relationships>", '<Relationship Id="rIdFooterPage" '
                            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
                            'Target="footer1.xml"/></Relationships>')
    files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    ct = files["[Content_Types].xml"].decode("utf-8")
    if "footer1.xml" not in ct:
        ct = ct.replace("</Types>", '<Override PartName="/word/footer1.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    files["[Content_Types].xml"] = ct.encode("utf-8")
    with zipfile.ZipFile(REFERENCE, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    shutil.rmtree(tmp)


def assemble():
    chunks = []
    for i, name in enumerate(PARTS):
        path = REPORT_DIR / name
        if not path.exists():
            print(f"  (skipping missing part {name})")
            continue
        chunks.append(path.read_text(encoding="utf-8"))
        if name == "00_bia.md":
            chunks.append(PAGE_BREAK + TOC + PAGE_BREAK)
        elif i < len(PARTS) - 1:
            chunks.append(PAGE_BREAK)
    return "\n\n".join(chunks)


def word_postprocess(docx, pdf):
    """Update the table of contents and export a PDF with Microsoft Word, if available."""
    ps = f'''
$ErrorActionPreference = "Stop"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
try {{
  $d = $w.Documents.Open("{docx}")
  foreach ($t in $d.TablesOfContents) {{ $t.Update() }}
  $d.Fields.Update() | Out-Null
  $d.Save()
  $d.ExportAsFixedFormat("{pdf}", 17)
  $d.Close()
}} finally {{ $w.Quit() }}
'''
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=600)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  Word post-processing skipped: {exc}")
        return False


def main():
    OUT_DIR.mkdir(exist_ok=True)
    make_reference()
    md = REPORT_DIR / "_report_full.md"
    md.write_text(assemble(), encoding="utf-8")
    docx = OUT_DIR / "Major_Assignment_Report.docx"
    subprocess.run(["pandoc", str(md.name), "-f", "markdown+tex_math_dollars+pipe_tables+raw_attribute",
                    "-o", str(docx), "--reference-doc", str(REFERENCE), "--resource-path", f"{REPORT_DIR}{';' if sys.platform == 'win32' else ':'}{ROOT}"],
                   cwd=REPORT_DIR, check=True)
    print(f"Word document: {docx}")
    if word_postprocess(docx, OUT_DIR / "Major_Assignment_Report.pdf"):
        print(f"PDF (table of contents updated): {OUT_DIR / 'Major_Assignment_Report.pdf'}")


if __name__ == "__main__":
    main()
