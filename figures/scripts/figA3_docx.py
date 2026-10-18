"""Editable Word version of Fig. A3, built from the same source as the PNG (figA3_flowchart.html).

Each box of the flowchart becomes a one-cell bordered table (the calibration box is shaded) and the boxes are
joined by arrow paragraphs. Typesetting follows the HTML source: <i> gives italic, <sub>/<sup> give subscript and
superscript, the box headers are bold, and all other text is roman.

Output: figures/output/FigA3_ModelFlowchart.docx
Requires: pip install python-docx
"""
import os, re, datetime
from lxml import html as LH
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'figA3_flowchart.html')
FONT = 'Times New Roman'
SIZE = {'title': 12, 'subtitle': 9, 'hdr': 10, 'eq': 9, 'note': 8, 'arrow': 9, 'end': 1}
BOX_WIDTH = Inches(7.0)
WS = re.compile(r'[ \t\r\n]+')        # ordinary white space only; non-breaking spaces are kept


def segments(el, italic=False):
    """Flatten an HTML element into [text, italic, vertAlign] runs."""
    out = []

    def push(text, it, va):
        if not text:
            return
        text = WS.sub(' ', text)
        if out and out[-1][1] == it and out[-1][2] == va:
            out[-1][0] += text
        else:
            out.append([text, it, va])

    def walk(e, it, va):
        push(e.text, it, va)
        for c in e:
            tag = c.tag.lower() if isinstance(c.tag, str) else ''
            nit = it or tag == 'i'
            nva = 'subscript' if tag == 'sub' else 'superscript' if tag == 'sup' else va
            walk(c, nit, nva)
            push(c.tail, it, va)

    walk(el, italic, None)
    if out:
        out[0][0] = out[0][0].lstrip(' ')
        out[-1][0] = out[-1][0].rstrip(' ')
    return [s for s in out if s[0]]


def set_font(run, size, bold=False, italic=False, va=None):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn('w:eastAsia'), FONT)
    rpr.rFonts.set(qn('w:cs'), FONT)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if va == 'subscript':
        run.font.subscript = True
    elif va == 'superscript':
        run.font.superscript = True


def paragraph(container, segs, kind, bold=False, before=1, after=1):
    p = container.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0
    for text, it, va in segs:
        set_font(p.add_run(text), SIZE[kind], bold=bold, italic=it, va=va)
    return p


def cell_format(cell, shaded):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right'):
        b = OxmlElement('w:' + side)
        for k, v in (('val', 'single'), ('sz', '8'), ('space', '0'), ('color', '000000')):
            b.set(qn('w:' + k), v)
        borders.append(b)
    tcpr.append(borders)
    if shaded:
        shd = OxmlElement('w:shd')
        for k, v in (('val', 'clear'), ('color', 'auto'), ('fill', 'EDEDED')):
            shd.set(qn('w:' + k), v)
        tcpr.append(shd)
    mar = OxmlElement('w:tcMar')
    for side, w in (('top', 40), ('left', 100), ('bottom', 60), ('right', 100)):
        m = OxmlElement('w:' + side)
        m.set(qn('w:w'), str(w)); m.set(qn('w:type'), 'dxa')
        mar.append(m)
    tcpr.append(mar)


def main():
    root = LH.parse(SRC).getroot()
    cls = lambda c: root.xpath('//div[contains(concat(" ", @class, " "), " %s ")]' % c)
    title = segments(cls('title')[0]) if cls('title') else None             # the figure carries no title (AMS style)
    subtitle = segments(cls('subtitle')[0], italic=True) if cls('subtitle') else None
    boxes = []
    for box in cls('box'):
        lines = [(d.get('class').strip(), segments(d)) for d in box if d.get('class', '').strip() in SIZE]
        boxes.append({'shaded': 'shaded' in box.get('class', ''), 'lines': lines})

    doc = Document()
    cp = doc.core_properties
    cp.title, cp.subject = 'Fig. A3. Land surface model calculation framework', 'FLSF energy partitioning'
    cp.author = cp.last_modified_by = cp.comments = cp.keywords = ''
    cp.created = cp.modified = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0, tzinfo=None)
    cp.revision = 1
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ('left_margin', 'right_margin'):
        setattr(sec, side, Inches(0.75))
    sec.top_margin = sec.bottom_margin = Inches(0.5)

    if title:
        paragraph(doc, title, 'title', bold=True, before=0, after=2)
    if subtitle:
        paragraph(doc, subtitle, 'subtitle', before=0, after=6)
    for i, b in enumerate(boxes):
        t = doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        cell = t.cell(0, 0)
        cell.width = BOX_WIDTH
        cell_format(cell, b['shaded'])
        cell.paragraphs[0]._element.getparent().remove(cell.paragraphs[0]._element)
        for kind, segs in b['lines']:
            paragraph(cell, segs, kind, bold=(kind == 'hdr'), before=1 if kind == 'hdr' else 0.5, after=0.5)
        if i < len(boxes) - 1:
            paragraph(doc, [['↓', False, None]], 'arrow', before=0, after=0)
    end = paragraph(doc, [], 'end', before=0, after=0)   # the document body ends with a (1-pt) paragraph
    mark = OxmlElement('w:rPr'); sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '2'); mark.append(sz)
    end._p.get_or_add_pPr().append(mark)

    zoom = doc.settings.element.find(qn('w:zoom'))
    if zoom is not None and zoom.get(qn('w:percent')) is None:
        zoom.set(qn('w:percent'), '100')

    out = S.paths.fig('FigA3_ModelFlowchart.docx')
    doc.save(out)
    print('->', 'figures/output/FigA3_ModelFlowchart.docx', '(%d boxes)' % len(boxes))


if __name__ == '__main__':
    main()
