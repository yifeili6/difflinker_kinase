from pathlib import Path
import argparse
import html
import json
import zipfile
import xml.etree.ElementTree as ET
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main', 'a': 'http://schemas.openxmlformats.org/drawingml/2006/main', 'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
E = 12700
(WIDTH, HEIGHT) = (420, 300)
PALETTE = {'teal': '007780', 'purple': '6B56A5', 'orange': 'B74725', 'edge': 'A5B5C2', 'text': '183247', 'white': 'FFFFFF'}
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
NAMESPACE = ' '.join(('xmlns:%s="%s"' % p for p in NS.items()))

def unit(x):
    return str(round(x * E))

def shape(node_id, name, x, y, diameter, color, label=None, font_size=24):
    body = ''
    if label is not None:
        body = '<p:txBody><a:bodyPr anchor="ctr" lIns="0" tIns="0" rIns="0" bIns="0"/><a:lstStyle/><a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="%d" b="1"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:latin typeface="Arial"/></a:rPr><a:t>%s</a:t></a:r></a:p></p:txBody>' % (round(font_size * 100), label)
    return '<p:sp><p:nvSpPr><p:cNvPr id="%d" name="%s"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="%s" y="%s"/><a:ext cx="%s" cy="%s"/></a:xfrm><a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr>%s</p:sp>' % (node_id, name, unit(x), unit(y), unit(diameter), unit(diameter), color, body)

def connector(edge_id, left, right, radius, line_width):
    (x1, y1) = (left['x'] + radius, left['y'])
    (x2, y2) = (right['x'] - radius, right['y'])
    flip = ' flipV="1"' if y2 < y1 else ''
    return '<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="%d" name="Connection %s to %s"/><p:cNvCxnSpPr><a:stCxn id="%d" idx="3"/><a:endCxn id="%d" idx="1"/></p:cNvCxnSpPr><p:nvPr/></p:nvCxnSpPr><p:spPr><a:xfrm%s><a:off x="%s" y="%s"/><a:ext cx="%s" cy="%s"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="%s" cap="rnd"><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:prstDash val="solid"/><a:headEnd type="none"/><a:tailEnd type="none"/></a:ln></p:spPr></p:cxnSp>' % (edge_id, left['name'], right['name'], left['id'], right['id'], flip, unit(min(x1, x2)), unit(min(y1, y2)), unit(abs(x2 - x1)), unit(abs(y2 - y1)), unit(line_width), PALETTE['edge'])

def textbox(shape_id, name, text, x, baseline, size, bold=False):
    return '<p:sp><p:nvSpPr><p:cNvPr id="%d" name="%s"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="%s" y="%s"/><a:ext cx="%s" cy="%s"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p><a:pPr marL="0" marR="0"/><a:r><a:rPr lang="en-US" sz="%d" b="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:latin typeface="Arial"/></a:rPr><a:t>%s</a:t></a:r><a:endParaRPr lang="en-US" sz="%d"/></a:p></p:txBody></p:sp>' % (shape_id, name, unit(x), unit(baseline - size * 0.95), unit(WIDTH - x - 8), unit(size * 1.4), round(size * 100), int(bold), PALETTE['text'], html.escape(text), round(size * 100))

def model_nodes(full_panel):
    if full_panel:
        columns = [(218, [116.8, 128, 139.2], 'teal'), (250, [114.56, 123.52, 132.48, 141.44], 'purple'), (282, [121.6, 134.4], 'orange')]
        transform = lambda x, y: ((x - 184) * 3, (y - 88) * 3)
        (radius, line_width) = (2.85 * 3, 0.82 * 3)
    else:
        columns = [(16, [17.7, 31, 44.3], 'teal'), (50, [15.04, 25.68, 36.32, 46.96], 'purple'), (84, [23.4, 38.6], 'orange')]
        transform = lambda x, y: ((x - 50) * 3.8 + 210, (y - 31) * 3.8 + 150)
        (radius, line_width) = (2.85 * 3.8, 0.82 * 3.8)
    (groups, nodes) = ([], [])
    for (column, (x, ys, color)) in enumerate(columns):
        group = []
        for (row, y) in enumerate(ys):
            (xx, yy) = transform(x, y)
            node = {'id': 100 + len(nodes), 'name': 'Node_%d_%d' % (column + 1, row + 1), 'x': xx, 'y': yy, 'color': PALETTE[color]}
            group.append(node)
            nodes.append(node)
        groups.append(group)
    edges = [(a, b) for (first, second) in zip(groups, groups[1:]) for a in first for b in second]
    return (nodes, edges, radius, line_width)

def slide(full_panel):
    (nodes, edges, radius, line_width) = model_nodes(full_panel)
    objects = [connector(10 + i, a, b, radius, line_width) for (i, (a, b)) in enumerate(edges)]
    objects += [shape(n['id'], n['name'], n['x'] - radius, n['y'] - radius, 2 * radius, n['color']) for n in nodes]
    if full_panel:
        objects += [shape(205, 'Panel_number', 6, 0, 42, PALETTE['purple'], '3', 24), textbox(201, 'Panel_title', 'Fine-tune DiffLinker', 63, 30, 27, True), textbox(202, 'Pretrained_checkpoint_label', 'Binding MOAD checkpoint', 15, 219, 22.8), textbox(203, 'Fine_tuning_label', 'KLIFS training; validation selection', 15, 252, 22.8)]
    tree = '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    return XML + '<p:sld %s><p:cSld name="%s"><p:spTree>%s%s</p:spTree></p:cSld>' % (NAMESPACE, 'Panel 3 complete' if full_panel else 'Panel 3 model symbol', tree, ''.join(objects)) + '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'

def svg(full_panel):
    (nodes, edges, radius, width) = model_nodes(full_panel)
    items = ['<svg xmlns="http://www.w3.org/2000/svg" width="420pt" height="300pt" viewBox="0 0 420 300">', '<title>Editable schematic for Figure 1 Option A Panel 3</title>']
    for (a, b) in edges:
        items.append('<line x1="%.4f" y1="%.4f" x2="%.4f" y2="%.4f" stroke="#%s" stroke-width="%.4f" stroke-linecap="round"/>' % (a['x'] + radius, a['y'], b['x'] - radius, b['y'], PALETTE['edge'], width))
    for node in nodes:
        items.append('<circle cx="%.4f" cy="%.4f" r="%.4f" fill="#%s"/>' % (node['x'], node['y'], radius, node['color']))
    if full_panel:
        items.append('<circle cx="27" cy="21" r="21" fill="#6B56A5"/><text x="27" y="29.1" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="bold" text-anchor="middle" fill="white">3</text>')
        for (text, x, baseline, size, weight) in [('Fine-tune DiffLinker', 63, 30, 27, 'bold'), ('Binding MOAD checkpoint', 15, 219, 22.8, 'normal'), ('KLIFS training; validation selection', 15, 252, 22.8, 'normal')]:
            items.append('<text x="%s" y="%s" font-family="Arial, Helvetica, sans-serif" font-size="%s" font-weight="%s" fill="#183247">%s</text>' % (x, baseline, size, weight, html.escape(text)))
    items.append('</svg>')
    return '\n'.join(items)

def rels(items):
    return XML + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/%s" Target="%s"/>' % item for item in items)) + '</Relationships>'

def package():
    parts = {}
    overrides = [('ppt/presentation.xml', 'presentationml.presentation.main'), ('ppt/slideMasters/slideMaster1.xml', 'presentationml.slideMaster'), ('ppt/slideLayouts/slideLayout1.xml', 'presentationml.slideLayout'), ('ppt/theme/theme1.xml', 'theme'), ('ppt/slides/slide1.xml', 'presentationml.slide'), ('ppt/slides/slide2.xml', 'presentationml.slide')]
    parts['[Content_Types].xml'] = XML + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' + ''.join(('<Override PartName="/%s" ContentType="application/vnd.openxmlformats-officedocument.%s+xml"/>' % item for item in overrides)) + '</Types>'
    parts['_rels/.rels'] = rels([('rId1', 'officeDocument', 'ppt/presentation.xml')])
    parts['ppt/presentation.xml'] = XML + '<p:presentation %s><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/></p:sldIdLst><p:sldSz cx="%s" cy="%s" type="custom"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/></p:presentation>' % (NAMESPACE, unit(WIDTH), unit(HEIGHT))
    parts['ppt/_rels/presentation.xml.rels'] = rels([('rId1', 'slideMaster', 'slideMasters/slideMaster1.xml'), ('rId2', 'slide', 'slides/slide1.xml'), ('rId3', 'slide', 'slides/slide2.xml')])
    group = '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>'
    cmap = '<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
    parts['ppt/slideMasters/slideMaster1.xml'] = XML + '<p:sldMaster %s><p:cSld>%s</p:cSld>%s<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>' % (NAMESPACE, group, cmap)
    parts['ppt/slideMasters/_rels/slideMaster1.xml.rels'] = rels([('rId1', 'slideLayout', '../slideLayouts/slideLayout1.xml'), ('rId2', 'theme', '../theme/theme1.xml')])
    parts['ppt/slideLayouts/slideLayout1.xml'] = XML + '<p:sldLayout %s type="blank" preserve="1"><p:cSld name="Blank">%s</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>' % (NAMESPACE, group)
    parts['ppt/slideLayouts/_rels/slideLayout1.xml.rels'] = rels([('rId1', 'slideMaster', '../slideMasters/slideMaster1.xml')])
    scheme = [('dk1', '183247'), ('lt1', 'FFFFFF'), ('dk2', '516576'), ('lt2', 'F7F8FA'), ('accent1', '007780'), ('accent2', '6B56A5'), ('accent3', 'B74725'), ('accent4', 'A5B5C2'), ('accent5', 'CAD5DC'), ('accent6', 'DDE4E9'), ('hlink', '007780'), ('folHlink', '6B56A5')]
    parts['ppt/theme/theme1.xml'] = XML + '<a:theme xmlns:a="%s" name="Kinase Option A"><a:themeElements><a:clrScheme name="Option A">%s</a:clrScheme><a:fontScheme name="Arial"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Plain"><a:fillStyleLst>%s</a:fillStyleLst><a:lnStyleLst>%s</a:lnStyleLst><a:effectStyleLst>%s</a:effectStyleLst><a:bgFillStyleLst>%s</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>' % (NS['a'], ''.join(('<a:%s><a:srgbClr val="%s"/></a:%s>' % (tag, color, tag) for (tag, color) in scheme)), '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3, '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>' * 3, '<a:effectStyle><a:effectLst/></a:effectStyle>' * 3, '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3)
    for (i, full) in enumerate([False, True], 1):
        parts['ppt/slides/slide%d.xml' % i] = slide(full)
        parts['ppt/slides/_rels/slide%d.xml.rels' % i] = rels([('rId1', 'slideLayout', '../slideLayouts/slideLayout1.xml')])
    return parts

def main():
    target = Path(__file__).resolve().parent / 'output'
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(target / 'panel3_editable.pptx', 'w', zipfile.ZIP_DEFLATED) as archive:
        for (name, value) in package().items():
            archive.writestr(name, value)
    for (filename, full) in [('panel3_symbol.svg', False), ('panel3_complete.svg', True)]:
        (target / filename).write_text(svg(full))
if __name__ == '__main__':
    main()
